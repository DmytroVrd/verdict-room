from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from band import Agent
from band.core.protocols import AgentToolsProtocol
from band.core.simple_adapter import SimpleAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.common.config import PROJECT_ROOT, AgentCredentials, Settings
from src.common.llm import make_llm

logger = logging.getLogger("ARBITER")


class DebatePhase(StrEnum):
    IDLE = "IDLE"
    CHAIN_RESEARCHER = "CHAIN_RESEARCHER"
    RESEARCH = "RESEARCH"
    SCOUTING = "SCOUTING"
    ROUND_1_CRITIC = "ROUND_1_CRITIC"
    COMPLIANCE = "COMPLIANCE"
    ROUND_1_ADVOCATE = "ROUND_1_ADVOCATE"
    ROUND_2_CRITIC = "ROUND_2_CRITIC"
    ROUND_2_ADVOCATE = "ROUND_2_ADVOCATE"


@dataclass
class DebateState:
    phase: DebatePhase = DebatePhase.IDLE
    case: str = ""
    initiator: str = ""
    compliance_requested: bool = False
    compliance_reviewed: bool = False
    missing_agents: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)


class ArbiterAdapter(SimpleAdapter[HistoryProvider]):
    def __init__(
        self,
        llm: BaseChatModel,
        turn_timeout: float | None = None,
        compliance_identifier: str = "Compliance",
    ) -> None:
        super().__init__()
        self.llm = llm
        self.turn_timeout = turn_timeout
        self.compliance_identifier = compliance_identifier
        self.states: dict[str, DebateState] = {}
        self.timeout_tasks: dict[str, asyncio.Task[None]] = {}
        self.last_messages: dict[str, PlatformMessage] = {}
        self.histories: dict[str, HistoryProvider] = {}

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: HistoryProvider,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        del participants_msg, contacts_msg, is_session_bootstrap
        state = self.states.setdefault(room_id, DebateState())
        previous = state.phase
        logger.info(
            "%s received from %s", state.phase, msg.sender_name or msg.sender_type
        )

        if not self._is_expected_sender(state.phase, msg):
            logger.warning(
                "Ignoring unexpected sender %s while waiting in %s",
                msg.sender_name or msg.sender_type,
                state.phase,
            )
            return

        self._cancel_timeout(room_id)
        self.last_messages[room_id] = msg
        self.histories[room_id] = history
        state.transcript.append(
            f"[{msg.sender_name or msg.sender_type}]: {msg.content}"
        )

        if state.phase == DebatePhase.IDLE:
            normalized = msg.content.lower()
            if "ping" in normalized:
                initiator = self._mention_for(msg)
                await tools.send_message(
                    "⚖️ PONG — Arbiter is online and ready for a purchase case.",
                    mentions=[initiator],
                )
                return
            if "researcher" in normalized and any(
                word in normalized for word in ("привіт", "hello", "hi", "greet")
            ):
                state.initiator = msg.sender_name or msg.sender_id
                state.phase = DebatePhase.CHAIN_RESEARCHER
                await tools.send_message(
                    "⚖️ CONNECTION TEST\n"
                    "@Researcher reply with one short greeting, then return control to @Arbiter.\n"
                    "HANDOFF: @Researcher | STATE: CHAIN_TEST | REQUEST: greeting",
                    mentions=["@Researcher"],
                )
                self._schedule_timeout(room_id, tools)
                return
            state.case = msg.content
            state.initiator = msg.sender_name or msg.sender_id
            state.phase = DebatePhase.RESEARCH
            await tools.send_message(
                "⚖️ CASE OPENED\n"
                f"- Request: {self._trim(msg.content, 600)}\n"
                "- Process: research → alternatives → two debate rounds → verdict\n"
                "@Researcher gather pricing, capabilities, company facts, and user complaints. "
                "Cite real URLs and return control to @Arbiter.\n"
                "HANDOFF: @Researcher | STATE: RESEARCH | REQUEST: sourced fact list",
                mentions=["@Researcher"],
            )
        elif state.phase == DebatePhase.CHAIN_RESEARCHER:
            initiator = (
                state.initiator
                if state.initiator.startswith("@")
                else f"@{state.initiator}"
            )
            await tools.send_message(
                "⚖️ CONNECTION TEST PASSED — Arbiter → Researcher → Arbiter.",
                mentions=[initiator],
            )
            self.states[room_id] = DebateState()
        elif state.phase == DebatePhase.RESEARCH:
            state.phase = DebatePhase.SCOUTING
            await tools.send_message(
                "⚖️ EVIDENCE RECEIVED\n"
                f"@Scout compare 2-3 credible alternatives for this case: {self._trim(state.case, 300)}. "
                "Use price, fit, and key trade-offs. Return control to @Arbiter.\n"
                "HANDOFF: @Scout | STATE: SCOUTING | REQUEST: alternative comparison",
                mentions=["@Scout"],
            )
        elif state.phase == DebatePhase.SCOUTING:
            state.phase = DebatePhase.ROUND_1_CRITIC
            await tools.send_message(
                "⚖️ ROUND 1: CHALLENGE\n"
                "@Critic present the strongest case AGAINST the purchase using only room evidence. "
                "Call out hidden cost, lock-in, migration, security, and privacy. "
                "Use the exact marker COMPLIANCE CONCERN when specialist review is needed. "
                "Return control to @Arbiter.\n"
                "HANDOFF: @Critic | STATE: ROUND_1 | REQUEST: strongest objections",
                mentions=["@Critic"],
            )
        elif state.phase == DebatePhase.ROUND_1_CRITIC:
            if "COMPLIANCE CONCERN" in msg.content.upper():
                state.compliance_requested = True
                if await self._recruit_compliance(tools):
                    state.phase = DebatePhase.COMPLIANCE
                    await tools.send_message(
                        "⚖️ SPECIALIST REVIEW\n"
                        f"@Compliance assess data residency, GDPR/privacy, certifications, and contractual risk "
                        f"for this case: {self._trim(state.case, 250)}. Return control to @Arbiter.\n"
                        "HANDOFF: @Compliance | STATE: RECRUIT | REQUEST: compliance opinion",
                        mentions=["@Compliance"],
                    )
                else:
                    state.phase = DebatePhase.ROUND_1_ADVOCATE
                    await self._ask_advocate(tools)
            else:
                state.phase = DebatePhase.ROUND_1_ADVOCATE
                await self._ask_advocate(tools)
        elif state.phase == DebatePhase.COMPLIANCE:
            state.compliance_reviewed = True
            state.phase = DebatePhase.ROUND_1_ADVOCATE
            await self._ask_advocate(tools)
        elif state.phase == DebatePhase.ROUND_1_ADVOCATE:
            state.phase = DebatePhase.ROUND_2_CRITIC
            await tools.send_message(
                "⚖️ ROUND 2: REBUTTAL\n"
                "@Critic strengthen only the objections that remain unresolved. Do not repeat closed points. "
                "Return control to @Arbiter.\n"
                "HANDOFF: @Critic | STATE: ROUND_2 | REQUEST: unresolved objections",
                mentions=["@Critic"],
            )
        elif state.phase == DebatePhase.ROUND_2_CRITIC:
            state.phase = DebatePhase.ROUND_2_ADVOCATE
            await tools.send_message(
                "⚖️ ROUND 2: CLOSING\n"
                "@Advocate give a concise closing statement. Address the remaining objections point by point "
                "and state any purchase conditions. Return control to @Arbiter.\n"
                "HANDOFF: @Advocate | STATE: ROUND_2 | REQUEST: final defense",
                mentions=["@Advocate"],
            )
        elif state.phase == DebatePhase.ROUND_2_ADVOCATE:
            await self._finalize_verdict(room_id, state, msg, history, tools)

        logger.info("%s -> %s", previous, self.states[room_id].phase)
        self._schedule_timeout(room_id, tools)

    async def _ask_advocate(self, tools: AgentToolsProtocol) -> None:
        await tools.send_message(
            "⚖️ ROUND 1: DEFENSE\n"
            "@Advocate answer the Critic point by point using room evidence only. "
            "Concede unsupported claims and propose concrete safeguards. Return control to @Arbiter.\n"
            "HANDOFF: @Advocate | STATE: ROUND_1 | REQUEST: point-by-point defense",
            mentions=["@Advocate"],
        )

    async def _recruit_compliance(self, tools: AgentToolsProtocol) -> bool:
        try:
            await tools.lookup_peers()
            await tools.add_participant(self.compliance_identifier)
            await tools.get_participants()
            await tools.send_event(
                content="Compliance specialist dynamically recruited.",
                message_type="task",
                metadata={"kind": "participant_recruited", "role": "compliance"},
            )
            return True
        except Exception:
            logger.exception(
                "Compliance recruitment failed; continuing with explicit mention"
            )
            return False

    async def _make_verdict(
        self,
        state: DebateState,
        msg: PlatformMessage,
        history: HistoryProvider,
    ) -> dict[str, Any]:
        hydrated_history = "\n".join(
            f"[{item.get('sender_name') or item.get('sender_type', 'Unknown')}]: "
            f"{item.get('content', '')}"
            for item in history.raw[-30:]
        )
        transcript_parts = [hydrated_history, *state.transcript]
        transcript = "\n".join(part for part in transcript_parts if part)
        prompt = f"""
You are the neutral Arbiter. Produce ONLY valid JSON for the purchase case below.
Scores are integers 0-25 and total must equal their sum.
recommendation must be BUY, BUY_WITH_CONDITIONS, or AVOID.
Keep rationale to 3-5 short strings and dissent to one strong opposing argument.

Schema:
{{
  "case": "...",
  "scores": {{
    "value_for_money": 0,
    "capability_fit": 0,
    "risk_profile": 0,
    "alternatives": 0
  }},
  "total": 0,
  "recommendation": "BUY_WITH_CONDITIONS",
  "rationale": ["..."],
  "conditions": ["..."],
  "dissent": "...",
  "compliance_reviewed": true
}}

Case: {state.case}
Transcript:
{transcript}
""".strip()
        response = await self.llm.ainvoke(prompt)
        text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        try:
            verdict = json.loads(self._extract_json(text))
            verdict = self._normalize_verdict(verdict, state)
            verdict["compliance_reviewed"] = state.compliance_reviewed
            verdict["evidence_gaps"] = state.missing_agents
            return verdict
        except Exception:
            logger.exception("Invalid verdict JSON; using safe fallback")
            return {
                "case": state.case,
                "scores": {
                    "value_for_money": 15,
                    "capability_fit": 15,
                    "risk_profile": 10,
                    "alternatives": 10,
                },
                "total": 50,
                "recommendation": "BUY_WITH_CONDITIONS",
                "rationale": [
                    "Automated scorecard parsing failed; review the room evidence."
                ],
                "conditions": ["Human review required before purchase."],
                "dissent": "The evidence may support a different decision.",
                "compliance_reviewed": state.compliance_reviewed,
                "evidence_gaps": state.missing_agents,
            }

    @staticmethod
    def _extract_json(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
        return text[start : end + 1]

    @staticmethod
    def _format_verdict(verdict: dict[str, Any]) -> str:
        scores = verdict["scores"]
        rationale = "\n".join(f"- {item}" for item in verdict.get("rationale", []))
        conditions = "\n".join(f"- {item}" for item in verdict.get("conditions", []))
        scorecard = json.dumps(verdict, ensure_ascii=False, indent=2)
        return (
            "⚖️ FINAL VERDICT\n"
            f"Recommendation: **{verdict['recommendation']}**\n"
            f"Score: **{verdict['total']}/100**\n"
            f"- Value for money: {scores['value_for_money']}/25\n"
            f"- Capability fit: {scores['capability_fit']}/25\n"
            f"- Risk profile: {scores['risk_profile']}/25\n"
            f"- Alternatives: {scores['alternatives']}/25\n"
            f"Rationale:\n{rationale}\n"
            f"Conditions:\n{conditions or '- None'}\n"
            f"Dissent: {verdict.get('dissent', 'None')}\n"
            f"```json\n{scorecard}\n```"
        )

    @staticmethod
    def _normalize_verdict(
        verdict: dict[str, Any], state: DebateState
    ) -> dict[str, Any]:
        raw_scores = verdict.get("scores", {})
        keys = ("value_for_money", "capability_fit", "risk_profile", "alternatives")
        scores = {key: max(0, min(25, int(raw_scores.get(key, 0)))) for key in keys}
        recommendation = str(verdict.get("recommendation", "BUY_WITH_CONDITIONS"))
        if recommendation not in {"BUY", "BUY_WITH_CONDITIONS", "AVOID"}:
            recommendation = "BUY_WITH_CONDITIONS"
        return {
            "case": str(verdict.get("case") or state.case),
            "scores": scores,
            "total": sum(scores.values()),
            "recommendation": recommendation,
            "rationale": [str(item) for item in verdict.get("rationale", [])][:5],
            "conditions": [str(item) for item in verdict.get("conditions", [])][:5],
            "dissent": str(verdict.get("dissent", "None")),
            "compliance_reviewed": state.compliance_reviewed,
            "evidence_gaps": list(state.missing_agents),
        }

    @staticmethod
    def _save_verdict(room_id: str, verdict: dict[str, Any]) -> Path:
        target_dir = PROJECT_ROOT / "data" / "verdicts"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{room_id}.json"
        target.write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target

    @staticmethod
    def _trim(value: str, limit: int) -> str:
        value = " ".join(value.split())
        return value if len(value) <= limit else value[: limit - 1] + "…"

    @staticmethod
    def _is_expected_sender(phase: DebatePhase, msg: PlatformMessage) -> bool:
        expected = {
            DebatePhase.CHAIN_RESEARCHER: "researcher",
            DebatePhase.RESEARCH: "researcher",
            DebatePhase.SCOUTING: "scout",
            DebatePhase.ROUND_1_CRITIC: "critic",
            DebatePhase.COMPLIANCE: "compliance",
            DebatePhase.ROUND_1_ADVOCATE: "advocate",
            DebatePhase.ROUND_2_CRITIC: "critic",
            DebatePhase.ROUND_2_ADVOCATE: "advocate",
        }.get(phase)
        if expected is None:
            return True
        return expected in (msg.sender_name or "").strip().lower()

    @staticmethod
    def _mention_for(msg: PlatformMessage) -> str:
        value = msg.sender_name or msg.sender_id
        return value if value.startswith("@") else f"@{value}"

    @staticmethod
    def _expected_handle(phase: DebatePhase) -> str:
        return {
            DebatePhase.CHAIN_RESEARCHER: "@Researcher",
            DebatePhase.RESEARCH: "@Researcher",
            DebatePhase.SCOUTING: "@Scout",
            DebatePhase.ROUND_1_CRITIC: "@Critic",
            DebatePhase.COMPLIANCE: "@Compliance",
            DebatePhase.ROUND_1_ADVOCATE: "@Advocate",
            DebatePhase.ROUND_2_CRITIC: "@Critic",
            DebatePhase.ROUND_2_ADVOCATE: "@Advocate",
        }[phase]

    async def _finalize_verdict(
        self,
        room_id: str,
        state: DebateState,
        msg: PlatformMessage,
        history: HistoryProvider,
        tools: AgentToolsProtocol,
    ) -> None:
        verdict = await self._make_verdict(state, msg, history)
        initiator = (
            state.initiator
            if state.initiator.startswith("@")
            else f"@{state.initiator}"
        )
        await tools.send_message(
            f"{self._format_verdict(verdict)}\n{initiator}",
            mentions=[initiator],
        )
        await tools.send_event(
            content=json.dumps(verdict, ensure_ascii=False),
            message_type="task",
            metadata={"kind": "verdict", "room_id": room_id},
        )
        self._save_verdict(room_id, verdict)
        self.states[room_id] = DebateState()
        self._cancel_timeout(room_id)

    def _cancel_timeout(self, room_id: str) -> None:
        task = self.timeout_tasks.pop(room_id, None)
        if task and task is not asyncio.current_task():
            task.cancel()

    def _schedule_timeout(self, room_id: str, tools: AgentToolsProtocol) -> None:
        if not self.turn_timeout or self.states[room_id].phase == DebatePhase.IDLE:
            return
        self._cancel_timeout(room_id)
        self.timeout_tasks[room_id] = asyncio.create_task(
            self._watch_turn(room_id, self.states[room_id].phase, tools)
        )

    async def _watch_turn(
        self,
        room_id: str,
        phase: DebatePhase,
        tools: AgentToolsProtocol,
    ) -> None:
        try:
            await asyncio.sleep(self.turn_timeout or 0)
            if self.states[room_id].phase != phase:
                return
            expected = self._expected_handle(phase)
            await tools.send_message(
                f"⚖️ TIMEOUT REMINDER\n{expected} please complete the pending turn. "
                "This is the only retry.",
                mentions=[expected],
            )
            await asyncio.sleep(self.turn_timeout or 0)
            if self.states[room_id].phase != phase:
                return
            self.states[room_id].missing_agents.append(expected.lstrip("@"))
            logger.warning(
                "%s timed out twice in room %s; continuing", expected, room_id
            )
            await self._advance_after_timeout(room_id, tools)
        except asyncio.CancelledError:
            return

    async def _advance_after_timeout(
        self, room_id: str, tools: AgentToolsProtocol
    ) -> None:
        state = self.states[room_id]
        phase = state.phase
        if phase in {DebatePhase.CHAIN_RESEARCHER}:
            initiator = (
                state.initiator
                if state.initiator.startswith("@")
                else f"@{state.initiator}"
            )
            await tools.send_message(
                "⚖️ CONNECTION TEST FAILED — Researcher did not answer.",
                mentions=[initiator],
            )
            self.states[room_id] = DebateState()
            return
        if phase == DebatePhase.RESEARCH:
            state.phase = DebatePhase.SCOUTING
            await tools.send_message(
                "⚖️ RESEARCH TIMED OUT\n"
                "@Scout compare 2-3 alternatives using available room evidence.\n"
                "HANDOFF: @Scout | STATE: SCOUTING | REQUEST: alternative comparison",
                mentions=["@Scout"],
            )
        elif phase == DebatePhase.SCOUTING:
            state.phase = DebatePhase.ROUND_1_CRITIC
            await tools.send_message(
                "⚖️ SCOUTING TIMED OUT\n"
                "@Critic present the strongest evidence-based objections.\n"
                "HANDOFF: @Critic | STATE: ROUND_1 | REQUEST: strongest objections",
                mentions=["@Critic"],
            )
        elif phase in {DebatePhase.ROUND_1_CRITIC, DebatePhase.COMPLIANCE}:
            state.phase = DebatePhase.ROUND_1_ADVOCATE
            await self._ask_advocate(tools)
        elif phase == DebatePhase.ROUND_1_ADVOCATE:
            state.phase = DebatePhase.ROUND_2_CRITIC
            await tools.send_message(
                "⚖️ DEFENSE TIMED OUT\n"
                "@Critic identify only unresolved objections for the closing round.\n"
                "HANDOFF: @Critic | STATE: ROUND_2 | REQUEST: unresolved objections",
                mentions=["@Critic"],
            )
        elif phase == DebatePhase.ROUND_2_CRITIC:
            state.phase = DebatePhase.ROUND_2_ADVOCATE
            await tools.send_message(
                "⚖️ REBUTTAL TIMED OUT\n"
                "@Advocate give the final defense and purchase conditions.\n"
                "HANDOFF: @Advocate | STATE: ROUND_2 | REQUEST: final defense",
                mentions=["@Advocate"],
            )
        elif phase == DebatePhase.ROUND_2_ADVOCATE:
            msg = self.last_messages[room_id]
            history = self.histories.get(room_id, HistoryProvider(raw=[]))
            await self._finalize_verdict(room_id, state, msg, history, tools)
            return
        self._schedule_timeout(room_id, tools)


def build_arbiter_agent(
    credentials: AgentCredentials,
    settings: Settings,
    compliance_identifier: str = "Compliance",
) -> Agent:
    return Agent.create(
        adapter=ArbiterAdapter(
            make_llm("arbiter", settings),
            turn_timeout=settings.debate_turn_timeout,
            compliance_identifier=compliance_identifier,
        ),
        agent_id=credentials.agent_id,
        api_key=credentials.api_key,
        rest_url=settings.band_rest_url,
        ws_url=settings.band_ws_url,
    )
