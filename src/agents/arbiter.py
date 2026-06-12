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
    research_evidence: str = ""
    scout_evidence: str = ""
    critic_round_1: str = ""
    compliance_evidence: str = ""
    advocate_round_1: str = ""
    critic_round_2: str = ""
    pending_message: str = ""
    pending_mentions: list[str] = field(default_factory=list)


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
                await self._delegate(
                    state,
                    tools,
                    "⚖️ CONNECTION TEST\n"
                    "@Researcher reply with one short greeting, then return control to @Arbiter.\n"
                    "HANDOFF: @Researcher | STATE: CHAIN_TEST | REQUEST: greeting",
                    ["@Researcher"],
                )
                self._schedule_timeout(room_id, tools)
                return
            state.case = self._clean_case(msg)
            state.initiator = msg.sender_name or msg.sender_id
            state.phase = DebatePhase.RESEARCH
            await self._delegate(
                state,
                tools,
                "⚖️ CASE OPENED\n"
                f"CASE BRIEF: {self._brief(state.case, 55)}\n"
                "- Process: research → alternatives → two debate rounds → verdict\n"
                "@Researcher gather pricing, capabilities, company facts, and user complaints. "
                "Cite real URLs and return control to @Arbiter.\n"
                "HANDOFF: @Researcher | STATE: RESEARCH | REQUEST: sourced fact list",
                ["@Researcher"],
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
            state.research_evidence = self._snapshot(msg.content, max_urls=2)
            state.phase = DebatePhase.SCOUTING
            await self._delegate(
                state,
                tools,
                "⚖️ EVIDENCE RECEIVED\n"
                f"{self._context_block(state, include=('research',), max_words=125)}\n"
                "@Scout compare 2-3 credible alternatives using price, fit, and trade-offs. "
                "Return control to @Arbiter.\n"
                "HANDOFF: @Scout | STATE: SCOUTING | REQUEST: alternative comparison",
                ["@Scout"],
            )
        elif state.phase == DebatePhase.SCOUTING:
            state.scout_evidence = self._snapshot(msg.content, max_urls=1)
            state.phase = DebatePhase.ROUND_1_CRITIC
            await self._delegate(
                state,
                tools,
                "⚖️ ROUND 1: CHALLENGE\n"
                f"{self._context_block(state, include=('research', 'scout'), max_words=115)}\n"
                "@Critic present the strongest case AGAINST the purchase using only room evidence. "
                "Call out hidden cost, lock-in, migration, security, and privacy. "
                "Use the exact marker COMPLIANCE CONCERN when specialist review is needed. "
                "Return control to @Arbiter.\n"
                "HANDOFF: @Critic | STATE: ROUND_1 | REQUEST: strongest objections",
                ["@Critic"],
            )
        elif state.phase == DebatePhase.ROUND_1_CRITIC:
            state.critic_round_1 = self._snapshot(msg.content)
            if "COMPLIANCE CONCERN" in msg.content.upper():
                state.compliance_requested = True
                if await self._recruit_compliance(tools):
                    state.phase = DebatePhase.COMPLIANCE
                    await self._delegate(
                        state,
                        tools,
                        "⚖️ SPECIALIST REVIEW\n"
                        f"{self._context_block(state, include=('research', 'scout', 'critic1'), max_words=115)}\n"
                        "@Compliance assess data residency, GDPR/privacy, certifications, and contractual risk. "
                        "Return control to @Arbiter.\n"
                        "HANDOFF: @Compliance | STATE: RECRUIT | REQUEST: compliance opinion",
                        ["@Compliance"],
                    )
                else:
                    state.phase = DebatePhase.ROUND_1_ADVOCATE
                    await self._ask_advocate(tools, state)
            else:
                state.phase = DebatePhase.ROUND_1_ADVOCATE
                await self._ask_advocate(tools, state)
        elif state.phase == DebatePhase.COMPLIANCE:
            state.compliance_reviewed = True
            state.compliance_evidence = self._snapshot(msg.content)
            state.phase = DebatePhase.ROUND_1_ADVOCATE
            await self._ask_advocate(tools, state)
        elif state.phase == DebatePhase.ROUND_1_ADVOCATE:
            state.advocate_round_1 = self._snapshot(msg.content)
            state.phase = DebatePhase.ROUND_2_CRITIC
            await self._delegate(
                state,
                tools,
                "⚖️ ROUND 2: REBUTTAL\n"
                f"{self._context_block(state, include=('research', 'scout', 'critic1', 'compliance', 'advocate1'), max_words=110)}\n"
                "@Critic strengthen only the objections that remain unresolved. Do not repeat closed points. "
                "Return control to @Arbiter.\n"
                "HANDOFF: @Critic | STATE: ROUND_2 | REQUEST: unresolved objections",
                ["@Critic"],
            )
        elif state.phase == DebatePhase.ROUND_2_CRITIC:
            state.critic_round_2 = self._snapshot(msg.content)
            state.phase = DebatePhase.ROUND_2_ADVOCATE
            await self._delegate(
                state,
                tools,
                "⚖️ ROUND 2: CLOSING\n"
                f"{self._context_block(state, include=('research', 'scout', 'critic2', 'compliance'), max_words=110)}\n"
                "@Advocate give a concise closing statement. Address the remaining objections point by point "
                "and state any purchase conditions. Return control to @Arbiter.\n"
                "HANDOFF: @Advocate | STATE: ROUND_2 | REQUEST: final defense",
                ["@Advocate"],
            )
        elif state.phase == DebatePhase.ROUND_2_ADVOCATE:
            await self._finalize_verdict(room_id, state, msg, history, tools)

        logger.info("%s -> %s", previous, self.states[room_id].phase)
        self._schedule_timeout(room_id, tools)

    async def _ask_advocate(
        self, tools: AgentToolsProtocol, state: DebateState
    ) -> None:
        await self._delegate(
            state,
            tools,
            "⚖️ ROUND 1: DEFENSE\n"
            f"{self._context_block(state, include=('research', 'scout', 'critic1', 'compliance'), max_words=115)}\n"
            "@Advocate answer the Critic point by point using room evidence only. "
            "Concede unsupported claims and propose concrete safeguards. Return control to @Arbiter.\n"
            "HANDOFF: @Advocate | STATE: ROUND_1 | REQUEST: point-by-point defense",
            ["@Advocate"],
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
    def _brief(value: str, max_words: int) -> str:
        words = " ".join(value.split()).split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]) + "…"

    @classmethod
    def _snapshot(cls, value: str, max_urls: int = 0) -> str:
        lines = [
            line.strip()
            for line in value.splitlines()
            if line.strip() and not line.strip().startswith("HANDOFF:")
        ]
        cleaned = " ".join(lines)
        cleaned = re.sub(r"^[⚖️🔍✅❌🕵️🛡️]\s*", "", cleaned)
        urls = list(dict.fromkeys(re.findall(r"https?://[^\s)\]>]+", cleaned)))
        prose = re.sub(r"https?://[^\s)\]>]+", "", cleaned)
        snapshot = cls._brief(prose, 120)
        if max_urls and urls:
            snapshot = f"{snapshot} SOURCES: {' '.join(urls[:max_urls])}"
        return " ".join(snapshot.split())

    @classmethod
    def _cap_message(cls, content: str, max_words: int = 195) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        handoff = next(
            (line for line in reversed(lines) if line.startswith("HANDOFF:")), ""
        )
        body_lines = [line for line in lines if not line.startswith("HANDOFF:")]
        reserved = len(handoff.split()) if handoff else 0
        body = cls._brief(" ".join(body_lines), max(1, max_words - reserved))
        return f"{body}\n{handoff}".strip() if handoff else body

    async def _delegate(
        self,
        state: DebateState,
        tools: AgentToolsProtocol,
        content: str,
        mentions: list[str],
    ) -> None:
        message = self._cap_message(content)
        state.pending_message = message
        state.pending_mentions = list(mentions)
        await tools.send_message(message, mentions=mentions)

    @classmethod
    def _clean_case(cls, msg: PlatformMessage) -> str:
        content = re.sub(
            r"@\[\[[0-9a-fA-F-]{16,}\]\]",
            cls._mention_for(msg),
            msg.content,
        )
        return " ".join(content.split())

    @classmethod
    def _context_block(
        cls,
        state: DebateState,
        *,
        include: tuple[str, ...],
        max_words: int,
    ) -> str:
        labels = {
            "research": ("FACTS", state.research_evidence),
            "scout": ("ALTERNATIVES", state.scout_evidence),
            "critic1": ("CRITIC R1", state.critic_round_1),
            "compliance": ("COMPLIANCE", state.compliance_evidence),
            "advocate1": ("ADVOCATE R1", state.advocate_round_1),
            "critic2": ("CRITIC R2", state.critic_round_2),
        }
        parts = [f"CASE BRIEF: {cls._brief(state.case, 38)}"]
        available = [
            (label, value) for key in include for label, value in [labels[key]] if value
        ]
        if available:
            remaining = max(20, max_words - len(parts[0].split()) - len(available) * 2)
            each = max(15, remaining // len(available))
            digest = " | ".join(
                f"{label}: {cls._brief(value, each)}" for label, value in available
            )
            parts.append(f"EVIDENCE DIGEST: {digest}")
        for role in state.missing_agents:
            parts.append(f"GAP: {role} unavailable")
        return "\n".join(parts)

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
        phase = self.states[room_id].phase
        scheduled_at = asyncio.get_running_loop().time()
        logger.info(
            "Timeout scheduled room=%s phase=%s delay=%.1fs deadline=%.3f",
            room_id,
            phase,
            self.turn_timeout,
            scheduled_at + self.turn_timeout,
        )
        self.timeout_tasks[room_id] = asyncio.create_task(
            self._watch_turn(room_id, phase, tools, scheduled_at)
        )

    async def _watch_turn(
        self,
        room_id: str,
        phase: DebatePhase,
        tools: AgentToolsProtocol,
        scheduled_at: float | None = None,
    ) -> None:
        try:
            started_at = scheduled_at or asyncio.get_running_loop().time()
            await asyncio.sleep(self.turn_timeout or 0)
            if self.states[room_id].phase != phase:
                return
            elapsed = asyncio.get_running_loop().time() - started_at
            logger.info(
                "Timeout reminder firing room=%s phase=%s elapsed=%.3fs",
                room_id,
                phase,
                elapsed,
            )
            expected = self._expected_handle(phase)
            reminder = self._cap_message(
                "⚖️ TIMEOUT REMINDER — this is the only retry.\n"
                f"{self.states[room_id].pending_message}"
            )
            await tools.send_message(reminder, mentions=[expected])
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
            await self._delegate(
                state,
                tools,
                "⚖️ RESEARCH TIMED OUT\n"
                f"{self._context_block(state, include=(), max_words=100)}\n"
                "@Scout compare 2-3 alternatives using available room evidence.\n"
                "HANDOFF: @Scout | STATE: SCOUTING | REQUEST: alternative comparison",
                ["@Scout"],
            )
        elif phase == DebatePhase.SCOUTING:
            state.phase = DebatePhase.ROUND_1_CRITIC
            await self._delegate(
                state,
                tools,
                "⚖️ SCOUTING TIMED OUT\n"
                f"{self._context_block(state, include=('research',), max_words=110)}\n"
                "@Critic present the strongest evidence-based objections.\n"
                "HANDOFF: @Critic | STATE: ROUND_1 | REQUEST: strongest objections",
                ["@Critic"],
            )
        elif phase in {DebatePhase.ROUND_1_CRITIC, DebatePhase.COMPLIANCE}:
            state.phase = DebatePhase.ROUND_1_ADVOCATE
            await self._ask_advocate(tools, state)
        elif phase == DebatePhase.ROUND_1_ADVOCATE:
            state.phase = DebatePhase.ROUND_2_CRITIC
            await self._delegate(
                state,
                tools,
                "⚖️ DEFENSE TIMED OUT\n"
                f"{self._context_block(state, include=('research', 'scout', 'critic1', 'compliance'), max_words=110)}\n"
                "@Critic identify only unresolved objections for the closing round.\n"
                "HANDOFF: @Critic | STATE: ROUND_2 | REQUEST: unresolved objections",
                ["@Critic"],
            )
        elif phase == DebatePhase.ROUND_2_CRITIC:
            state.phase = DebatePhase.ROUND_2_ADVOCATE
            await self._delegate(
                state,
                tools,
                "⚖️ REBUTTAL TIMED OUT\n"
                f"{self._context_block(state, include=('research', 'scout', 'critic1', 'compliance'), max_words=110)}\n"
                "@Advocate give the final defense and purchase conditions.\n"
                "HANDOFF: @Advocate | STATE: ROUND_2 | REQUEST: final defense",
                ["@Advocate"],
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
