from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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


class ArbiterAdapter(SimpleAdapter[HistoryProvider]):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__()
        self.llm = llm
        self.states: dict[str, DebateState] = {}

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

        if state.phase == DebatePhase.IDLE:
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

        logger.info("%s -> %s", previous, self.states[room_id].phase)

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
            await tools.add_participant("Compliance")
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
        transcript = "\n".join(
            f"[{item.get('sender_name') or item.get('sender_type', 'Unknown')}]: "
            f"{item.get('content', '')}"
            for item in history.raw[-30:]
        )
        transcript += f"\n[{msg.sender_name or msg.sender_type}]: {msg.content}"
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
            scores = verdict["scores"]
            verdict["total"] = sum(int(value) for value in scores.values())
            verdict["compliance_reviewed"] = state.compliance_requested
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
                "compliance_reviewed": state.compliance_requested,
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
            f"Dissent: {verdict.get('dissent', 'None')}"
        )

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


def build_arbiter_agent(
    credentials: AgentCredentials,
    settings: Settings,
) -> Agent:
    return Agent.create(
        adapter=ArbiterAdapter(make_llm("arbiter", settings)),
        agent_id=credentials.agent_id,
        api_key=credentials.api_key,
        rest_url=settings.band_rest_url,
        ws_url=settings.band_ws_url,
    )
