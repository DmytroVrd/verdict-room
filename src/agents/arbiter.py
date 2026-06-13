from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from band import Agent
from band.core.protocols import AgentToolsProtocol
from band.core.simple_adapter import SimpleAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.agents.base import EMERGENCY_WORD_LIMIT, truncate_at_sentence_boundary
from src.common.config import PROJECT_ROOT, AgentCredentials, Settings
from src.common.llm import make_llm

logger = logging.getLogger("ARBITER")
VerdictRiskArea = Literal[
    "cost",
    "migration",
    "security",
    "privacy",
    "capability",
    "alternatives",
]

VERDICT_RISK_QUESTIONS = {
    "cost": "Could the documented pricing create budget pressure?",
    "migration": "Might migration affect continuity or data quality?",
    "security": "Could the documented controls leave a buyer requirement unresolved?",
    "privacy": "Might the documented privacy terms leave a buyer requirement unresolved?",
    "capability": "Could the documented capabilities miss an important buyer workflow?",
    "alternatives": "Might an alternative provide a better documented fit?",
}

VERDICT_CHECKS = {
    "cost": "Compare equivalent vendor quotes and contract terms.",
    "migration": "Run a representative import and export pilot.",
    "security": "Map documented controls to the buyer security requirements.",
    "privacy": "Review the privacy agreement, data flows, and subprocessors.",
    "capability": "Test the critical buyer workflows in a pilot.",
    "alternatives": "Compare equivalent capabilities and contract terms.",
}

ARBITER_EVIDENCE_CONTRACT = """
Use only facts and numbers present in the transcript. Do not calculate new
totals, infer that a missing certification is absent, or turn an evidence gap
into a negative fact. When evidence is missing, describe it as unknown.
Do not repeat the buyer headcount in rationale, conditions, or dissent.
Treat a list of documented features as proof only of those listed features;
never infer that an omitted feature, integration, control, or certification is
absent. Use `lacks`, `without`, `does not have`, `only`, `limited to`, or
equivalent limitation wording only when authoritative evidence explicitly
states that same negative fact. Do not name a numbered certification unless
its exact name appears in authoritative evidence and is cited; otherwise use
the generic phrase `certification documentation`.
Every factual number, price, percentage, duration, or quantity in rationale,
conditions, or dissent must include a short exact evidence quote in the same
string using `(evidence: "...")`. If no exact quote exists, use qualitative
wording without a number. Scores and total are exempt from this citation format.
Every evidence quote must be one character-for-character contiguous substring
of the authoritative CASE BRIEF or EVIDENCE DIGEST, not another agent's
argument. If exact copying is uncertain, omit the quote and express the point
qualitatively or as unknown.
""".strip()

ARBITER_REVIEW_CONTRACT = """
You are the final evidence editor for a purchase verdict. Return ONLY one
complete valid JSON object matching the draft schema.

Silently verify every rationale, condition, and dissent string:
- Do not repeat the buyer headcount or other case quantities.
- Do not calculate or derive any new quantity.
- Every remaining factual quantity must have an immediate `(evidence: "...")`
  quote copied character-for-character from AUTHORITATIVE EVIDENCE.
- Every evidence quote must be one contiguous substring. Never use `...` or an
  ellipsis to combine source fragments, and never cite debate arguments.
- A feature list proves only the listed features. Do not infer absence from
  omission.
- Use negative limitation wording only when AUTHORITATIVE EVIDENCE explicitly
  states that same negative fact.
- Delete unsupported certification names containing numbers and use the generic
  phrase `certification documentation`.
- When support is missing, write the point qualitatively or as unknown.

Preserve supported reasoning and the draft's score values. Return no markdown
fence and no commentary.
""".strip()


class VerdictEvidence(BaseModel):
    quote: str = Field(
        description=(
            "One exact contiguous source-line substring copied from CASE BRIEF "
            "or EVIDENCE DIGEST. Do not prepend a heading or product name from "
            "another line, combine lines, add markdown, or use an ellipsis."
        )
    )


class VerdictCheck(BaseModel):
    area: VerdictRiskArea = Field(
        description=(
            "Choose the category for a purchase verification step. Do not "
            "generate free-form condition prose."
        )
    )


class GroundedVerdictResponse(BaseModel):
    value_for_money: int = Field(ge=0, le=25)
    capability_fit: int = Field(ge=0, le=25)
    risk_profile: int = Field(ge=0, le=25)
    alternatives: int = Field(ge=0, le=25)
    recommendation: Literal["BUY", "BUY_WITH_CONDITIONS", "AVOID"]
    rationale_primary: VerdictEvidence
    rationale_secondary: VerdictEvidence
    rationale_tertiary: VerdictEvidence
    condition_primary: VerdictCheck
    condition_secondary: VerdictCheck
    dissent_area: VerdictRiskArea = Field(
        description=(
            "Choose the category for the strongest opposing buyer-risk question. "
            "Do not generate free-form dissent prose."
        )
    )


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
        mention_names: dict[str, str] | None = None,
        verdict_llm: Runnable[Any, GroundedVerdictResponse] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.turn_timeout = turn_timeout
        self.compliance_identifier = compliance_identifier
        self.verdict_llm = verdict_llm
        self.mention_names = {
            identifier.lower(): name
            for identifier, name in (mention_names or {}).items()
        }
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
            f"[{msg.sender_name or msg.sender_type}]: "
            f"{self._sanitize_mentions(msg.content)}"
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
                f"CASE BRIEF: {state.case}\n"
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
            state.research_evidence = self._snapshot(msg.content)
            state.phase = DebatePhase.SCOUTING
            await self._delegate(
                state,
                tools,
                "⚖️ EVIDENCE RECEIVED\n"
                f"{self._context_block(state, include=('research',))}\n"
                "@Scout compare 2-3 credible alternatives using price, fit, and trade-offs. "
                "Return control to @Arbiter.\n"
                "HANDOFF: @Scout | STATE: SCOUTING | REQUEST: alternative comparison",
                ["@Scout"],
            )
        elif state.phase == DebatePhase.SCOUTING:
            state.scout_evidence = self._snapshot(msg.content)
            state.phase = DebatePhase.ROUND_1_CRITIC
            await self._delegate(
                state,
                tools,
                "⚖️ ROUND 1: CHALLENGE\n"
                f"{self._context_block(state, include=('research', 'scout'))}\n"
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
                        f"{self._context_block(state, include=('critic1', 'research', 'scout'))}\n"
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
                f"{self._context_block(state, include=('advocate1', 'critic1', 'compliance', 'research', 'scout'))}\n"
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
                f"{self._context_block(state, include=('critic2', 'compliance', 'research', 'scout'))}\n"
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
            f"{self._context_block(state, include=('critic1', 'compliance', 'research', 'scout'))}\n"
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
        if self.verdict_llm is not None:
            return await self._make_grounded_verdict(state)

        hydrated_history = "\n".join(
            f"[{item.get('sender_name') or item.get('sender_type', 'Unknown')}]: "
            f"{self._sanitize_mentions(str(item.get('content', '')))}"
            for item in history.raw[-30:]
        )
        transcript_parts = [hydrated_history, *state.transcript]
        transcript = "\n".join(part for part in transcript_parts if part)
        prompt = f"""
You are the neutral Arbiter. Produce ONLY valid JSON for the purchase case below.
Scores are integers 0-25 and total must equal their sum.
recommendation must be BUY, BUY_WITH_CONDITIONS, or AVOID.
Keep rationale to 3-5 short strings and dissent to one strong opposing argument.
{ARBITER_EVIDENCE_CONTRACT}

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
        authoritative = self._context_block(
            state,
            include=("research", "scout", "compliance"),
        )
        review = await self.llm.ainvoke(
            f"{ARBITER_REVIEW_CONTRACT}\n\n"
            f"AUTHORITATIVE EVIDENCE:\n{authoritative}\n\n"
            f"DRAFT JSON:\n{text}\n\n"
            "FINAL MANDATORY CHECK BEFORE OUTPUT: Return the complete corrected "
            "JSON now. Remove buyer headcount, derived quantities, non-contiguous "
            "or paraphrased evidence quotes, absence claims inferred from "
            "omission, and unsupported certification names containing numbers."
        )
        reviewed_text = (
            review.content
            if isinstance(review.content, str)
            else str(review.content)
        ).strip()
        try:
            verdict = json.loads(self._extract_json(reviewed_text or text))
            verdict = self._normalize_verdict(verdict, state)
            verdict["compliance_reviewed"] = state.compliance_reviewed
            verdict["evidence_gaps"] = state.missing_agents
            return verdict
        except Exception:
            logger.exception("Invalid reviewed verdict JSON; trying the draft")
            try:
                verdict = json.loads(self._extract_json(text))
                verdict = self._normalize_verdict(verdict, state)
                verdict["compliance_reviewed"] = state.compliance_reviewed
                verdict["evidence_gaps"] = state.missing_agents
                return verdict
            except Exception:
                logger.exception("Invalid draft verdict JSON; using safe fallback")
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

    async def _make_grounded_verdict(
        self,
        state: DebateState,
    ) -> dict[str, Any]:
        authoritative = self._context_block(
            state,
            include=("research", "scout", "compliance"),
        )
        debate = self._context_block(
            state,
            include=("critic1", "advocate1", "critic2"),
        )
        response = await self.verdict_llm.ainvoke(
            "Create the final structured purchase verdict.\n"
            "- Use DEBATE CONTEXT only to choose scores and recommendation.\n"
            "- Put every factual digit, price, percentage, duration, or quantity "
            "only inside a rationale quote.\n"
            "- Each rationale quote must be one exact contiguous source-line "
            "substring from CASE BRIEF or EVIDENCE DIGEST. Do not prepend a "
            "product name or heading from another line, combine lines, add "
            "markdown, or use an ellipsis.\n"
            "- Use an absence quote only when it explicitly states the absence "
            "or evidence gap. Omission is not evidence.\n"
            "- Conditions are checks with no predicted result.\n"
            "- Dissent is an uncertain buyer-risk question, not a product fact.\n\n"
            f"AUTHORITATIVE EVIDENCE:\n{authoritative}\n\n"
            f"DEBATE CONTEXT FOR SCORING ONLY:\n{debate}"
        )
        scores = {
            "value_for_money": response.value_for_money,
            "capability_fit": response.capability_fit,
            "risk_profile": response.risk_profile,
            "alternatives": response.alternatives,
        }
        rationale = [
            f'Evidence: (evidence: "{item.quote}")'
            for item in (
                response.rationale_primary,
                response.rationale_secondary,
                response.rationale_tertiary,
            )
        ]
        return {
            "case": state.case,
            "scores": scores,
            "total": sum(scores.values()),
            "recommendation": response.recommendation,
            "rationale": rationale,
            "conditions": [
                VERDICT_CHECKS[response.condition_primary.area],
                VERDICT_CHECKS[response.condition_secondary.area],
            ],
            "dissent": VERDICT_RISK_QUESTIONS[response.dissent_area],
            "compliance_reviewed": state.compliance_reviewed,
            "evidence_gaps": list(state.missing_agents),
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

    def _snapshot(self, value: str) -> str:
        lines = [
            line.strip()
            for line in value.splitlines()
            if line.strip() and not line.strip().startswith("HANDOFF:")
        ]
        cleaned = self._sanitize_mentions("\n".join(lines))
        cleaned = re.sub(r"^[⚖️🔍✅❌🕵️🛡️]\s*", "", cleaned)
        return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())

    @staticmethod
    def _cap_message(content: str, max_words: int = EMERGENCY_WORD_LIMIT) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        handoff = next(
            (line for line in reversed(lines) if line.startswith("HANDOFF:")), ""
        )
        body_lines = [line for line in lines if not line.startswith("HANDOFF:")]
        directive = next(
            (line for line in reversed(body_lines) if line.startswith("@")),
            "",
        )
        if directive:
            body_lines.remove(directive)
        reserved = len(handoff.split()) + len(directive.split())
        body = truncate_at_sentence_boundary(
            "\n".join(body_lines),
            max(1, max_words - reserved),
        )
        ending = "\n".join(line for line in (directive, handoff) if line)
        return f"{body}\n{ending}".strip() if ending else body

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

    def _clean_case(self, msg: PlatformMessage) -> str:
        content = self._sanitize_mentions(msg.content)
        return " ".join(content.split())

    def _sanitize_mentions(self, value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            identifier = match.group(1).lower()
            name = self.mention_names.get(identifier, "participant")
            return name if name.startswith("@") else f"@{name}"

        return re.sub(r"@\[\[([0-9a-fA-F-]{16,})\]\]", replace, value)

    def _context_block(
        self,
        state: DebateState,
        *,
        include: tuple[str, ...],
    ) -> str:
        labels = {
            "research": ("evidence", "FACTS", state.research_evidence),
            "scout": ("evidence", "ALTERNATIVES", state.scout_evidence),
            "critic1": ("debate", "CRITIC R1", state.critic_round_1),
            "compliance": ("evidence", "COMPLIANCE", state.compliance_evidence),
            "advocate1": ("debate", "ADVOCATE R1", state.advocate_round_1),
            "critic2": ("debate", "CRITIC R2", state.critic_round_2),
        }
        parts = [f"CASE BRIEF:\n{self._sanitize_mentions(state.case)}"]
        available = [
            (kind, label, value)
            for key in include
            for kind, label, value in [labels[key]]
            if value
        ]
        debate = [(label, value) for kind, label, value in available if kind == "debate"]
        evidence = [
            (label, value) for kind, label, value in available if kind == "evidence"
        ]
        if evidence:
            digest = "\n\n".join(
                f"{label}:\n{self._sanitize_mentions(value)}"
                for label, value in evidence
            )
            parts.append(f"EVIDENCE DIGEST:\n{digest}")
        if debate:
            context = "\n\n".join(
                f"{label}:\n{self._sanitize_mentions(value)}"
                for label, value in debate
            )
            parts.append(f"DEBATE CONTEXT (ARGUMENTS, NOT EVIDENCE):\n{context}")
        for role in state.missing_agents:
            parts.append(f"GAP: {role} unavailable")
        return "\n".join(parts)

    @staticmethod
    def _is_expected_sender(phase: DebatePhase, msg: PlatformMessage) -> bool:
        if phase == DebatePhase.IDLE and "HANDOFF:" in msg.content.upper():
            return False
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
                f"{self._context_block(state, include=())}\n"
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
                f"{self._context_block(state, include=('research',))}\n"
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
                f"{self._context_block(state, include=('critic1', 'compliance', 'research', 'scout'))}\n"
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
                f"{self._context_block(state, include=('critic1', 'compliance', 'research', 'scout'))}\n"
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
    mention_names: dict[str, str] | None = None,
) -> Agent:
    llm = make_llm("arbiter", settings)
    return Agent.create(
        adapter=ArbiterAdapter(
            llm,
            turn_timeout=settings.debate_turn_timeout,
            compliance_identifier=compliance_identifier,
            mention_names=mention_names,
            verdict_llm=llm.with_structured_output(
                GroundedVerdictResponse,
                method="function_calling",
                strict=True,
            ),
        ),
        agent_id=credentials.agent_id,
        api_key=credentials.api_key,
        rest_url=settings.band_rest_url,
        ws_url=settings.band_ws_url,
    )
