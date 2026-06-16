from __future__ import annotations

import re
from typing import Any, Literal

from band import Agent
from band.core.protocols import AgentToolsProtocol
from band.core.simple_adapter import SimpleAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.agents.base import (
    HANDOFF_LINE_PATTERN,
    build_langgraph_agent,
    load_prompt,
    prepare_worker_delivery,
)
from src.common.config import AgentCredentials, Settings
from src.common.llm import make_llm

RiskArea = Literal[
    "cost",
    "migration",
    "security",
    "privacy",
    "capability",
    "alternatives",
]

RISK_QUESTIONS = {
    "cost": "Could the documented pricing create budget pressure?",
    "migration": "Might migration affect continuity or data quality?",
    "security": "Could the documented controls leave a buyer requirement unresolved?",
    "privacy": "Might the documented privacy terms leave a buyer requirement unresolved?",
    "capability": "Could the documented capabilities miss an important buyer workflow?",
    "alternatives": "Might an alternative provide a better documented fit?",
}

RISK_CHECKS = {
    "cost": "Compare equivalent vendor quotes and contract terms.",
    "migration": "Run a representative import and export pilot.",
    "security": "Map documented controls to the buyer security requirements.",
    "privacy": "Review the privacy agreement, data flows, and subprocessors.",
    "capability": "Test the critical buyer workflows in a pilot.",
    "alternatives": "Compare equivalent capabilities and contract terms.",
}


DEBATE_EVIDENCE_OUTPUT_CONTRACT = """
FINAL OUTPUT CONTRACT:
- Do not repeat the buyer headcount or other case quantities. Say `the buyer`,
  `the company`, or `the team` instead.
- Every other factual digit, price, percentage, duration, or quantity must be
  followed immediately by `(evidence: "<verbatim digest substring>")` on the
  same line.
- Copy that substring exactly, including punctuation and units. Never
  paraphrase inside an evidence quote. If exact copying is uncertain, omit the
  number and write the point qualitatively.
- Never use `...` or an ellipsis inside an evidence quote unless those exact
  characters occur in the authoritative source. Prefer a shorter exact
  contiguous quote.
- A list of documented features proves only that those features are documented.
  Never infer that an omitted feature, integration, control, or certification
  is absent.
- Use `lacks`, `without`, `does not have`, `only`, `limited to`, or an
  equivalent product limitation only when the authoritative evidence contains
  that same explicit negative fact. Otherwise say the evidence does not address
  that topic.
- Evidence quotes may come only from CASE BRIEF, FACTS, ALTERNATIVES, or
  COMPLIANCE, never from DEBATE CONTEXT.
- Do not introduce a certification name containing a number unless that exact
  name appears in authoritative evidence and is cited. Otherwise say
  `certification documentation`.
- Use neutral topic headings such as `Cost`, `Migration`, `Security`, and
  `Certification evidence gap`. Never repeat an unsupported opponent claim in
  a heading, even when the body will rebut it.
- If a number has no exact authoritative quote, remove the number and write the
  point qualitatively. Do not calculate or derive a new quantity.
""".strip()

DEBATE_REVIEW_CONTRACT = """
STRUCTURED FINAL FORMAT:
- Return exactly the six required schema items: three evidence quotes, two
  hypotheses, and one condition.
- Put every factual digit, price, percentage, duration, or quantity only inside
  an evidence.quote field.
- evidence.quote must be one exact contiguous source line substring. Never
  shorten, combine lines, add markdown, or use an ellipsis.
- Copy only characters that occur on that source line. Do not prepend a product
  name or heading from a neighboring line.
- Use an absence quote only when that same quote explicitly states the absence
  or evidence gap. Omission is not evidence.
- Put all interpretation or buyer impact in hypothesis questions starting with
  Could, Might, or May. Never assert a new product fact there.
- Conditions request checks without predicting results.
- Use no facts from DEBATE CONTEXT or RECENT ROOM TRANSCRIPT.
""".strip()


class GroundedEvidence(BaseModel):
    quote: str = Field(
        description=(
            "One exact contiguous line substring copied from CASE BRIEF, FACTS, "
            "ALTERNATIVES, or COMPLIANCE. Do not shorten or combine source text. "
            "Do not prepend a product name or section heading from another line."
        )
    )


class GroundedHypothesis(BaseModel):
    area: RiskArea = Field(
        description=(
            "Choose the single buyer-risk category best supported by the "
            "authoritative evidence. Do not generate free-form risk prose."
        )
    )


class GroundedCondition(BaseModel):
    area: RiskArea = Field(
        description=(
            "Choose the single category for the most important purchase check. "
            "Do not generate free-form condition prose."
        )
    )


class GroundedDebateResponse(BaseModel):
    compliance_concern: bool = Field(
        description=(
            "True only for Critic when the authoritative case materially involves "
            "privacy, GDPR, data handling, security assurance, or certification "
            "review and specialist compliance review is warranted."
        )
    )
    evidence_primary: GroundedEvidence
    evidence_secondary: GroundedEvidence
    evidence_tertiary: GroundedEvidence
    hypothesis_primary: GroundedHypothesis
    hypothesis_secondary: GroundedHypothesis
    condition: GroundedCondition


def _research_tools() -> list[Any]:
    from src.tools.web_research import fetch_page, reddit_search, web_search

    return [
        StructuredTool.from_function(web_search),
        StructuredTool.from_function(fetch_page),
        StructuredTool.from_function(reddit_search),
    ]


def build_worker_agent(
    role: str,
    credentials: AgentCredentials,
    settings: Settings,
):
    if role in {"researcher", "scout"}:
        return build_langgraph_agent(
            role, credentials, settings, tools=_research_tools()
        )
    return Agent.create(
        adapter=TextWorkerAdapter(
            role=role,
            llm=make_llm(
                _model_role_for_worker(role),
                settings,
                temperature_role=role,
            ),
            instructions=load_prompt(role),
        ),
        agent_id=credentials.agent_id,
        api_key=credentials.api_key,
        rest_url=settings.band_rest_url,
        ws_url=settings.band_ws_url,
    )


def _model_role_for_worker(role: str) -> str:
    return "arbiter"


class TextWorkerAdapter(SimpleAdapter[HistoryProvider]):
    """Generate text with any chat model, then send it through Band explicitly."""

    def __init__(
        self,
        role: str,
        llm: BaseChatModel,
        instructions: str,
        grounded_llm: Runnable[Any, GroundedDebateResponse] | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.llm = llm
        self.instructions = instructions
        self.grounded_llm = grounded_llm

    def _fallback_handoff(self) -> str:
        return (
            f"HANDOFF: @Arbiter | STATE: {self.role.upper()}_COMPLETE "
            "| REQUEST: continue the protocol"
        )

    def _handoff_for_request(self, request: str) -> str:
        normalized = request.upper()
        if self.role == "compliance":
            return (
                "HANDOFF: @Arbiter | STATE: COMPLIANCE_COMPLETE "
                "| REQUEST: resume the debate with this assessment"
            )
        round_name = "ROUND_2" if "ROUND 2" in normalized else "ROUND_1"
        if self.role == "critic":
            return (
                f"HANDOFF: @Arbiter | STATE: {round_name}_CRITIC_COMPLETE "
                "| REQUEST: task @Advocate"
            )
        return (
            f"HANDOFF: @Arbiter | STATE: {round_name}_ADVOCATE_COMPLETE "
            "| REQUEST: continue the protocol"
        )

    def _prepare_delivery(self, content: str, handoff: str | None = None) -> str:
        handoff_lines = HANDOFF_LINE_PATTERN.findall(content)
        handoff_lines = [
            re.sub(
                r"(?i)^\s*(?:\*\*)?HANDOFF:(?:\*\*)?\s*",
                "HANDOFF: ",
                line,
            ).strip().removesuffix("**").rstrip()
            for line in handoff_lines
        ]
        handoff = handoff or (
            handoff_lines[-1] if handoff_lines else self._fallback_handoff()
        )
        return prepare_worker_delivery(content, handoff=handoff)

    def _render_grounded(self, response: GroundedDebateResponse) -> str:
        marker = {"advocate": "✅", "critic": "❌", "compliance": "🛡️"}[self.role]
        lines = [marker]
        if self.role == "critic" and response.compliance_concern:
            lines.append("🚨 COMPLIANCE CONCERN")
        evidence = (
            response.evidence_primary,
            response.evidence_secondary,
            response.evidence_tertiary,
        )
        hypotheses = (
            response.hypothesis_primary,
            response.hypothesis_secondary,
        )
        lines.extend(
            f'- EVIDENCE | (evidence: "{item.quote}")'
            for item in evidence
        )
        lines.extend(
            f"- HYPOTHESIS | QUESTION: {RISK_QUESTIONS[item.area]} "
            f"| CHECK: {RISK_CHECKS[item.area]}"
            for item in hypotheses
        )
        lines.append(
            f"- CONDITION | CHECK: {RISK_CHECKS[response.condition.area]}"
        )
        return "\n".join(lines)

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
        del participants_msg, contacts_msg, is_session_bootstrap, room_id
        transcript = "\n".join(
            f"[{item.get('sender_name') or item.get('sender_type', 'Unknown')}]: "
            f"{item.get('content', '')}"
            for item in history.raw[-20:]
        )
        grounded_llm = self.grounded_llm or self.llm.with_structured_output(
            GroundedDebateResponse,
            method="function_calling",
            strict=True,
        )
        response = await grounded_llm.ainvoke(
            f"{self.instructions}\n\n"
            f"{DEBATE_EVIDENCE_OUTPUT_CONTRACT}\n\n"
            f"{DEBATE_REVIEW_CONTRACT}\n\n"
            f"RECENT ROOM TRANSCRIPT FOR ARGUMENT CONTEXT ONLY:\n{transcript}\n\n"
            f"AUTHORITATIVE REQUEST AND CONTEXT:\n{msg.content}\n\n"
            "Populate only the structured schema. Use no facts from DEBATE "
            "CONTEXT or RECENT ROOM TRANSCRIPT. Set compliance_concern true "
            "only for Critic when the authoritative case materially warrants "
            "specialist privacy, GDPR, data-handling, security-assurance, or "
            "certification review."
        )
        content = self._render_grounded(response)
        await tools.send_message(
            self._prepare_delivery(
                content,
                handoff=self._handoff_for_request(msg.content),
            ),
            mentions=["@Arbiter"],
        )
