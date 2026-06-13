from __future__ import annotations

import logging
import re
from typing import Any

from band import Agent
from band.core.protocols import AgentToolsProtocol
from band.core.simple_adapter import SimpleAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from src.agents.base import (
    HANDOFF_LINE_PATTERN,
    build_langgraph_agent,
    load_prompt,
    prepare_worker_delivery,
)
from src.common.config import AgentCredentials, Settings
from src.common.llm import make_llm

logger = logging.getLogger(__name__)

NUMERIC_CLAIM_PATTERN = re.compile(
    r"""(?ix)
    (?:[$€£]\s*)?
    \d[\d,]*(?:\.\d+)?
    (?:\s*[-–]\s*(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?)?
    (?:\s*[-–]?\s*(?:%|/(?:users?|months?|years?|seats?|credits?)|users?|persons?|people|members?|teams?|seats?|credits?|
        billing\s+cycles?|days?|weeks?|months?|years?))?
    |
    \b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)
    \s+(?:users?|people|members?|teams?|seats?|billing\s+cycles?|days?|weeks?|
        months?|years?)\b
    """
)
UNSUPPORTED_CERTIFICATION_ABSENCE_PATTERN = re.compile(
    r"(?:"
    r"\b(?:lacks?|has\s+no|does\s+not\s+have|is\s+without)\b.{0,100}"
    r"\b(?:soc\s*2|iso\s*27001|certification|audit report)"
    r"|"
    r"\b(?:soc\s*2|iso\s*27001|certifications?|audit reports?)\b.{0,120}"
    r"\b(?:but\s+)?not\s+(?:in|for|on)\b"
    r")",
    re.IGNORECASE,
)
CONTRADICTED_MARKDOWN_EXPORT_PATTERN = re.compile(
    r"\bexports?\b.{0,80}\b(?:limited|only)\b.{0,40}\bmarkdown\b",
    re.IGNORECASE,
)
PRESCRIPTIVE_NUMERIC_PATTERN = re.compile(
    r"\b(?:pilot|trial|migration test|rollout|sample|test group|phase)\b",
    re.IGNORECASE,
)


def normalize_numeric_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace(",", "")).strip()


def authoritative_evidence_text(value: str) -> str:
    lines: list[str] = []
    include = True
    saw_debate_section = False
    for line in value.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?:CRITIC|ADVOCATE)\s+R\d+:", stripped, re.IGNORECASE):
            include = False
            saw_debate_section = True
            continue
        if re.fullmatch(
            r"(?:FACTS|ALTERNATIVES|COMPLIANCE):",
            stripped,
            re.IGNORECASE,
        ):
            include = True
            lines.append(line)
            continue
        if stripped.startswith("HANDOFF:"):
            include = False
        if include:
            lines.append(line)
    return "\n".join(lines) if saw_debate_section else value


def numeric_claim_supported(claim: str, normalized_evidence: str) -> bool:
    normalized = normalize_numeric_text(claim)
    if normalized in {"1", "2", "3"}:
        return True
    if normalized in normalized_evidence:
        return True
    count = re.match(
        r"^(\d+)\s*(?:-?\s*)(?:users?|people|persons?|members?|teams?|seats?)$",
        normalized,
    )
    if count and re.search(
        rf"\b{re.escape(count.group(1))}(?:-|\s)*(?:"
        r"users?|people|persons?|members?|teams?|seats?)\b",
        normalized_evidence,
    ):
        return True
    currency = re.match(r"^([$€£])\s*(\d[\d,]*(?:\.\d+)?)", claim.strip())
    if currency:
        amount = normalize_numeric_text("".join(currency.groups()))
        return amount in normalized_evidence
    return False


def remove_unsupported_numeric_claims(
    content: str,
    evidence: str,
) -> tuple[str, list[str]]:
    normalized_evidence = normalize_numeric_text(authoritative_evidence_text(evidence))
    cleaned_lines: list[str] = []
    unsupported_claims: list[str] = []
    omitted_claims: list[str] = []
    for line in content.splitlines():
        if HANDOFF_LINE_PATTERN.fullmatch(line):
            cleaned_lines.append(line)
            continue
        absence_claim = UNSUPPORTED_CERTIFICATION_ABSENCE_PATTERN.search(line)
        if (
            absence_claim
            and normalize_numeric_text(line) not in normalized_evidence
        ):
            unsupported_claims.append(absence_claim.group(0).strip())
            omitted_claims.append(absence_claim.group(0).strip())
            continue
        export_claim = CONTRADICTED_MARKDOWN_EXPORT_PATTERN.search(line)
        if (
            export_claim
            and normalize_numeric_text(line) not in normalized_evidence
        ):
            unsupported_claims.append(export_claim.group(0).strip())
            omitted_claims.append(export_claim.group(0).strip())
            continue
        numeric_matches = list(NUMERIC_CLAIM_PATTERN.finditer(line))
        exact_line_supported = normalize_numeric_text(line) in normalized_evidence
        prescriptive_matches = {
            match
            for match in numeric_matches
            if (
                PRESCRIPTIVE_NUMERIC_PATTERN.search(
                    line[max(0, match.start() - 50) : match.start()]
                )
                or PRESCRIPTIVE_NUMERIC_PATTERN.search(
                    line[match.end() : match.end() + 25]
                )
            )
        }
        unsupported_matches = [
            match
            for match in numeric_matches
            if (
                (match in prescriptive_matches and not exact_line_supported)
                or not numeric_claim_supported(match.group(0), normalized_evidence)
            )
        ]
        unsupported = [match.group(0).strip() for match in unsupported_matches]
        if unsupported:
            unsupported_claims.extend(unsupported)
            if all(match in prescriptive_matches for match in unsupported_matches):
                sanitized = line
                for match in reversed(unsupported_matches):
                    sanitized = sanitized[: match.start()] + sanitized[match.end() :]
                sanitized = re.sub(r"\s+", " ", sanitized)
                for _ in range(2):
                    sanitized = re.sub(
                        r"\b(?:with|for|of)\s+"
                        r"(?=(?:with|for|of|to|and)\b|[.,;:]|$)",
                        "",
                        sanitized,
                        flags=re.IGNORECASE,
                    )
                sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized).strip()
                if sanitized:
                    cleaned_lines.append(sanitized)
                continue
            omitted_claims.extend(unsupported)
            continue
        cleaned_lines.append(line)

    if omitted_claims:
        replacement = (
            "- DATA UNAVAILABLE: An unsupported factual claim was omitted because "
            "it was not present in CASE BRIEF / EVIDENCE DIGEST."
        )
        handoff_index = next(
            (
                index
                for index, line in enumerate(cleaned_lines)
                if HANDOFF_LINE_PATTERN.fullmatch(line)
            ),
            len(cleaned_lines),
        )
        cleaned_lines.insert(handoff_index, replacement)
    return "\n".join(cleaned_lines), unsupported_claims


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
            llm=make_llm(role, settings),
            instructions=load_prompt(role),
        ),
        agent_id=credentials.agent_id,
        api_key=credentials.api_key,
        rest_url=settings.band_rest_url,
        ws_url=settings.band_ws_url,
    )


class TextWorkerAdapter(SimpleAdapter[HistoryProvider]):
    """Generate text with any chat model, then send it through Band explicitly."""

    def __init__(self, role: str, llm: BaseChatModel, instructions: str) -> None:
        super().__init__()
        self.role = role
        self.llm = llm
        self.instructions = instructions

    def _fallback_handoff(self) -> str:
        return (
            f"HANDOFF: @Arbiter | STATE: {self.role.upper()}_COMPLETE "
            "| REQUEST: continue the protocol"
        )

    def _prepare_delivery(self, content: str) -> str:
        handoff_lines = HANDOFF_LINE_PATTERN.findall(content)
        handoff_lines = [
            re.sub(
                r"(?i)^\s*(?:\*\*)?HANDOFF:(?:\*\*)?\s*",
                "HANDOFF: ",
                line,
            ).strip().removesuffix("**").rstrip()
            for line in handoff_lines
        ]
        handoff = handoff_lines[-1] if handoff_lines else self._fallback_handoff()
        return prepare_worker_delivery(content, handoff=handoff)

    def _remove_unsupported_numeric_claims(
        self,
        content: str,
        evidence: str,
    ) -> str:
        if self.role not in {"advocate", "critic", "compliance"}:
            return content

        cleaned, unsupported = remove_unsupported_numeric_claims(content, evidence)
        if unsupported:
            logger.warning(
                "%s omitted unsupported factual claim(s): %s",
                self.role,
                ", ".join(unsupported),
            )
        return cleaned

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
        prompt = (
            f"{self.instructions}\n\n"
            "Return only the complete final Band message. Aim for 150-300 words "
            "using short bullets or one or two concise paragraphs. Make it "
            "self-contained and readable by both the other agents and a human "
            "reviewer. Never end with an unfinished thought. Do not wrap it in "
            "commentary.\n\n"
            f"Room context:\n{transcript}\n"
            f"Current request:\n[{msg.sender_name or msg.sender_type}]: {msg.content}"
        )
        response = await self.llm.ainvoke(prompt)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()
        content = self._remove_unsupported_numeric_claims(content, msg.content)
        await tools.send_message(
            self._prepare_delivery(content),
            mentions=["@Arbiter"],
        )
