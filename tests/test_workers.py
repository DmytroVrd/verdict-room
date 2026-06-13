from __future__ import annotations

from datetime import UTC, datetime

import pytest
from band.adapters import LangGraphAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.agents.base import EMERGENCY_WORD_LIMIT, ReliableLangGraphAdapter, load_prompt
from src.agents.workers import (
    DEBATE_EVIDENCE_OUTPUT_CONTRACT,
    DEBATE_REVIEW_CONTRACT,
    GroundedCondition,
    GroundedDebateResponse,
    GroundedEvidence,
    GroundedHypothesis,
    TextWorkerAdapter,
)


class FakeTools:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[str] | None]] = []

    async def send_message(
        self, content: str, mentions: list[str] | None = None
    ) -> None:
        self.messages.append((content, mentions))


class FakeGroundedLLM:
    def __init__(self, response: GroundedDebateResponse) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> GroundedDebateResponse:
        self.prompts.append(prompt)
        return self.response


def message(sender: str, content: str) -> PlatformMessage:
    return PlatformMessage(
        id="message-1",
        room_id="room-1",
        content=content,
        sender_id="arbiter",
        sender_type="Agent",
        sender_name=sender,
        message_type="text",
        metadata={},
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_text_worker_sends_response_with_arbiter_mention() -> None:
    handoff = (
        "HANDOFF: @Arbiter | STATE: ROUND_1_ADVOCATE_COMPLETE "
        "| REQUEST: continue the protocol"
    )
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["unused"]),
        grounded_llm=FakeGroundedLLM(
            GroundedDebateResponse(
                compliance_concern=False,
                evidence_primary=GroundedEvidence(quote="Supported source text."),
                evidence_secondary=GroundedEvidence(quote="Second source text."),
                evidence_tertiary=GroundedEvidence(quote="Third source text."),
                hypothesis_primary=GroundedHypothesis(area="cost"),
                hypothesis_secondary=GroundedHypothesis(area="migration"),
                condition=GroundedCondition(area="security"),
            )
        ),
        instructions="Be concise.",
    )
    tools = FakeTools()
    await adapter.on_message(
        message("Arbiter", "@Advocate respond"),
        tools,
        HistoryProvider(raw=[]),
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-1",
    )
    assert tools.messages == [
        (
            "✅\n"
            '- EVIDENCE | (evidence: "Supported source text.")\n'
            '- EVIDENCE | (evidence: "Second source text.")\n'
            '- EVIDENCE | (evidence: "Third source text.")\n'
            "- HYPOTHESIS | QUESTION: Could the documented pricing create "
            "budget pressure? | CHECK: Compare equivalent vendor quotes and "
            "contract terms.\n"
            "- HYPOTHESIS | QUESTION: Might migration affect continuity or data "
            "quality? | CHECK: Run a representative import and export pilot.\n"
            "- CONDITION | CHECK: Map documented controls to the buyer security "
            "requirements.\n\n"
            f"{handoff}",
            ["@Arbiter"],
        )
    ]


def test_grounded_render_keeps_complete_items_without_censoring() -> None:
    adapter = TextWorkerAdapter(
        role="critic",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    response = GroundedDebateResponse(
        compliance_concern=True,
        evidence_primary=GroundedEvidence(quote="Business: $20/user/month"),
        evidence_secondary=GroundedEvidence(quote="A second exact fact."),
        evidence_tertiary=GroundedEvidence(quote="A third exact fact."),
        hypothesis_primary=GroundedHypothesis(area="cost"),
        hypothesis_secondary=GroundedHypothesis(area="migration"),
        condition=GroundedCondition(area="security"),
    )
    rendered = adapter._render_grounded(response)
    assert '$20/user/month")' in rendered
    assert "Could the documented pricing create budget pressure?" in rendered
    assert "🚨 COMPLIANCE CONCERN" in rendered
    assert "DATA UNAVAILABLE" not in rendered


@pytest.mark.parametrize(
    ("role", "case_text", "state"),
    [
        ("critic", "ROUND 1: CHALLENGE", "ROUND_1_CRITIC_COMPLETE"),
        ("critic", "ROUND 2: REBUTTAL", "ROUND_2_CRITIC_COMPLETE"),
        ("advocate", "ROUND 1: DEFENSE", "ROUND_1_ADVOCATE_COMPLETE"),
        ("advocate", "ROUND 2: CLOSING", "ROUND_2_ADVOCATE_COMPLETE"),
        ("compliance", "SPECIALIST REVIEW", "COMPLIANCE_COMPLETE"),
    ],
)
def test_grounded_handoff_matches_role_and_round(
    role: str,
    case_text: str,
    state: str,
) -> None:
    adapter = TextWorkerAdapter(
        role=role,
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    assert state in adapter._handoff_for_request(case_text)


def test_reliable_adapter_does_not_truncate_normal_scout_response() -> None:
    adapter = ReliableLangGraphAdapter(
        role="scout",
        llm=FakeListChatModel(responses=["unused"]),
    )
    original = " ".join(["fact"] * 250) + "."
    content = adapter._prepare_delivery(original)
    assert content.startswith("🕵️")
    assert original in content
    assert len(content.split()) > 200
    assert content.count("HANDOFF:") == 1
    assert "STATE: SCOUTING_COMPLETE" in content


def test_reliable_adapter_emergency_guard_ends_at_sentence_boundary() -> None:
    adapter = ReliableLangGraphAdapter(
        role="scout",
        llm=FakeListChatModel(responses=["unused"]),
    )
    content = adapter._prepare_delivery(
        " ".join(f"Finding {index} is fully supported." for index in range(180))
    )
    body, handoff = content.rsplit("\n\n", 1)
    assert body.endswith(".")
    assert len(content.split()) <= EMERGENCY_WORD_LIMIT
    assert content.count("HANDOFF:") == 1
    assert handoff.startswith("HANDOFF: @Arbiter")


def test_reliable_adapter_extracts_final_ai_message() -> None:
    text = ReliableLangGraphAdapter._extract_text(
        {"output": {"messages": [AIMessage(content="final answer")]}}
    )
    assert text == "final answer"


def test_reliable_adapter_replaces_duplicate_handoff() -> None:
    adapter = ReliableLangGraphAdapter(
        role="researcher",
        llm=FakeListChatModel(responses=["unused"]),
    )
    content = adapter._prepare_delivery(
        "🔍 Facts.\nHANDOFF: @Someone | STATE: WRONG | REQUEST: continue"
    )
    assert content.count("HANDOFF:") == 1
    assert "STATE: RESEARCH_COMPLETE" in content
    assert "@Someone" not in content


def test_text_worker_guard_preserves_one_existing_handoff() -> None:
    adapter = TextWorkerAdapter(
        role="critic",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    first = "HANDOFF: @Arbiter | STATE: WRONG | REQUEST: ignore"
    expected = (
        "HANDOFF: @Arbiter | STATE: ROUND_1_CRITIC_COMPLETE "
        "| REQUEST: task @Advocate"
    )
    body = " ".join(
        f"Risk {index} has a complete explanation." for index in range(180)
    )
    content = adapter._prepare_delivery(f"{body}\n{first}\n{expected}")
    response_body, handoff = content.rsplit("\n\n", 1)
    assert response_body.endswith(".")
    assert len(content.split()) <= EMERGENCY_WORD_LIMIT
    assert content.count("HANDOFF:") == 1
    assert handoff == expected


def test_text_worker_normalizes_markdown_handoff() -> None:
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    content = adapter._prepare_delivery(
        "✅ Complete defense.\n"
        "**HANDOFF:** @Arbiter | STATE: ROUND_1_ADVOCATE_COMPLETE "
        "| REQUEST: continue**"
    )
    assert content.count("HANDOFF:") == 1
    assert "**HANDOFF:**" not in content
    assert content.endswith(
        "HANDOFF: @Arbiter | STATE: ROUND_1_ADVOCATE_COMPLETE | REQUEST: continue"
    )


def test_shared_prompt_uses_soft_length_guidance() -> None:
    prompt = load_prompt("critic")
    assert "roughly 150-300 words" in prompt
    assert "not a hard limit" in prompt
    assert "stay at or below 200 words" not in prompt


@pytest.mark.parametrize("role", ["advocate", "critic", "compliance"])
def test_debate_prompts_forbid_unsupported_facts(role: str) -> None:
    prompt = load_prompt(role)
    assert "CASE BRIEF" in prompt
    assert "EVIDENCE DIGEST" in prompt
    assert "Data unavailable in CASE BRIEF / EVIDENCE DIGEST." in prompt
    assert "HYPOTHESIS" in prompt
    assert "Never invent" in prompt
    assert '(evidence: "...")' in prompt
    assert "exact quote" in prompt


@pytest.mark.parametrize("role", ["advocate", "critic"])
def test_debate_prompts_require_qualitative_fallback_for_missing_numbers(
    role: str,
) -> None:
    prompt = load_prompt(role)
    assert "qualitative wording" in prompt
    assert "Never derive annual totals from monthly prices" in prompt
    assert "Never infer that an omitted integration" in prompt
    assert "`lacks`, `without`, `does not have`" in prompt
    assert "Do not name a certification containing a number" in prompt
    assert "Do not repeat buyer headcount" in prompt
    assert "verbatim digest substring" in prompt
    assert "Never normalize `user` to `member`" in prompt


def test_final_debate_output_contract_repeats_critical_rules_near_request() -> None:
    assert "Do not repeat the buyer headcount" in DEBATE_EVIDENCE_OUTPUT_CONTRACT
    assert "(evidence:" in DEBATE_EVIDENCE_OUTPUT_CONTRACT
    assert "never from DEBATE CONTEXT" in DEBATE_EVIDENCE_OUTPUT_CONTRACT
    assert "certification name containing a number" in (
        DEBATE_EVIDENCE_OUTPUT_CONTRACT
    )
    assert "Never use `...` or an ellipsis" in DEBATE_EVIDENCE_OUTPUT_CONTRACT


def test_debate_review_contract_uses_prompt_review_not_code_censor() -> None:
    assert "STRUCTURED FINAL FORMAT" in DEBATE_REVIEW_CONTRACT
    assert "only inside" in DEBATE_REVIEW_CONTRACT
    assert "one exact contiguous source line substring" in DEBATE_REVIEW_CONTRACT
    assert "Omission is not evidence" in DEBATE_REVIEW_CONTRACT
    assert "Never assert a new product fact" in DEBATE_REVIEW_CONTRACT
    assert "Use no facts from DEBATE CONTEXT" in DEBATE_REVIEW_CONTRACT


@pytest.mark.asyncio
async def test_reliable_adapter_delivers_plain_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_on_message(
        adapter: ReliableLangGraphAdapter,
        *args: object,
        room_id: str,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        adapter._final_text[room_id] = "Two sourced alternatives were compared."

    monkeypatch.setattr(LangGraphAdapter, "on_message", fake_on_message)
    adapter = ReliableLangGraphAdapter(
        role="scout",
        llm=FakeListChatModel(responses=["unused"]),
    )
    tools = FakeTools()
    await adapter.on_message(
        message("Arbiter", "@Scout compare alternatives"),
        tools,
        HistoryProvider(raw=[]),
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-1",
    )
    assert len(tools.messages) == 1
    content, mentions = tools.messages[0]
    assert "Two sourced alternatives were compared." in content
    assert "STATE: SCOUTING_COMPLETE" in content
    assert mentions == ["@Arbiter"]
