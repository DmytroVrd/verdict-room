from __future__ import annotations

from datetime import UTC, datetime

import pytest
from band.adapters import LangGraphAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.agents.base import EMERGENCY_WORD_LIMIT, ReliableLangGraphAdapter, load_prompt
from src.agents.workers import TextWorkerAdapter


class FakeTools:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[str] | None]] = []

    async def send_message(
        self, content: str, mentions: list[str] | None = None
    ) -> None:
        self.messages.append((content, mentions))


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
        llm=FakeListChatModel(responses=[f"✅ Defense complete.\n{handoff}"]),
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
        (f"✅ Defense complete.\n\n{handoff}", ["@Arbiter"])
    ]


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


def test_advocate_omits_unsupported_calculations_and_pilot_size() -> None:
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    evidence = (
        "CASE BRIEF: 50-person company.\n"
        "EVIDENCE DIGEST: Notion costs $20/user/month."
    )
    response = (
        "✅ EVIDENCE: Notion costs $20/user/month.\n"
        "- Derived annual cost is $12,000.\n"
        "- Pilot with 5-10 people for one billing cycle.\n"
        "HANDOFF: @Arbiter | STATE: ROUND_1_ADVOCATE_COMPLETE | REQUEST: continue"
    )
    cleaned = adapter._remove_unsupported_numeric_claims(response, evidence)
    assert "$20/user/month" in cleaned
    assert "$12,000" not in cleaned
    assert "5-10 people" not in cleaned
    assert "one billing cycle" not in cleaned
    assert cleaned.count("DATA UNAVAILABLE") == 1
    assert "HANDOFF: @Arbiter" in cleaned


def test_advocate_omits_hyphenated_unsupported_duration() -> None:
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    cleaned = adapter._remove_unsupported_numeric_claims(
        "✅ Run a 3-month trial.\nHANDOFF: @Arbiter | STATE: DONE | REQUEST: next",
        "CASE BRIEF: 50-person team. EVIDENCE DIGEST: Business plan is monthly.",
    )
    assert "3-month" not in cleaned
    assert "Run a trial." in cleaned


def test_advocate_scrubs_prescriptive_numbers_even_when_token_exists_elsewhere() -> None:
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    cleaned = adapter._remove_unsupported_numeric_claims(
        (
            "Conduct a fixed-price pilot for 10 users to validate feature fit.\n"
            "Perform a controlled migration test for one team to assess disruption."
        ),
        (
            "CASE BRIEF: 50-person company.\n"
            "EVIDENCE DIGEST: The free plan supports up to 10 users."
        ),
    )
    assert "10 users" not in cleaned
    assert "one team" not in cleaned
    assert "Conduct a fixed-price pilot to validate feature fit." in cleaned
    assert "Perform a controlled migration test to assess disruption." in cleaned


def test_advocate_keeps_case_size_when_pilot_appears_later_in_same_line() -> None:
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    response = (
        "For a 50-person company, the risks define due diligence: "
        "a pilot for workflow fit."
    )
    assert (
        adapter._remove_unsupported_numeric_claims(
            response,
            "CASE BRIEF: A 50-person company.",
        )
        == response
    )


def test_critic_omits_export_claim_contradicted_by_evidence() -> None:
    adapter = TextWorkerAdapter(
        role="critic",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    cleaned = adapter._remove_unsupported_numeric_claims(
        "EVIDENCE: Notion export is limited to markdown.",
        "EVIDENCE DIGEST: Notion exports to markdown, HTML, CSV, and PDF.",
    )
    assert "limited to markdown" not in cleaned
    assert "DATA UNAVAILABLE" in cleaned


def test_debate_claim_does_not_become_authoritative_by_repetition() -> None:
    adapter = TextWorkerAdapter(
        role="critic",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    cleaned = adapter._remove_unsupported_numeric_claims(
        "EVIDENCE: Notion export is limited to markdown.",
        (
            "CASE BRIEF:\nCompare workspace tools.\n"
            "EVIDENCE DIGEST:\n"
            "ADVOCATE R1:\nNotion export is limited to markdown.\n"
            "FACTS:\nNotion exports to markdown, HTML, CSV, and PDF."
        ),
    )
    assert "limited to markdown" not in cleaned
    assert "DATA UNAVAILABLE" in cleaned


def test_critic_omits_unsupported_certification_absence() -> None:
    adapter = TextWorkerAdapter(
        role="critic",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    cleaned = adapter._remove_unsupported_numeric_claims(
        (
            "Notion lacks explicit certifications like SOC 2, which Slite holds.\n"
            "HANDOFF: @Arbiter | STATE: DONE | REQUEST: next"
        ),
        "EVIDENCE DIGEST: Slite has SOC 2. Notion certification data is unavailable.",
    )
    assert "Notion lacks" not in cleaned
    assert "DATA UNAVAILABLE" in cleaned
    assert "HANDOFF: @Arbiter" in cleaned


def test_critic_keeps_structural_and_supported_numbers() -> None:
    adapter = TextWorkerAdapter(
        role="critic",
        llm=FakeListChatModel(responses=["unused"]),
        instructions="Be concise.",
    )
    evidence = "CASE BRIEF: 50-person team. EVIDENCE DIGEST: SOC 2."
    response = (
        "❌ Top 3 risks for 50 users.\n"
        "1. SOC 2/HIPAA is documented for the alternative."
    )
    assert adapter._remove_unsupported_numeric_claims(response, evidence) == response


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
