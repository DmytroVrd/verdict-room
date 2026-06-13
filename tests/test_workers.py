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
