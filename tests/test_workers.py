from __future__ import annotations

from datetime import UTC, datetime

import pytest
from band.adapters import LangGraphAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.agents.base import ReliableLangGraphAdapter
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
    adapter = TextWorkerAdapter(
        role="advocate",
        llm=FakeListChatModel(responses=["✅ Defense complete. @Arbiter"]),
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
    assert tools.messages == [("✅ Defense complete. @Arbiter", ["@Arbiter"])]


def test_reliable_adapter_prepares_bounded_scout_handoff() -> None:
    adapter = ReliableLangGraphAdapter(
        role="scout",
        llm=FakeListChatModel(responses=["unused"]),
    )
    content = adapter._prepare_delivery(" ".join(["fact"] * 250))
    assert content.startswith("🕵️")
    assert len(content.split()) <= 200
    assert content.count("HANDOFF:") == 1
    assert "STATE: SCOUTING_COMPLETE" in content


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
