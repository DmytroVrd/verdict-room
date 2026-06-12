from __future__ import annotations

from datetime import UTC, datetime

import pytest
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import FakeListChatModel

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
