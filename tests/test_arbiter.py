from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import FakeListChatModel

from src.agents.arbiter import ArbiterAdapter, DebatePhase


def test_extract_json_from_fence() -> None:
    payload = {"total": 80}
    text = f"```json\n{json.dumps(payload)}\n```"
    assert json.loads(ArbiterAdapter._extract_json(text)) == payload


class FakeTools:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[str] | None]] = []
        self.events: list[dict[str, Any]] = []
        self.added: list[str] = []

    async def send_message(
        self, content: str, mentions: list[str] | None = None
    ) -> None:
        self.messages.append((content, mentions))

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {"content": content, "message_type": message_type, "metadata": metadata}
        )

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> list[Any]:
        return []

    async def add_participant(self, identifier: str, role: str = "member") -> None:
        self.added.append(identifier)

    async def get_participants(self) -> list[str]:
        return self.added


def message(sender: str, content: str) -> PlatformMessage:
    return PlatformMessage(
        id=f"id-{sender}-{content[:5]}",
        room_id="room-1",
        content=content,
        sender_id=sender.lower(),
        sender_type="Agent" if sender != "Dmytro" else "User",
        sender_name=sender,
        message_type="text",
        metadata={},
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_full_debate_with_dynamic_compliance() -> None:
    verdict = {
        "case": "Notion for 50 people",
        "scores": {
            "value_for_money": 18,
            "capability_fit": 21,
            "risk_profile": 14,
            "alternatives": 16,
        },
        "total": 69,
        "recommendation": "BUY_WITH_CONDITIONS",
        "rationale": ["Strong fit", "Migration risk"],
        "conditions": ["Run a pilot"],
        "dissent": "Confluence has stronger governance.",
        "compliance_reviewed": True,
    }
    adapter = ArbiterAdapter(FakeListChatModel(responses=[json.dumps(verdict)]))
    tools = FakeTools()
    history = HistoryProvider(raw=[])

    turns = [
        ("Dmytro", "@Arbiter analyze Notion"),
        ("Researcher", "Facts with sources. @Arbiter"),
        ("Scout", "Alternatives compared. @Arbiter"),
        ("Critic", "COMPLIANCE CONCERN: data residency. @Arbiter"),
        ("Compliance", "Conditional GDPR fit. @Arbiter"),
        ("Advocate", "Risks can be mitigated. @Arbiter"),
        ("Critic", "Migration remains unresolved. @Arbiter"),
        ("Advocate", "Pilot and export test. @Arbiter"),
    ]
    for sender, content in turns:
        await adapter.on_message(
            message(sender, content),
            tools,
            history,
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

    assert tools.added == ["Compliance"]
    assert any("HANDOFF: @Compliance" in item[0] for item in tools.messages)
    assert any("FINAL VERDICT" in item[0] for item in tools.messages)
    assert any("```json" in item[0] for item in tools.messages)
    critic_handoff = next(
        content for content, _ in tools.messages if "ROUND 1: CHALLENGE" in content
    )
    assert "CASE BRIEF:" in critic_handoff
    assert "EVIDENCE DIGEST:" in critic_handoff
    assert "Facts with sources" in critic_handoff
    assert "Alternatives compared" in critic_handoff
    assert all(len(content.split()) <= 200 for content, _ in tools.messages)
    assert any(item["metadata"]["kind"] == "verdict" for item in tools.events)
    assert adapter.states["room-1"].phase == DebatePhase.IDLE


@pytest.mark.asyncio
async def test_unexpected_sender_does_not_advance() -> None:
    adapter = ArbiterAdapter(FakeListChatModel(responses=["{}"]))
    tools = FakeTools()
    history = HistoryProvider(raw=[])
    await adapter.on_message(
        message("Dmytro", "@Arbiter analyze Notion"),
        tools,
        history,
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-2",
    )
    await adapter.on_message(
        message("Scout", "I answered early"),
        tools,
        history,
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-2",
    )
    assert adapter.states["room-2"].phase == DebatePhase.RESEARCH


@pytest.mark.asyncio
async def test_ping_does_not_open_case() -> None:
    adapter = ArbiterAdapter(FakeListChatModel(responses=["unused"]))
    tools = FakeTools()
    await adapter.on_message(
        message("Dmytro", "@Arbiter ping"),
        tools,
        HistoryProvider(raw=[]),
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-ping",
    )
    assert adapter.states["room-ping"].phase == DebatePhase.IDLE
    assert tools.messages[-1][0].startswith("⚖️ PONG")


@pytest.mark.asyncio
async def test_mention_chain_returns_to_idle() -> None:
    adapter = ArbiterAdapter(FakeListChatModel(responses=["unused"]))
    tools = FakeTools()
    history = HistoryProvider(raw=[])
    await adapter.on_message(
        message("Dmytro", "@Arbiter передай привіт Researcher"),
        tools,
        history,
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-chain",
    )
    assert adapter.states["room-chain"].phase == DebatePhase.CHAIN_RESEARCHER
    await adapter.on_message(
        message("Researcher", "🔍 Hello. @Arbiter"),
        tools,
        history,
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-chain",
    )
    assert adapter.states["room-chain"].phase == DebatePhase.IDLE
    assert "CONNECTION TEST PASSED" in tools.messages[-1][0]


@pytest.mark.asyncio
async def test_timeout_reminds_once_then_continues() -> None:
    adapter = ArbiterAdapter(
        FakeListChatModel(responses=["unused"]),
        turn_timeout=0.02,
    )
    tools = FakeTools()
    await adapter.on_message(
        message("Dmytro", "@Arbiter analyze Notion"),
        tools,
        HistoryProvider(raw=[]),
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-timeout",
    )
    await asyncio.sleep(0.025)
    assert adapter.states["room-timeout"].phase == DebatePhase.RESEARCH
    assert sum("TIMEOUT REMINDER" in item[0] for item in tools.messages) == 1
    await asyncio.sleep(0.025)
    assert adapter.states["room-timeout"].phase == DebatePhase.SCOUTING
    assert adapter.states["room-timeout"].missing_agents == ["Researcher"]
    adapter._cancel_timeout("room-timeout")


@pytest.mark.asyncio
async def test_reply_cancels_previous_phase_timeout() -> None:
    adapter = ArbiterAdapter(
        FakeListChatModel(responses=["unused"]),
        turn_timeout=0.03,
    )
    tools = FakeTools()
    history = HistoryProvider(raw=[])
    await adapter.on_message(
        message("Dmytro", "@Arbiter analyze Notion"),
        tools,
        history,
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-cancel",
    )
    await asyncio.sleep(0.01)
    await adapter.on_message(
        message("Researcher", "Facts ready. @Arbiter"),
        tools,
        history,
        None,
        None,
        is_session_bootstrap=False,
        room_id="room-cancel",
    )
    await asyncio.sleep(0.025)
    assert not any("TIMEOUT REMINDER" in content for content, _ in tools.messages)
    assert adapter.states["room-cancel"].phase == DebatePhase.SCOUTING
    adapter._cancel_timeout("room-cancel")


def test_clean_case_replaces_raw_platform_mention() -> None:
    msg = message(
        "Dmytro",
        "@[[12345678-1234-1234-1234-123456789abc]] Analyze Notion for 50 users",
    )
    assert ArbiterAdapter._clean_case(msg) == ("@Dmytro Analyze Notion for 50 users")


def test_snapshot_strips_old_handoff_and_keeps_sources() -> None:
    source = (
        "FACTS: Notion has tiered plans.\n"
        "SOURCES: https://www.notion.com/pricing https://example.com/review\n"
        "HANDOFF: @Arbiter | STATE: RESEARCH_COMPLETE | REQUEST: continue"
    )
    snapshot = ArbiterAdapter._snapshot(source, max_urls=2)
    assert "HANDOFF:" not in snapshot
    assert "https://www.notion.com/pricing" in snapshot
    assert "https://example.com/review" in snapshot


def test_cap_message_keeps_single_handoff_under_200_words() -> None:
    content = (
        "ROUND\n"
        + " ".join(["evidence"] * 300)
        + "\nHANDOFF: @Critic | STATE: ROUND_1 | REQUEST: objections"
    )
    capped = ArbiterAdapter._cap_message(content)
    assert len(capped.split()) <= 195
    assert capped.count("HANDOFF:") == 1
    assert capped.endswith("HANDOFF: @Critic | STATE: ROUND_1 | REQUEST: objections")
