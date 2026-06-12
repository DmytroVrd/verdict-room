from __future__ import annotations

import logging
from typing import Any

from band import AdapterFeatures, Agent, Emit
from band.adapters import LangGraphAdapter
from band.core.protocols import AgentToolsProtocol
from band.core.types import PlatformMessage

from src.common.config import PROJECT_ROOT, AgentCredentials, Settings
from src.common.llm import make_llm

logger = logging.getLogger(__name__)

DELIVERY_CONTRACT = """

# Delivery contract

You MUST deliver the completed response by calling `band_send_message` exactly
once with `@Arbiter` mentioned. Do not finish by returning plain assistant text.
""".strip()

ROLE_DELIVERY = {
    "researcher": (
        "🔍",
        "RESEARCH_COMPLETE",
        "review facts and task @Scout",
    ),
    "scout": (
        "🕵️",
        "SCOUTING_COMPLETE",
        "begin ROUND_1 with @Critic",
    ),
}


def load_prompt(role: str) -> str:
    prompt_dir = PROJECT_ROOT / "src" / "prompts"
    shared = (prompt_dir / "shared_protocol.md").read_text(encoding="utf-8")
    role_prompt = (prompt_dir / f"{role}.md").read_text(encoding="utf-8")
    return f"{shared}\n\n# Role-specific instructions\n{role_prompt}".strip()


class ReliableLangGraphAdapter(LangGraphAdapter):
    """Guarantee that a tool-driven worker delivers one response to Arbiter."""

    def __init__(self, *, role: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.role = role
        self._delivery_succeeded: dict[str, bool] = {}
        self._final_text: dict[str, str] = {}

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: Any,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        self._delivery_succeeded[room_id] = False
        self._final_text.pop(room_id, None)
        try:
            await super().on_message(
                msg,
                tools,
                history,
                participants_msg,
                contacts_msg,
                is_session_bootstrap=is_session_bootstrap,
                room_id=room_id,
            )
            if not self._delivery_succeeded.get(room_id, False):
                logger.info(
                    "%s completed without band_send_message; using delivery fallback",
                    self.role,
                )
                content = self._prepare_delivery(self._final_text.get(room_id, ""))
                await tools.send_message(content, mentions=["@Arbiter"])
        finally:
            self._delivery_succeeded.pop(room_id, None)
            self._final_text.pop(room_id, None)

    async def _handle_stream_event(
        self,
        event: Any,
        room_id: str,
        tools: AgentToolsProtocol,
    ) -> None:
        if isinstance(event, dict):
            event_type = event.get("event")
            name = event.get("name")
            data = event.get("data")
            if event_type == "on_tool_end" and name == "band_send_message":
                is_error = isinstance(data, dict) and bool(data.get("error"))
                if not is_error:
                    self._delivery_succeeded[room_id] = True
            elif event_type in {"on_chat_model_end", "on_chain_end"}:
                text = self._extract_text(data)
                if text:
                    self._final_text[room_id] = text
        await super()._handle_stream_event(event, room_id, tools)

    def _prepare_delivery(self, content: str) -> str:
        marker, state, request = ROLE_DELIVERY[self.role]
        handoff = f"HANDOFF: @Arbiter | STATE: {state} | REQUEST: {request}"
        cleaned = " ".join(content.split())
        if not cleaned:
            cleaned = f"{marker} GAPS: No usable model response was produced."
        elif not cleaned.startswith(marker):
            cleaned = f"{marker} {cleaned}"
        if "HANDOFF:" in cleaned:
            cleaned = cleaned.split("HANDOFF:", 1)[0].rstrip()
        budget = 200 - len(handoff.split())
        words = cleaned.split()
        if len(words) > budget:
            cleaned = " ".join(words[:budget]).rstrip(" ,;:")
        return f"{cleaned}\n\n{handoff}"

    @classmethod
    def _extract_text(cls, value: Any) -> str:
        if isinstance(value, dict):
            for key in ("output", "messages"):
                text = cls._extract_text(value.get(key))
                if text:
                    return text
            return ""
        if isinstance(value, (list, tuple)):
            for item in reversed(value):
                text = cls._extract_text(item)
                if text:
                    return text
            return ""
        content = getattr(value, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return " ".join(part for part in parts if part).strip()
        return value.strip() if isinstance(value, str) else ""


def build_langgraph_agent(
    role: str,
    credentials: AgentCredentials,
    settings: Settings,
    *,
    tools: list[Any] | None = None,
) -> Agent:
    adapter = ReliableLangGraphAdapter(
        role=role,
        llm=make_llm(role, settings),
        custom_section=f"{load_prompt(role)}\n\n{DELIVERY_CONTRACT}",
        additional_tools=tools or [],
        features=AdapterFeatures(emit={Emit.EXECUTION}),
    )
    return Agent.create(
        adapter=adapter,
        agent_id=credentials.agent_id,
        api_key=credentials.api_key,
        rest_url=settings.band_rest_url,
        ws_url=settings.band_ws_url,
    )
