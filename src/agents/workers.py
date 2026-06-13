from __future__ import annotations

from typing import Any

from band import Agent
from band.core.protocols import AgentToolsProtocol
from band.core.simple_adapter import SimpleAdapter
from band.core.types import HistoryProvider, PlatformMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from src.agents.base import build_langgraph_agent, load_prompt, prepare_worker_delivery
from src.common.config import AgentCredentials, Settings
from src.common.llm import make_llm


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
        handoff_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip().startswith("HANDOFF:")
        ]
        handoff = handoff_lines[-1] if handoff_lines else self._fallback_handoff()
        return prepare_worker_delivery(content, handoff=handoff)

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
        await tools.send_message(
            self._prepare_delivery(content),
            mentions=["@Arbiter"],
        )
