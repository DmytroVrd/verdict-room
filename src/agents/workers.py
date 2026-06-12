from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from src.agents.base import build_langgraph_agent
from src.common.config import AgentCredentials, Settings


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
    tools = _research_tools() if role in {"researcher", "scout"} else []
    return build_langgraph_agent(role, credentials, settings, tools=tools)
