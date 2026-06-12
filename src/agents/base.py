from __future__ import annotations

from typing import Any

from band import AdapterFeatures, Agent, Emit
from band.adapters import LangGraphAdapter

from src.common.config import PROJECT_ROOT, AgentCredentials, Settings
from src.common.llm import make_llm


def load_prompt(role: str) -> str:
    prompt_dir = PROJECT_ROOT / "src" / "prompts"
    shared = (prompt_dir / "shared_protocol.md").read_text(encoding="utf-8")
    role_prompt = (prompt_dir / f"{role}.md").read_text(encoding="utf-8")
    return f"{shared}\n\n# Role-specific instructions\n{role_prompt}".strip()


def build_langgraph_agent(
    role: str,
    credentials: AgentCredentials,
    settings: Settings,
    *,
    tools: list[Any] | None = None,
) -> Agent:
    adapter = LangGraphAdapter(
        llm=make_llm(role, settings),
        custom_section=load_prompt(role),
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
