from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROLES = ("arbiter", "researcher", "scout", "advocate", "critic", "compliance")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = {
    "arbiter": "groq:openai/gpt-oss-120b",
    "researcher": "groq:openai/gpt-oss-120b",
    "scout": "openrouter:openrouter/free",
    "advocate": "gemini:gemini-3.5-flash",
    "critic": "aiml:openai/gpt-4.1-mini",
    "compliance": "gemini:gemini-3.5-flash",
}
DEFAULT_FALLBACK_MODELS = (
    "groq:openai/gpt-oss-120b",
    "gemini:gemini-3.5-flash",
    "openrouter:openrouter/free",
)


@dataclass(frozen=True)
class AgentCredentials:
    agent_id: str
    api_key: str


@dataclass(frozen=True)
class Settings:
    band_rest_url: str
    band_ws_url: str
    llm_temperature: float
    llm_timeout_seconds: float
    debate_turn_timeout: float
    start_compliance: bool
    model_specs: dict[str, str]
    fallback_models: tuple[str, ...]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    model_specs = {
        role: os.getenv(f"{role.upper()}_MODEL", DEFAULT_MODELS[role]).strip()
        for role in ROLES
    }
    fallback_models = tuple(
        item.strip()
        for item in os.getenv(
            "LLM_FALLBACK_MODELS", ",".join(DEFAULT_FALLBACK_MODELS)
        ).split(",")
        if item.strip()
    )
    return Settings(
        band_rest_url=os.getenv("BAND_REST_URL", "https://app.band.ai").rstrip("/"),
        band_ws_url=os.getenv(
            "BAND_WS_URL", "wss://app.band.ai/api/v1/socket/websocket"
        ),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        debate_turn_timeout=float(os.getenv("DEBATE_TURN_TIMEOUT", "120")),
        start_compliance=_as_bool(os.getenv("START_COMPLIANCE"), default=True),
        model_specs=model_specs,
        fallback_models=fallback_models,
    )


def load_agent_credentials(
    config_path: Path | None = None,
) -> dict[str, AgentCredentials]:
    path = config_path or PROJECT_ROOT / "agent_config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy agent_config.yaml.example to agent_config.yaml."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    credentials: dict[str, AgentCredentials] = {}
    for role in ROLES:
        item = raw.get(role)
        if not item:
            continue
        agent_id = str(item.get("agent_id", "")).strip()
        api_key = str(item.get("api_key", "")).strip()
        if not agent_id or not api_key:
            raise ValueError(f"Incomplete Band credentials for role '{role}'")
        credentials[role] = AgentCredentials(agent_id=agent_id, api_key=api_key)
    return credentials


def configured_provider_keys() -> dict[str, bool]:
    return {
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "gemini": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "aiml": bool(os.getenv("AIML_API_KEY")),
    }
