from __future__ import annotations

from pathlib import Path

from src.common.config import load_settings
from src.common.llm import ModelSpec, available_model_specs


def test_v2_default_model_mapping(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AIML_API_KEY=test\nFEATHERLESS_API_KEY=test\n", encoding="utf-8"
    )
    for name in (
        "ARBITER_MODEL",
        "RESEARCHER_MODEL",
        "SCOUT_MODEL",
        "ADVOCATE_MODEL",
        "CRITIC_MODEL",
        "COMPLIANCE_MODEL",
        "LLM_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(env_path)
    assert settings.model_specs["arbiter"] == "aiml:openai/gpt-4.1-mini"
    assert settings.model_specs["advocate"] == ("featherless:deepseek-ai/DeepSeek-V3.2")
    assert available_model_specs("critic", settings)[0] == ModelSpec(
        provider="featherless",
        model="deepseek-ai/DeepSeek-V3.1-Terminus",
    )


def test_model_spec_preserves_slashes() -> None:
    assert ModelSpec.parse("featherless:deepseek-ai/DeepSeek-V3.2") == ModelSpec(
        provider="featherless",
        model="deepseek-ai/DeepSeek-V3.2",
    )


def test_tool_roles_skip_featherless_fallback(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AIML_API_KEY=test\nFEATHERLESS_API_KEY=test\n"
        "LLM_FALLBACK_MODELS=featherless:deepseek-ai/DeepSeek-V3.2,"
        "aiml:openai/gpt-4.1-mini\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RESEARCHER_MODEL", raising=False)
    settings = load_settings(env_path)
    assert all(
        spec.provider != "featherless"
        for spec in available_model_specs("researcher", settings)
    )
