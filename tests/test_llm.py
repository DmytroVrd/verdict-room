from __future__ import annotations

from pathlib import Path

from src.common.config import (
    DEFAULT_FALLBACK_MODELS,
    configured_provider_keys,
    load_settings,
)
from src.common.llm import (
    ModelSpec,
    _build_model,
    _temperature_for_role,
    available_model_specs,
)


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


def test_only_aiml_and_featherless_providers_are_configured(monkeypatch) -> None:
    monkeypatch.setenv("AIML_API_KEY", "test")
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "legacy")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy")
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy")

    assert configured_provider_keys() == {
        "aiml": True,
        "featherless": True,
    }
    assert DEFAULT_FALLBACK_MODELS == (
        "aiml:openai/gpt-4.1-mini",
        "featherless:deepseek-ai/DeepSeek-V3.2",
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


def test_debate_roles_use_zero_temperature(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_TEMPERATURE=0.2\n", encoding="utf-8")
    settings = load_settings(env_path)
    for role in ("advocate", "critic", "compliance"):
        assert _temperature_for_role(role, settings) == 0.0
    for role in ("arbiter", "researcher", "scout"):
        assert _temperature_for_role(role, settings) == 0.2


def test_arbiter_model_can_be_built_at_debate_temperature(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AIML_API_KEY=test\nLLM_TEMPERATURE=0.2\n",
        encoding="utf-8",
    )
    settings = load_settings(env_path)
    monkeypatch.setenv("AIML_API_KEY", "test")
    model = _build_model(
        ModelSpec("aiml", "openai/gpt-4.1-mini"),
        settings,
        role="critic",
    )
    assert model.temperature == 0.0
