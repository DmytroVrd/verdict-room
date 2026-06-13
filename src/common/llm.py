from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.common.config import Settings, configured_provider_keys, load_settings

DEBATE_ROLES = frozenset({"advocate", "critic", "compliance"})


class LLMConfigurationError(RuntimeError):
    """Raised when no configured provider can serve a role."""


class FallbackChatModel(BaseChatModel):
    """Tool-compatible provider fallback for LangChain agents."""

    candidates: list[Runnable[Any, BaseMessage]]

    @property
    def _llm_type(self) -> str:
        return "verdict-room-fallback"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"candidate_count": len(self.candidates)}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        last_error: Exception | None = None
        for candidate in self.candidates:
            try:
                message = candidate.invoke(messages, stop=stop, **kwargs)
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:
                last_error = exc
        raise last_error or LLMConfigurationError("No fallback candidates configured")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        last_error: Exception | None = None
        for candidate in self.candidates:
            try:
                message = await candidate.ainvoke(messages, stop=stop, **kwargs)
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:
                last_error = exc
        raise last_error or LLMConfigurationError("No fallback candidates configured")

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "FallbackChatModel":
        bound = [
            candidate.bind_tools(tools, tool_choice=tool_choice, **kwargs)
            for candidate in self.candidates
        ]
        return FallbackChatModel(candidates=bound)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> "ModelSpec":
        provider, separator, model = value.partition(":")
        if not separator or not provider or not model:
            raise LLMConfigurationError(
                f"Invalid model spec '{value}'. Expected provider:model."
            )
        return cls(provider=provider.lower(), model=model)


def _temperature_for_role(role: str, settings: Settings) -> float:
    return 0.0 if role in DEBATE_ROLES else settings.llm_temperature


def _build_model(
    spec: ModelSpec,
    settings: Settings,
    *,
    role: str,
) -> BaseChatModel:
    common: dict[str, Any] = {
        "model": spec.model,
        "temperature": _temperature_for_role(role, settings),
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 2,
    }
    if spec.provider == "groq":
        return ChatGroq(api_key=os.environ["GROQ_API_KEY"], **common)
    if spec.provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise KeyError("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(google_api_key=key, **common)
    if spec.provider == "openrouter":
        return ChatOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/",
                "X-Title": "Verdict Room",
            },
            **common,
        )
    if spec.provider == "aiml":
        return ChatOpenAI(
            api_key=os.environ["AIML_API_KEY"],
            base_url="https://api.aimlapi.com/v1",
            **common,
        )
    if spec.provider == "featherless":
        return ChatOpenAI(
            api_key=os.environ["FEATHERLESS_API_KEY"],
            base_url="https://api.featherless.ai/v1",
            **common,
        )
    raise LLMConfigurationError(f"Unsupported provider '{spec.provider}'")


def available_model_specs(
    role: str, settings: Settings | None = None
) -> list[ModelSpec]:
    settings = settings or load_settings()
    configured = configured_provider_keys()
    values = [settings.model_specs.get(role, ""), *settings.fallback_models]
    result: list[ModelSpec] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        spec = ModelSpec.parse(value)
        if role in {"researcher", "scout"} and spec.provider == "featherless":
            continue
        if configured.get(spec.provider, False):
            result.append(spec)
    return result


def make_llm(
    role: str,
    settings: Settings | None = None,
    *,
    temperature_role: str | None = None,
) -> BaseChatModel:
    settings = settings or load_settings()
    specs = available_model_specs(role, settings)
    if not specs:
        requested = settings.model_specs.get(role) or "<unset>"
        raise LLMConfigurationError(
            f"No provider key is configured for role '{role}' ({requested})."
        )
    models = [
        _build_model(spec, settings, role=temperature_role or role) for spec in specs
    ]
    return models[0] if len(models) == 1 else FallbackChatModel(candidates=models)
