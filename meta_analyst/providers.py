"""Multi-provider LLM clients (OpenAI, Anthropic/Claude, xAI/Grok, Gemini, custom OpenAI-compatible)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from meta_analyst.config import (
    ANTHROPIC_API_KEY,
    CHAT_MODEL,
    GEMINI_API_KEY,
    GROK_API_KEY,
    OPENAI_API_KEY,
)

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GROK = "grok"
PROVIDER_GEMINI = "gemini"
PROVIDER_CUSTOM = "custom"

PROVIDER_OPTIONS = [
    (PROVIDER_OPENAI, "OpenAI (ChatGPT)"),
    (PROVIDER_ANTHROPIC, "Anthropic (Claude)"),
    (PROVIDER_GROK, "xAI (Grok)"),
    (PROVIDER_GEMINI, "Google (Gemini)"),
    (PROVIDER_CUSTOM, "Custom OpenAI-compatible"),
]

PROVIDER_MODELS: dict[str, list[str]] = {
    PROVIDER_OPENAI: [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
        "o4-mini",
        "o3-mini",
    ],
    PROVIDER_ANTHROPIC: [
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    PROVIDER_GROK: [
        "grok-2-latest",
        "grok-2",
        "grok-3",
        "grok-3-mini",
    ],
    PROVIDER_GEMINI: [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    PROVIDER_CUSTOM: [
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "llama-3.3-70b",
        "custom",
    ],
}

DEFAULT_MODELS = {k: v[0] for k, v in PROVIDER_MODELS.items()}

# Rough USD per 1M tokens (input, output) for session cost estimates
_PRICE_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-5": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "grok-2-latest": (2.00, 10.00),
    "grok-2": (2.00, 10.00),
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


@dataclass
class CompletionResult:
    text: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def estimated_cost_usd(self) -> float:
        prices = _PRICE_PER_M.get(self.model)
        if not prices:
            return 0.0
        inp, out = prices
        return (self.prompt_tokens / 1_000_000) * inp + (self.completion_tokens / 1_000_000) * out


@dataclass
class UsageMeter:
    """Session-scoped metered usage across providers/models."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent: list[dict[str, Any]] = field(default_factory=list)

    def record(self, result: CompletionResult, *, feature: str = "") -> None:
        cost = result.estimated_cost_usd()
        self.requests += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.total_tokens += result.total_tokens or (
            result.prompt_tokens + result.completion_tokens
        )
        self.estimated_cost_usd += cost

        bucket = self.by_provider.setdefault(
            result.provider or "unknown",
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "models": {},
            },
        )
        bucket["requests"] += 1
        bucket["prompt_tokens"] += result.prompt_tokens
        bucket["completion_tokens"] += result.completion_tokens
        bucket["total_tokens"] += result.total_tokens or (
            result.prompt_tokens + result.completion_tokens
        )
        bucket["estimated_cost_usd"] += cost
        model_key = result.model or "unknown"
        models = bucket["models"]
        models[model_key] = models.get(model_key, 0) + (
            result.total_tokens or result.prompt_tokens + result.completion_tokens
        )

        self.recent.insert(
            0,
            {
                "feature": feature,
                "provider": result.provider,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens
                or (result.prompt_tokens + result.completion_tokens),
                "estimated_cost_usd": round(cost, 6),
            },
        )
        self.recent = self.recent[:20]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "by_provider": {
                k: {
                    **v,
                    "estimated_cost_usd": round(v["estimated_cost_usd"], 6),
                }
                for k, v in self.by_provider.items()
            },
            "recent": self.recent,
        }


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, model: str | None = None) -> CompletionResult: ...


@dataclass
class ProviderConfig:
    provider: str = PROVIDER_OPENAI
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    label: str = ""

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, CHAT_MODEL)


@dataclass
class ProviderStore:
    active: str = PROVIDER_OPENAI
    configs: dict[str, ProviderConfig] = field(default_factory=dict)

    def upsert(self, config: ProviderConfig) -> None:
        self.configs[config.provider] = config
        self.active = config.provider

    def remove(self, provider: str) -> bool:
        if provider not in self.configs:
            return False
        del self.configs[provider]
        if self.active == provider:
            self.active = next(iter(self.configs), "")
        return True

    def get_active(self) -> ProviderConfig | None:
        return self.configs.get(self.active)

    def has_active_key(self) -> bool:
        cfg = self.get_active()
        return bool(cfg and cfg.api_key.strip())


def seed_store_from_env() -> ProviderStore:
    store = ProviderStore()
    if OPENAI_API_KEY:
        store.upsert(
            ProviderConfig(
                provider=PROVIDER_OPENAI,
                api_key=OPENAI_API_KEY,
                model=DEFAULT_MODELS[PROVIDER_OPENAI],
                label="OpenAI (.env)",
            )
        )
    if ANTHROPIC_API_KEY:
        store.upsert(
            ProviderConfig(
                provider=PROVIDER_ANTHROPIC,
                api_key=ANTHROPIC_API_KEY,
                model=DEFAULT_MODELS[PROVIDER_ANTHROPIC],
                label="Anthropic (.env)",
            )
        )
    if GROK_API_KEY:
        store.upsert(
            ProviderConfig(
                provider=PROVIDER_GROK,
                api_key=GROK_API_KEY,
                model=DEFAULT_MODELS[PROVIDER_GROK],
                label="Grok (.env)",
            )
        )
    if GEMINI_API_KEY:
        store.upsert(
            ProviderConfig(
                provider=PROVIDER_GEMINI,
                api_key=GEMINI_API_KEY,
                model=DEFAULT_MODELS[PROVIDER_GEMINI],
                label="Gemini (.env)",
            )
        )
    if OPENAI_API_KEY:
        store.active = PROVIDER_OPENAI
    elif store.configs:
        store.active = next(iter(store.configs))
    return store


def _estimate_tokens(text: str) -> int:
    # Rough heuristic when provider omits usage
    return max(1, len(text) // 4)


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        default_model: str,
        provider: str = PROVIDER_OPENAI,
    ):
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.default_model = default_model
        self.provider = provider

    def complete(self, *, system: str, user: str, model: str | None = None) -> CompletionResult:
        model_name = model or self.default_model
        try:
            response = self._client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (getattr(response, "output_text", None) or "").strip()
            if text:
                usage = getattr(response, "usage", None)
                prompt = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
                completion = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
                total = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                if not total:
                    prompt = prompt or _estimate_tokens(system + user)
                    completion = completion or _estimate_tokens(text)
                    total = prompt + completion
                return CompletionResult(
                    text=text,
                    model=model_name,
                    provider=self.provider,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                )
        except Exception:
            pass

        response = self._client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else _estimate_tokens(system + user)
        completion = (
            int(getattr(usage, "completion_tokens", 0) or 0) if usage else _estimate_tokens(text)
        )
        total = int(getattr(usage, "total_tokens", 0) or 0) if usage else prompt + completion
        return CompletionResult(
            text=text,
            model=model_name,
            provider=self.provider,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )


class AnthropicClient:
    def __init__(self, api_key: str, *, default_model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model
        self.provider = PROVIDER_ANTHROPIC

    def complete(self, *, system: str, user: str, model: str | None = None) -> CompletionResult:
        model_name = model or self.default_model
        response = self._client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        text = "\n".join(parts).strip()
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "input_tokens", 0) or 0) if usage else _estimate_tokens(system + user)
        completion = (
            int(getattr(usage, "output_tokens", 0) or 0) if usage else _estimate_tokens(text)
        )
        return CompletionResult(
            text=text,
            model=model_name,
            provider=self.provider,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


class GeminiClient:
    def __init__(self, api_key: str, *, default_model: str):
        self.api_key = api_key
        self.default_model = default_model
        self.provider = PROVIDER_GEMINI

    def complete(self, *, system: str, user: str, model: str | None = None) -> CompletionResult:
        import requests

        model_name = model or self.default_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
        }
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=120,
        )
        data = response.json()
        if response.status_code >= 400:
            raise RuntimeError(data.get("error", {}).get("message") or response.text)
        candidates = data.get("candidates") or []
        if not candidates:
            return CompletionResult(text="", model=model_name, provider=self.provider)
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        meta = data.get("usageMetadata") or {}
        prompt = int(meta.get("promptTokenCount") or 0) or _estimate_tokens(system + user)
        completion = int(meta.get("candidatesTokenCount") or 0) or _estimate_tokens(text)
        total = int(meta.get("totalTokenCount") or 0) or prompt + completion
        return CompletionResult(
            text=text,
            model=model_name,
            provider=self.provider,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )


class TrackingClient:
    """Wraps an LLM client and records metered usage."""

    def __init__(self, client: LLMClient, meter: UsageMeter, *, feature: str = ""):
        self._client = client
        self._meter = meter
        self.feature = feature

    def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        result = self._client.complete(system=system, user=user, model=model)
        self._meter.record(result, feature=self.feature)
        return result.text


def build_client(config: ProviderConfig) -> LLMClient:
    if not config.api_key.strip():
        raise ValueError(f"No API key configured for {config.provider}.")

    model = config.resolved_model()
    if config.provider == PROVIDER_OPENAI:
        return OpenAICompatibleClient(
            config.api_key, default_model=model, provider=PROVIDER_OPENAI
        )
    if config.provider == PROVIDER_GROK:
        return OpenAICompatibleClient(
            config.api_key,
            base_url="https://api.x.ai/v1",
            default_model=model,
            provider=PROVIDER_GROK,
        )
    if config.provider == PROVIDER_CUSTOM:
        if not config.base_url.strip():
            raise ValueError("Custom provider requires a base URL.")
        return OpenAICompatibleClient(
            config.api_key,
            base_url=config.base_url.strip(),
            default_model=model,
            provider=PROVIDER_CUSTOM,
        )
    if config.provider == PROVIDER_ANTHROPIC:
        return AnthropicClient(config.api_key, default_model=model)
    if config.provider == PROVIDER_GEMINI:
        return GeminiClient(config.api_key, default_model=model)
    raise ValueError(f"Unsupported provider: {config.provider}")


def get_session_client(
    store: ProviderStore,
    meter: UsageMeter | None = None,
    *,
    feature: str = "",
) -> Any:
    cfg = store.get_active()
    if not cfg:
        raise ValueError(
            "No AI provider configured. Open AI Providers and add an API key "
            "(OpenAI, Claude, Grok, Gemini, or custom)."
        )
    client = build_client(cfg)
    if meter is None:
        # Back-compat: return object whose complete() yields str
        class _Plain:
            def complete(self, *, system: str, user: str, model: str | None = None) -> str:
                return client.complete(system=system, user=user, model=model).text

        return _Plain()
    return TrackingClient(client, meter, feature=feature)
