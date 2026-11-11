from __future__ import annotations

import abc
import functools
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .budget import (
    Budget,
    BudgetPolicy,
    LLMError,
    ProviderConfigError,
    ProviderError,
    register_secret,
)

__all__ = [
    "Budget",
    "BudgetPolicy",
    "CONFIG_PATH",
    "Completion",
    "LLMError",
    "Message",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderSpec",
    "Usage",
    "api_key_for",
    "auth_headers",
    "build",
    "chat_request_kwargs",
    "config",
    "load_credentials",
    "role_models",
    "roles",
    "spec_for",
    "uses_max_completion_tokens",
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "providers.yaml"

Message = Mapping[str, str]


_KINDS: dict[str, tuple[str, str]] = {
    "openai_compat": (".openai_compat", "OpenAICompatProvider"),
}


@dataclass(frozen=True, slots=True)
class Usage:


    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass(frozen=True, slots=True)
class Completion:


    text: str
    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def as_openai(self) -> SimpleNamespace:

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.text, role="assistant"),
                    finish_reason=self.finish_reason,
                    index=0,
                )
            ],
            model=self.model,
            usage=SimpleNamespace(
                prompt_tokens=self.usage.prompt_tokens,
                completion_tokens=self.usage.completion_tokens,
                total_tokens=self.usage.total_tokens,
            ),
        )


_MAX_COMPLETION_TOKENS_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def uses_max_completion_tokens(model: str) -> bool:

    return (model or "").strip().lower().startswith(_MAX_COMPLETION_TOKENS_PREFIXES)


def chat_request_kwargs(
    *,
    model: str,
    messages: Sequence[Message],
    max_tokens: int | None = None,
    temperature: float | None = None,
    **extra: Any,
) -> dict[str, Any]:

    kwargs: dict[str, Any] = {"model": model, "messages": [dict(m) for m in messages]}
    reasoning = uses_max_completion_tokens(model)
    if max_tokens is not None:
        if reasoning:


            kwargs["max_completion_tokens"] = max(max_tokens * 4, 2000)
        else:
            kwargs["max_tokens"] = max_tokens
    if temperature is not None and not reasoning:
        kwargs["temperature"] = temperature
    kwargs.update(extra)
    return kwargs


def auth_headers(auth_header: str, api_key: str) -> dict[str, str]:

    if auth_header.lower() == "authorization":
        return {"Authorization": f"Bearer {api_key}"}
    return {auth_header: api_key}


@dataclass(frozen=True, slots=True)
class ProviderSpec:


    kind: str
    model: str
    base_url: str
    role: str = "chat"
    api_key_env: str = "LLM_API_KEY"
    auth_header: str = "Authorization"
    api_version: str | None = None
    temperature: float | None = 0.0
    max_tokens: int | None = 1000
    timeout: float = 300.0
    extra_body: Mapping[str, Any] | None = None
    policy: BudgetPolicy = field(default_factory=BudgetPolicy)

    @property
    def label(self) -> str:
        return f"{self.role}:{self.kind}:{self.model}"

    def to_dict(self) -> dict[str, Any]:

        return {
            "role": self.role,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "auth_header": self.auth_header,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "concurrency": self.policy.concurrency,
            "max_attempts": self.policy.max_attempts,
        }


@functools.lru_cache(maxsize=1)
def config() -> dict[str, Any]:

    if not CONFIG_PATH.is_file():
        raise ProviderConfigError(f"provider config not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ProviderConfigError(f"{CONFIG_PATH} did not parse to a mapping")
    return loaded


def roles() -> list[str]:
    return sorted((config().get("roles") or {}).keys())


def role_models() -> dict[str, str]:

    return {role: spec_for(role).model for role in roles()}


_POLICY_KEYS = {f.name for f in fields(BudgetPolicy)}
_SPEC_KEYS = {
    "kind", "model", "base_url", "api_key_env", "auth_header", "api_version",
    "temperature", "max_tokens", "timeout", "extra_body",
}


def spec_for(role: str = "chat", **overrides: Any) -> ProviderSpec:

    cfg = config()
    role_cfg = (cfg.get("roles") or {}).get(role)
    if role_cfg is None:
        raise ProviderConfigError(
            f"unknown role {role!r}; config/providers.yaml defines {roles()}"
        )
    if not isinstance(role_cfg, dict):
        raise ProviderConfigError(f"roles.{role} must be a mapping")

    profile_name = role_cfg.get("profile")
    profiles = cfg.get("profiles") or {}
    if profile_name is not None and profile_name not in profiles:
        raise ProviderConfigError(
            f"roles.{role}.profile={profile_name!r} is not in profiles: "
            f"{sorted(profiles)}"
        )
    layers: list[Mapping[str, Any]] = [
        cfg.get("defaults") or {},
        profiles.get(profile_name) or {},
        {k: v for k, v in role_cfg.items() if k != "profile"},
        overrides,
    ]

    merged: dict[str, Any] = {}
    policy_kw: dict[str, Any] = {}
    for layer in layers:
        nested = layer.get("budget") or {}
        if not isinstance(nested, dict):
            raise ProviderConfigError(f"budget block for role {role!r} must be a mapping")
        policy_kw.update(nested)
        for key, value in layer.items():
            if key == "budget":
                continue
            if key in _POLICY_KEYS:
                policy_kw[key] = value
            elif key in _SPEC_KEYS:
                merged[key] = value
            else:
                raise ProviderConfigError(
                    f"unknown provider setting {key!r} for role {role!r}"
                )

    missing = [k for k in ("kind", "model", "base_url") if not merged.get(k)]
    if missing:
        raise ProviderConfigError(f"role {role!r} is missing required key(s): {missing}")
    if merged["kind"] not in _KINDS:
        raise ProviderConfigError(
            f"unknown provider kind {merged['kind']!r}; known: {sorted(_KINDS)}"
        )
    return ProviderSpec(role=role, policy=BudgetPolicy(**policy_kw), **merged)


@functools.lru_cache(maxsize=1)
def load_credentials() -> str | None:

    dotenv_cfg = config().get("credentials") or {}
    if not dotenv_cfg.get("dotenv", True):
        return None
    from dotenv import load_dotenv
    path = os.environ.get("EVOFORESTMEM_ENV_FILE")
    if path:
        load_dotenv(path, override=False)
    return path


def api_key_for(spec: ProviderSpec) -> str:

    load_credentials()
    key = os.environ.get(spec.api_key_env, "").strip()
    if not key:
        raise ProviderConfigError(
            f"no API key: ${spec.api_key_env} is unset. Export it, or put it in "
            f"EvoForestMem/.env -- copy .env.example and fill it in; it is loaded "
            f"automatically."
        )
    register_secret(key)
    return key


class Provider(abc.ABC):


    def __init__(self, spec: ProviderSpec, budget: Budget | None = None) -> None:
        self.spec = spec
        self.budget = budget or Budget(spec.policy, name=spec.role)


    @abc.abstractmethod
    async def _complete_once(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        extra: Mapping[str, Any],
    ) -> Completion:
        pass

    async def aclose(self) -> None:
        pass


    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> Completion:
        temp = self.spec.temperature if temperature is None else temperature
        mtok = self.spec.max_tokens if max_tokens is None else max_tokens
        return await self.budget.run(
            lambda: self._complete_once(
                messages, temperature=temp, max_tokens=mtok, extra=extra
            ),
            tokens=lambda c: (c.usage.prompt_tokens, c.usage.completion_tokens),
        )

    async def ask(
        self, prompt: str, *, system: str | None = None, **kw: Any
    ) -> Completion:

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.complete(messages, **kw)

    async def complete_many(
        self,
        batch: Iterable[Sequence[Message]],
        *,
        return_exceptions: bool = False,
        **kw: Any,
    ) -> list[Completion | BaseException]:

        import asyncio

        return await asyncio.gather(
            *(self.complete(m, **kw) for m in batch),
            return_exceptions=return_exceptions,
        )

    async def __aenter__(self) -> "Provider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.spec.label}>"


def build(
    role: str = "chat",
    *,
    spec: ProviderSpec | None = None,
    budget: Budget | None = None,
    **overrides: Any,
) -> Provider:

    import importlib

    resolved = spec or spec_for(role, **overrides)
    if spec is not None and overrides:
        resolved = replace(resolved, **overrides)
    module_name, class_name = _KINDS[resolved.kind]
    module = importlib.import_module(module_name, package=__package__)
    cls = getattr(module, class_name)
    return cls(resolved, budget)
