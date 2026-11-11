from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from .budget import ProviderConfigError, ProviderError
from .provider import (
    Completion,
    Message,
    Provider,
    ProviderSpec,
    Usage,
    api_key_for,
    chat_request_kwargs,
)


class OpenAICompatProvider(Provider):


    def __init__(self, spec: ProviderSpec, budget=None) -> None:
        super().__init__(spec, budget)
        self._api_key = api_key_for(spec)
        self._client: Any | None = None


    def client_kwargs(self) -> dict[str, Any]:

        return {
            "api_key": self._api_key,
            "base_url": self.spec.base_url,
            "timeout": self.spec.timeout,

            "max_retries": 0,
            "default_headers": {"Accept-Encoding": "identity"},
        }

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ProviderConfigError(
                    "the openai package is required for kind="
                    f"{self.spec.kind!r}: pip install 'openai>=1.30'"
                ) from exc
            self._client = AsyncOpenAI(**self.client_kwargs())
        return self._client


    async def _complete_once(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        extra: Mapping[str, Any],
    ) -> Completion:
        import openai

        client = self._get_client()
        kwargs = chat_request_kwargs(
            model=self.spec.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **dict(extra),
        )
        if self.spec.extra_body:
            kwargs.setdefault("extra_body", dict(self.spec.extra_body))

        t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise self._error(exc, kind="timeout") from exc
        except openai.APIConnectionError as exc:

            raise self._error(exc, kind="connection") from exc
        except openai.APIStatusError as exc:
            raise self._error(exc, kind="http", status=exc.status_code) from exc
        except openai.OpenAIError as exc:
            raise self._error(exc, kind="transport") from exc
        latency_ms = 1000.0 * (time.perf_counter() - t0)

        choices = getattr(response, "choices", None) or []
        if not choices:


            raise ProviderError(
                soft_error(response) or "response contained no choices",
                kind="protocol",
                body=str(response)[:1000],
                provider=self.spec.kind,
                model=self.spec.model,
            )
        choice = choices[0]
        return Completion(
            text=(getattr(choice.message, "content", None) or ""),
            model=getattr(response, "model", self.spec.model),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=_usage(getattr(response, "usage", None)),
            latency_ms=latency_ms,
        )


    def _error(
        self, exc: Exception, *, kind: str, status: int | None = None
    ) -> ProviderError:

        body = ""
        retry_after: float | None = None
        response = getattr(exc, "response", None)
        if response is not None:
            body = getattr(response, "text", "") or ""
            header = getattr(response, "headers", None)
            if header is not None:
                raw = header.get("retry-after")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        retry_after = None
        return ProviderError(
            str(exc),
            status=status,
            kind=kind,
            body=body,
            provider=self.spec.kind,
            model=self.spec.model,
            retry_after=retry_after,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def soft_error(response: Any) -> str | None:

    extra = dict(getattr(response, "model_extra", None) or {})
    code = extra.get("Code", extra.get("code"))
    message = extra.get("Error", extra.get("error", extra.get("Message")))
    if code is None and message is None:
        return None
    if isinstance(message, dict):
        message = message.get("message") or message.get("Message")
    return f"gateway soft error Code={code}: {message}"


def _usage(raw: Any) -> Usage:

    if raw is None:
        return Usage()
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        cached_tokens=cached,
    )
