from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Literal, TypeVar

T = TypeVar("T")


FailureKind = Literal["http", "timeout", "connection", "transport", "protocol"]


RETRYABLE_4XX: frozenset[int] = frozenset({408, 429})


_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:

    if value and len(value) >= 8:
        _SECRETS.add(value)


def scrub(text: str) -> str:

    for secret in _SECRETS:
        text = text.replace(secret, "<redacted>")
    return text


class LLMError(Exception):
    pass


class ProviderConfigError(LLMError):
    pass


class ProviderError(LLMError):


    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        kind: FailureKind = "http",
        body: str = "",
        provider: str = "",
        model: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.status = status
        self.kind: FailureKind = kind
        self.body = scrub(body)
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        super().__init__(scrub(message))

    def __str__(self) -> str:
        head = super().__str__()
        where = f"{self.provider}/{self.model}" if self.provider else self.model
        status = f" HTTP {self.status}" if self.status is not None else ""
        tail = f" | body: {self.body[:500]}" if self.body else ""
        return f"[{self.kind}{status}] {where}: {head}{tail}"


def is_retryable(exc: BaseException) -> bool:

    if isinstance(exc, ProviderConfigError):
        return False
    if isinstance(exc, ProviderError):
        if exc.kind in ("timeout", "connection"):
            return True
        if exc.kind != "http" or exc.status is None:
            return False
        if exc.status in RETRYABLE_4XX:
            return True
        return exc.status >= 500


    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    return False


@dataclass(frozen=True, slots=True)
class BudgetPolicy:


    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    jitter: float = 0.5
    retry_after_cap: float = 60.0
    concurrency: int = 8

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ProviderConfigError("max_attempts must be >= 1")
        if self.concurrency < 1:
            raise ProviderConfigError("concurrency must be >= 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ProviderConfigError("jitter must be a fraction in [0, 1]")


@dataclass(slots=True)
class Counters:


    calls: int = 0
    ok: int = 0
    failed: int = 0
    attempts: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    rate_limited: int = 0
    timeouts: int = 0
    connection_errors: int = 0
    status_4xx: Counter[int] = field(default_factory=Counter)
    status_5xx: Counter[int] = field(default_factory=Counter)
    sleep_seconds: float = 0.0
    gate_wait_seconds: float = 0.0
    busy_seconds: float = 0.0
    wall_seconds: float = 0.0
    max_in_flight: int = 0

    @property
    def rate_limit_rate(self) -> float:

        return self.rate_limited / self.attempts if self.attempts else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return 1000.0 * self.busy_seconds / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "ok": self.ok,
            "failed": self.failed,
            "attempts": self.attempts,
            "retries": self.retries,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "rate_limited": self.rate_limited,
            "rate_limit_rate": round(self.rate_limit_rate, 4),
            "timeouts": self.timeouts,
            "connection_errors": self.connection_errors,
            "status_4xx": {str(k): v for k, v in sorted(self.status_4xx.items())},
            "status_5xx": {str(k): v for k, v in sorted(self.status_5xx.items())},
            "sleep_seconds": round(self.sleep_seconds, 3),
            "gate_wait_seconds": round(self.gate_wait_seconds, 3),
            "busy_seconds": round(self.busy_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "max_in_flight": self.max_in_flight,
        }

    def merge(self, other: "Counters") -> None:

        self.calls += other.calls
        self.ok += other.ok
        self.failed += other.failed
        self.attempts += other.attempts
        self.retries += other.retries
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.rate_limited += other.rate_limited
        self.timeouts += other.timeouts
        self.connection_errors += other.connection_errors
        self.status_4xx.update(other.status_4xx)
        self.status_5xx.update(other.status_5xx)
        self.sleep_seconds += other.sleep_seconds
        self.gate_wait_seconds += other.gate_wait_seconds
        self.busy_seconds += other.busy_seconds
        self.wall_seconds = max(self.wall_seconds, other.wall_seconds)
        self.max_in_flight = max(self.max_in_flight, other.max_in_flight)


Sleeper = Callable[[float], Awaitable[None]]
TokenExtractor = Callable[[T], tuple[int, int]]


class Budget:


    def __init__(
        self,
        policy: BudgetPolicy | None = None,
        *,
        name: str = "llm",
        sleep: Sleeper | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.policy = policy or BudgetPolicy()
        self.name = name
        self.counters = Counters()
        self._sleep: Sleeper = sleep or asyncio.sleep
        self._rng = rng or random.Random(20260730)
        self._clock = clock
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self._in_flight = 0
        self._t_first: float | None = None


    def _gate(self) -> asyncio.Semaphore:

        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.policy.concurrency)
            self._sem_loop = loop
        return self._sem

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:

        if retry_after is not None and retry_after >= 0.0:
            return min(retry_after, self.policy.retry_after_cap)
        ceiling = min(self.policy.base_delay * (2.0 ** (attempt - 1)), self.policy.max_delay)
        jitter = self.policy.jitter
        return ceiling * (1.0 - jitter) + self._rng.random() * ceiling * jitter

    def _record_failure(self, exc: BaseException) -> None:
        if not isinstance(exc, ProviderError):
            return
        if exc.kind == "timeout":
            self.counters.timeouts += 1
        elif exc.kind == "connection":
            self.counters.connection_errors += 1
        status = exc.status
        if status is None:
            return
        if status == 429:
            self.counters.rate_limited += 1
        if 400 <= status < 500:
            self.counters.status_4xx[status] += 1
        elif status >= 500:
            self.counters.status_5xx[status] += 1


    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        tokens: TokenExtractor | None = None,
    ) -> T:

        sem = self._gate()
        queued = self._clock()
        async with sem:
            self.counters.gate_wait_seconds += self._clock() - queued
            self._in_flight += 1
            self.counters.max_in_flight = max(self.counters.max_in_flight, self._in_flight)
            started = self._clock()
            if self._t_first is None:
                self._t_first = started
            self.counters.calls += 1
            try:
                return await self._attempt_loop(fn, tokens)
            finally:
                self._in_flight -= 1
                finished = self._clock()
                self.counters.busy_seconds += finished - started
                self.counters.wall_seconds = finished - (self._t_first or finished)

    async def _attempt_loop(
        self, fn: Callable[[], Awaitable[T]], tokens: TokenExtractor | None
    ) -> T:
        last: BaseException | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            self.counters.attempts += 1
            try:
                result = await fn()
            except asyncio.CancelledError:

                raise
            except Exception as exc:
                last = exc
                self._record_failure(exc)
                if attempt >= self.policy.max_attempts or not is_retryable(exc):
                    self.counters.failed += 1
                    raise
                self.counters.retries += 1
                retry_after = getattr(exc, "retry_after", None)
                delay = self.backoff(attempt, retry_after)
                self.counters.sleep_seconds += delay
                await self._sleep(delay)
            else:
                self.counters.ok += 1
                if tokens is not None:
                    tin, tout = tokens(result)
                    self.counters.tokens_in += tin
                    self.counters.tokens_out += tout
                return result


        self.counters.failed += 1
        raise last or ProviderError("retry loop exhausted without an error", kind="transport")


    async def map(
        self,
        fns: Iterable[Callable[[], Awaitable[T]]],
        *,
        tokens: TokenExtractor | None = None,
        return_exceptions: bool = False,
    ) -> list[T | BaseException]:

        return await asyncio.gather(
            *(self.run(fn, tokens=tokens) for fn in fns),
            return_exceptions=return_exceptions,
        )

    def render(self) -> str:

        c = self.counters
        lines = [
            f"budget[{self.name}]  concurrency={self.policy.concurrency} "
            f"max_attempts={self.policy.max_attempts}",
            f"  calls        {c.calls}  ok={c.ok} failed={c.failed}",
            f"  attempts     {c.attempts}  retries={c.retries}",
            f"  tokens       in={c.tokens_in} out={c.tokens_out}",
            f"  429          {c.rate_limited}  ({100.0 * c.rate_limit_rate:.1f}% of attempts)",
            f"  timeouts     {c.timeouts}  connection_errors={c.connection_errors}",
            f"  wall         {c.wall_seconds:.2f}s  busy={c.busy_seconds:.2f}s "
            f"backoff={c.sleep_seconds:.2f}s gate_wait={c.gate_wait_seconds:.2f}s",
            f"  latency      mean {c.mean_latency_ms:.0f} ms  max_in_flight={c.max_in_flight}",
        ]
        if c.status_4xx:
            lines.append(f"  4xx          {dict(sorted(c.status_4xx.items()))}")
        if c.status_5xx:
            lines.append(f"  5xx          {dict(sorted(c.status_5xx.items()))}")
        return "\n".join(lines)
