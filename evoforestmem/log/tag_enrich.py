from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..llm.provider import Completion, Provider, build as build_provider
from ..prompts.tag_enrich import PROMPT_VERSION, build_batch_messages, build_messages
from . import tags as rule_tags
from .message import Message, Tags
from .store import MessageLog
from .tags_vocab import LIFECYCLE_LABELS, MARKER_LABELS

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class TagEnrichConfig:
    mode: str = "eager"
    role: str = "tag_enrich"
    max_inflight: int | None = None
    prompt_version: str = PROMPT_VERSION
    context_window: int = 3
    request_batch: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TagEnrichConfig":
        raw = raw or {}
        mode = str(raw.get("mode", "eager"))
        if mode not in {"off", "eager"}:
            raise ValueError("tag_enrich.mode must be 'off' or 'eager'")
        max_inflight = raw.get("max_inflight")
        if max_inflight is not None and int(max_inflight) < 1:
            raise ValueError("tag_enrich.max_inflight must be positive or null")
        context_window = int(raw.get("context_window", 3))
        if context_window < 0:
            raise ValueError("tag_enrich.context_window must be non-negative")
        request_batch = int(raw.get("request_batch", 1))
        if request_batch < 1 or request_batch > 8:
            raise ValueError("tag_enrich.request_batch must be in [1, 8]")
        return cls(
            mode=mode,
            role=str(raw.get("role", "tag_enrich")),
            max_inflight=(int(max_inflight) if max_inflight is not None else None),
            prompt_version=str(raw.get("prompt_version", PROMPT_VERSION)),
            context_window=context_window,
            request_batch=request_batch,
        )


@dataclass(slots=True)
class TagEnrichReport:
    messages: int = 0
    calls: int = 0
    cached: int = 0
    enriched: int = 0
    fallback: int = 0
    skipped: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    batch_requests: int = 0
    salvage_requests: int = 0
    elapsed_sec: float = 0.0
    provider: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: int(getattr(self, name)) for name in (
            "messages", "calls", "cached", "enriched", "fallback", "skipped",
            "tokens_in", "tokens_out", "batch_requests", "salvage_requests",
        )}
        out["elapsed_sec"] = round(self.elapsed_sec, 3)
        if self.provider:
            out["provider"] = dict(self.provider)
        return out


def _dedupe(values: Sequence[str], *, limit: int = 12) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = " ".join(str(value).strip().split())[:160]
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
        if len(out) >= limit:
            break
    return tuple(out)


def _cache_key(message: Message, prompt_version: str) -> str:
    text_hash = hashlib.sha1(message.text.encode("utf-8")).hexdigest()
    raw = f"{message.msg_id}\0{text_hash}\0{prompt_version}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_trivial(text: str) -> bool:
    compact = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return len(compact) < 2


def parse_enrichment(text: str, *, baseline: Tags) -> Tags:

    hit = _JSON_OBJECT.search(text.strip())
    if hit is None:
        raise ValueError("tag enrichment response is not a JSON object")
    raw = json.loads(hit.group(0))
    if not isinstance(raw, dict):
        raise ValueError("tag enrichment response must be an object")

    return _tags_from_mapping(raw, baseline=baseline)


def _tags_from_mapping(raw: Mapping[str, Any], *, baseline: Tags) -> Tags:

    deliverables = raw.get("deliverables", [])
    markers = raw.get("markers", [])
    lifecycle = raw.get("lifecycle")
    confidence = raw.get("lifecycle_confidence")
    if not isinstance(deliverables, list) or not all(isinstance(v, str) for v in deliverables):
        raise ValueError("deliverables must be an array of strings")
    if not isinstance(markers, list) or not all(isinstance(v, str) for v in markers):
        raise ValueError("markers must be an array of strings")
    if not set(markers) <= MARKER_LABELS:
        raise ValueError("response contains an unknown marker label")
    if lifecycle is not None and lifecycle not in LIFECYCLE_LABELS:
        raise ValueError("response contains an unknown lifecycle label")
    if confidence is not None:
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("lifecycle_confidence must be in [0, 1]")

    return Tags(
        mentions=baseline.mentions,
        artifacts=_dedupe((*baseline.artifacts, *deliverables)),
        markers=tuple(markers),
        lifecycle=lifecycle,
        deliverables=_dedupe(deliverables),
        tag_source="llm+rule" if baseline.artifacts else "llm",
        lifecycle_confidence=confidence,
    )


def parse_enrichment_batch(
    text: str, *, messages: Sequence[Message], baselines: Mapping[str, Tags]
) -> dict[str, Tags]:

    hit = _JSON_OBJECT.search(text.strip())
    if hit is None:
        raise ValueError("tag enrichment batch response is not a JSON object")
    raw = json.loads(hit.group(0))
    results = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(results, list):
        raise ValueError("batch response results must be an array")
    expected = [message.msg_id for message in messages]
    parsed: dict[str, Tags] = {}
    for row in results:
        if not isinstance(row, dict) or not isinstance(row.get("msg_id"), str):
            raise ValueError("each batch result must carry msg_id")
        msg_id = row["msg_id"]
        if msg_id not in baselines or msg_id in parsed:
            raise ValueError(f"unexpected or duplicate batch msg_id {msg_id!r}")
        parsed[msg_id] = _tags_from_mapping(row, baseline=baselines[msg_id])
    if set(parsed) != set(expected):
        raise ValueError("batch response does not cover every input msg_id")
    return parsed


def _preceding_windows(messages: Sequence[Message], size: int) -> dict[str, tuple[Message, ...]]:
    windows: dict[str, tuple[Message, ...]] = {}
    prior: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=size or 1))


    for message in sorted(messages, key=lambda m: (m.ts, m.seq)):
        windows[message.msg_id] = tuple(prior[message.speaker_key]) if size else ()
        if size:
            prior[message.speaker_key].append(message)
    return windows


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("key"), str):
                    out[row["key"]] = row
            except (json.JSONDecodeError, OSError):
                continue
    return out


def _append_cache(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _baseline(message: Message) -> Tags:
    derived = rule_tags.derive(message.text)
    return Tags(
        mentions=message.tags.mentions or derived.mentions,
        artifacts=derived.artifacts,
        markers=derived.markers,
        lifecycle=derived.lifecycle,
    )


async def enrich_log(
    log: MessageLog,
    *,
    config: TagEnrichConfig,
    cache_path: str | Path,
    provider: Provider | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> TagEnrichReport:

    started = time.perf_counter()
    report = TagEnrichReport(messages=len(log))
    cache_path = Path(cache_path)
    cache = _read_cache(cache_path)
    windows = _preceding_windows(log.messages, config.context_window)
    owned_provider = provider is None
    client = provider or build_provider(config.role)

    pending: list[Message] = []
    new_rows: list[dict[str, Any]] = []
    for message in list(log):
        baseline = _baseline(message)
        if _is_trivial(message.text):
            log.replace_tags(message.msg_id, baseline)
            report.skipped += 1
            continue
        key = _cache_key(message, config.prompt_version)
        row = cache.get(key)
        if row is not None:
            try:
                tags = Tags.from_dict(row.get("tags"))
                frozen_fallback = row.get("status") == "fallback"
                if tags.tag_source not in {"llm", "llm+rule"} and not frozen_fallback:
                    raise ValueError("cached row is neither enriched nor a frozen fallback")
                log.replace_tags(message.msg_id, tags)
                report.cached += 1
                if frozen_fallback:
                    report.fallback += 1
                else:
                    report.enriched += 1
                continue
            except (TypeError, ValueError):
                pass
        pending.append(message)

    if progress is not None:
        progress(report.cached + report.skipped, len(log))

    try:


        chunk_size = max(config.request_batch, config.max_inflight or 256)
        processed = report.cached + report.skipped
        for offset in range(0, len(pending), chunk_size):
            chunk_messages = pending[offset: offset + chunk_size]
            groups = [
                chunk_messages[i:i + config.request_batch]
                for i in range(0, len(chunk_messages), config.request_batch)
            ]
            request_prompts = [
                (build_messages(
                    group[0], windows[group[0].msg_id],
                    prompt_version=config.prompt_version,
                ) if len(group) == 1 else build_batch_messages(
                    [(message, windows[message.msg_id]) for message in group],
                    prompt_version=config.prompt_version,
                ))
                for group in groups
            ]
            results = await client.complete_many(request_prompts, return_exceptions=True)
            report.calls += len(request_prompts)
            report.batch_requests += sum(len(group) > 1 for group in groups)
            salvage: list[Message] = []

            def persist(message: Message, tags: Tags, *, fallback: bool = False) -> None:
                log.replace_tags(message.msg_id, tags)
                report.fallback += int(fallback)
                report.enriched += int(not fallback)
                row: dict[str, Any] = {
                    "key": _cache_key(message, config.prompt_version),
                    "msg_id": message.msg_id,
                    "prompt_version": config.prompt_version,
                    "tags": tags.to_dict(),
                }
                if fallback:
                    row["status"] = "fallback"
                new_rows.append(row)

            for group, result in zip(groups, results):
                if isinstance(result, BaseException):
                    if len(group) > 1:
                        salvage.extend(group)
                    else:
                        persist(group[0], _baseline(group[0]), fallback=True)
                    continue
                assert isinstance(result, Completion)
                report.tokens_in += result.usage.prompt_tokens
                report.tokens_out += result.usage.completion_tokens
                baselines = {message.msg_id: _baseline(message) for message in group}
                try:
                    parsed = (
                        {group[0].msg_id: parse_enrichment(
                            result.text, baseline=baselines[group[0].msg_id]
                        )}
                        if len(group) == 1
                        else parse_enrichment_batch(
                            result.text, messages=group, baselines=baselines
                        )
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    if len(group) > 1:
                        salvage.extend(group)
                    else:
                        persist(group[0], baselines[group[0].msg_id], fallback=True)
                    continue
                for message in group:
                    persist(message, parsed[message.msg_id])


            if salvage:
                salvage_prompts = [
                    build_messages(
                        message, windows[message.msg_id],
                        prompt_version=config.prompt_version,
                    )
                    for message in salvage
                ]
                salvage_results = await client.complete_many(
                    salvage_prompts, return_exceptions=True
                )
                report.calls += len(salvage_prompts)
                report.salvage_requests += len(salvage_prompts)
                for message, result in zip(salvage, salvage_results):
                    baseline = _baseline(message)
                    if isinstance(result, BaseException):
                        persist(message, baseline, fallback=True)
                        continue
                    assert isinstance(result, Completion)
                    report.tokens_in += result.usage.prompt_tokens
                    report.tokens_out += result.usage.completion_tokens
                    try:
                        tags = parse_enrichment(result.text, baseline=baseline)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        persist(message, baseline, fallback=True)
                    else:
                        persist(message, tags)

            _append_cache(cache_path, new_rows)
            new_rows.clear()
            processed += len(chunk_messages)
            if progress is not None:
                progress(processed, len(log))
    finally:
        report.elapsed_sec = time.perf_counter() - started
        budget = getattr(client, "budget", None)
        counters = getattr(budget, "counters", None)
        if counters is not None:
            report.provider = counters.to_dict()
        if owned_provider:
            await client.aclose()
    return report
