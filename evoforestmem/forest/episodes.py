from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..log.message import Message
from ..log.store import MessageLog
from ..log.tags_vocab import NONTERMINAL_LABELS, START_LABELS, TERMINAL_END_LABELS
from ..retrieve.s0_analyse import norm_tokens
from ._persist import (
    advance_corpus_digest,
    atomic_write,
    corpus_fingerprint as _corpus_fingerprint,
)


START_DECL = re.compile(
    r"\b(?:I\s*(?:'m|’m| am)|we\s*(?:'re|’re| are)?)\s+"
    r"(?:now\s+|also\s+|officially\s+|formally\s+)*"
    r"(?:start|starting|begin|beginning|kick(?:ing)?[- ]?off|launch|launching|initiating)\b"
    r"|\b(?:today|this\s+morning|this\s+afternoon)\b[^.!?]{0,80}?\b"
    r"(?:officially\s+)?(?:start|started|starting|begin|began|beginning|launch|launched|"
    r"launching|kick(?:ed|ing)?[- ]?off|initiate[sd]?|initiating)\b"
    r"|\b(?:start|starting|begin|beginning|launch|launching)\s+(?:a|the|our|my|this)\s+"
    r"(?:new\s+)?(?:task|design task|documentation task|work|assignment|round)\b"
    r"|\bI['’]ll\s+start\b|\bI\s+will\s+start\b|\bI\s+have\s+started\b|\bI['’]ve\s+started\b",
    re.IGNORECASE,
)


STRONG_END = re.compile(
    r"\[task\s+completed\]"
    r"|\b(?:has|have)\s+been\s+(?:officially\s+|successfully\s+|fully\s+|all\s+)?"
    r"(?:completed|finished|concluded|finalized|finalised)\b"
    r"|\b(?:is|are)\s+(?:now\s+)?(?:officially\s+)?(?:complete|completed|finished)\b"
    r"|\bofficially\s+(?:announce|declare|mark)\w*\b[^.!?]{0,90}"
    r"\b(?:complete|completed|done|finished)\b"
    r"|\bmark\s+(?:the\s+)?task\b[^.!?]{0,40}\bas\s+complete"
    r"|\bhas\s+(?:officially\s+)?concluded\b"
    r"|\bhas\s+been\s+(?:officially\s+)?(?:released|delivered)\b",
    re.IGNORECASE,
)


WEAK_END = re.compile(
    r"\b(?:has|have)\s+been\s+uploaded\s+to\b"
    r"|\bfinal\s+version\s+(?:of\s+)?[^.!?]{0,60}\bhas\s+been\b"
    r"|\bdelivery\s+page\b|\barchiv(?:ed|ing)\b",
    re.IGNORECASE,
)


PREMATURE = re.compile(
    r"\b(?:will|going to|expect to|plan to|aim to|should)\s+(?:be\s+)?"
    r"(?:finish|complete|deliver|submit|wrap up|close)\w*\b"
    r"|\bwill be (?:completed|delivered|finished)\b",
    re.IGNORECASE,
)


_NOISE = frozenset(norm_tokens(
    "today tomorrow yesterday morning afternoon evening hello everyone good all the and "
    "for with this that our your their start starting started begin beginning task new "
    "work working also here update progress please thanks thank team project group "
    "everyone received okay next week day days"
))


_NAMED = re.compile(r"\[([^\]\n]{6,120})\]|[\"“《「]([^\"”》」\n]{6,120})[\"”》」]")


def _content_terms(text: str) -> frozenset[str]:
    return frozenset(w for w in norm_tokens(text) if len(w) > 2 and w not in _NOISE)


EPISODE_SCHEMA_VERSION = "evoforestmem.episode-table.v1"
EPISODE_POLICY_VERSION = "arrival-idf-arrival-candidates-v1"


def episode_corpus_fingerprint(messages: Iterable[Message]) -> str:

    return _corpus_fingerprint(messages)


@dataclass(frozen=True, slots=True)
class Interval:


    start_date: str
    end_date: str
    calendar_days_inclusive: int
    calendar_days_exclusive: int
    working_days: int
    corpus_days: int


    activity_days: int | None = None

    @property
    def valid(self) -> bool:
        return self.calendar_days_exclusive >= 0

    def value(self, quantity: str, convention: str) -> int:
        if quantity == "working":
            return (self.activity_days if self.activity_days is not None
                    else self.working_days)
        if quantity == "corpus":
            return self.corpus_days
        return (self.calendar_days_inclusive if convention == "inclusive"
                else self.calendar_days_exclusive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "calendar_days_inclusive": self.calendar_days_inclusive,
            "calendar_days_exclusive": self.calendar_days_exclusive,
            "working_days": self.working_days,
            "corpus_days": self.corpus_days,
            "activity_days": self.activity_days,
        }


def working_days_between(start: str, end: str,
                         corpus_days: Sequence[str] | None = None) -> int:

    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d1 < d0:
        return 0
    if corpus_days is not None:
        return sum(1 for d in corpus_days if start <= d <= end)
    n, cursor = 0, d0
    while cursor <= d1:
        n += cursor.weekday() < 5
        cursor += timedelta(days=1)
    return n


def interval_between(start: str, end: str,
                     corpus_days: Sequence[str] | None = None, *,
                     activity_days: int | None = None) -> Interval:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    delta = (d1 - d0).days
    return Interval(
        start_date=start,
        end_date=end,
        calendar_days_inclusive=delta + 1,
        calendar_days_exclusive=delta,
        working_days=working_days_between(start, end),
        corpus_days=working_days_between(start, end, corpus_days),
        activity_days=(max(0, int(activity_days)) if activity_days is not None else None),
    )


@dataclass(frozen=True, slots=True)
class Boundary:


    msg_id: str
    kind: str
    strength: str
    date: str
    ts: float
    speaker_key: str
    channel_key: str
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "msg_id": self.msg_id, "kind": self.kind, "strength": self.strength,
            "date": self.date, "ts": float(self.ts), "speaker": self.speaker_key,
            "channel": self.channel_key, "quote": self.quote,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Boundary":
        return cls(
            msg_id=str(row["msg_id"]), kind=str(row["kind"]),
            strength=str(row["strength"]), date=str(row["date"]), ts=float(row["ts"]),
            speaker_key=str(row["speaker"]), channel_key=str(row["channel"]),
            quote=str(row.get("quote") or ""),
        )


@dataclass(frozen=True, slots=True)
class Episode:


    name: str
    aliases: tuple[str, ...]
    actor: str
    channel: str
    start: Boundary
    end: Boundary
    interval: Interval

    pair_score: float = 0.0
    terms: frozenset[str] = frozenset()

    @property
    def key(self) -> str:
        return f"{self.start.msg_id}:{self.end.msg_id}"

    @property
    def msg_ids(self) -> tuple[str, str]:
        return (self.start.msg_id, self.end.msg_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "actor": self.actor,
            "channel": self.channel,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "interval": self.interval.to_dict(),
            "pair_score": round(self.pair_score, 4),
            "terms": sorted(self.terms),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Episode":
        interval = row["interval"]
        return cls(
            name=str(row["name"]),
            aliases=tuple(str(a) for a in row.get("aliases", ())),
            actor=str(row["actor"]),
            channel=str(row["channel"]),
            start=Boundary.from_dict(row["start"]),
            end=Boundary.from_dict(row["end"]),
            interval=Interval(
                start_date=str(interval["start_date"]),
                end_date=str(interval["end_date"]),
                calendar_days_inclusive=int(interval["calendar_days_inclusive"]),
                calendar_days_exclusive=int(interval["calendar_days_exclusive"]),
                working_days=int(interval["working_days"]),
                corpus_days=int(interval["corpus_days"]),
                activity_days=(None if interval.get("activity_days") is None
                               else int(interval["activity_days"])),
            ),
            pair_score=float(row.get("pair_score", 0.0)),
            terms=frozenset(str(t) for t in row.get("terms", ())),
        )


@dataclass(frozen=True, slots=True)
class EpisodeConfig:


    max_span_days: int = 30


    ends_per_start: int = 3

    strong_end_bonus: float = 3.0

    span_penalty: float = 0.05

    artifact_weight: float = 1.0

    deliverable_weight: float = 3.0


DEFAULT_CONFIG = EpisodeConfig()


def is_start(message: Message) -> bool:

    enriched = message.tags.tag_source in {"llm", "llm+rule"}
    return ((enriched and message.tags.lifecycle in START_LABELS)
            or (not enriched and bool(START_DECL.search(message.text))))


def is_end(message: Message) -> bool:

    enriched = message.tags.tag_source in {"llm", "llm+rule"}
    lifecycle = message.tags.lifecycle
    if ((enriched and lifecycle in NONTERMINAL_LABELS)
            or PREMATURE.search(message.text)):
        return False
    return ((enriched and lifecycle in TERMINAL_END_LABELS)
            or (not enriched and bool(STRONG_END.search(message.text)
                                      or WEAK_END.search(message.text))))


def is_strong_end(message: Message) -> bool:

    lifecycle = message.tags.lifecycle
    if ((message.tags.tag_source in {"llm", "llm+rule"}
            and lifecycle in NONTERMINAL_LABELS) or PREMATURE.search(message.text)):
        return False
    return lifecycle in TERMINAL_END_LABELS or bool(STRONG_END.search(message.text))


@dataclass
class EpisodeStats:
    messages: int = 0
    start_declarations: int = 0
    end_declarations: int = 0
    strong_ends: int = 0
    episodes: int = 0
    unpaired_starts: int = 0
    corpus_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "start_declarations": self.start_declarations,
            "end_declarations": self.end_declarations,
            "strong_ends": self.strong_ends,
            "episodes": self.episodes,
            "unpaired_starts": self.unpaired_starts,
            "corpus_days": self.corpus_days,
        }


class EpisodeTable:


    def __init__(self, log: MessageLog | None = None, *,
                 config: EpisodeConfig = DEFAULT_CONFIG,
                 build: bool = True):
        self.log = log
        self.config = config
        self.stats = EpisodeStats(messages=len(log) if log is not None else 0)


        self._df: Counter[str] = Counter()
        self._n_docs: int = 0
        self._idf_cache: dict[str, float] | None = None
        self._seen_msg_ids: set[str] = set()
        self._corpus_digest: bytes = bytes(32)

        self._terms: dict[str, frozenset[str]] = {}
        self._artifact_terms: dict[str, frozenset[str]] = {}
        self._deliverable_terms: dict[str, frozenset[str]] = {}


        self._by_speaker_day: dict[str, dict[str, list[str]]] = {}
        self.corpus_days: tuple[str, ...] = ()
        self.episodes: list[Episode] = []

        self.by_start: dict[str, list[Episode]] = {}


        self._ends_of_start: dict[str, list[str]] = {}

        self._start_seq: dict[str, int] = {}
        self._seq: dict[str, int] = {}
        self._strong: set[str] = set()


        self._pending_ends_by_channel: dict[str, list[str]] = defaultdict(list)


        self._starts_by_channel: dict[str, list[str]] = defaultdict(list)
        self.starts: list[Message] = []
        self.ends: list[Message] = []


        self._candidate_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self._messages: dict[str, Message] = {}
        self._topk_decisions: list[dict[str, Any]] = []


        self._start_boundary_cache: dict[str, Boundary] = {}
        self._start_name_cache: dict[str, tuple[str, tuple[str, ...]]] = {}
        if build and log is not None:
            self._build()


    @property
    def corpus_fingerprint(self) -> str:
        return self._corpus_digest.hex()

    @property
    def idf(self) -> dict[str, float]:
        if self._idf_cache is None:
            n = max(1, self._n_docs)
            self._idf_cache = {
                w: math.log(1.0 + (n - c + 0.5) / (c + 0.5))
                for w, c in self._df.items()
            }
        return self._idf_cache

    def _observe_message(self, m: Message) -> bool:

        if m.msg_id in self._seen_msg_ids:
            return False
        self._seen_msg_ids.add(m.msg_id)
        self._n_docs += 1
        terms = _content_terms(m.text)
        self._terms[m.msg_id] = terms
        self._artifact_terms[m.msg_id] = frozenset(
            w for a in m.tags.artifacts for w in norm_tokens(a)
            if len(w) > 2 and w not in _NOISE
        )
        self._deliverable_terms[m.msg_id] = frozenset(
            w for d in m.tags.deliverables for w in norm_tokens(d)
            if len(w) > 2 and w not in _NOISE
        )
        self._df.update(terms)
        self._idf_cache = None
        date_str = m.date
        if date_str not in self.corpus_days:
            self.corpus_days = tuple(sorted(self.corpus_days + (date_str,)))
        self._by_speaker_day.setdefault(m.speaker_key, {}).setdefault(
            date_str, []).append(m.msg_id)
        self._messages[m.msg_id] = m
        self._seq[m.msg_id] = m.seq
        self._corpus_digest = advance_corpus_digest(self._corpus_digest, m)
        self.stats.messages = self._n_docs
        self.stats.corpus_days = len(self.corpus_days)
        return True


    def _pair_score(self, a: Message, b: Message) -> float | None:

        cfg = self.config
        window = cfg.max_span_days * 86400.0
        if b.ts < a.ts or b.ts - a.ts > window or b.msg_id == a.msg_id:
            return None
        a_terms = self._terms[a.msg_id]
        overlap = self._wov(a_terms, self._terms[b.msg_id])
        deliverable_overlap = self._wov(
            self._deliverable_terms[a.msg_id], self._deliverable_terms[b.msg_id]
        )
        artifact_overlap = self._wov(
            self._artifact_terms[a.msg_id], self._artifact_terms[b.msg_id]
        )
        if overlap <= 0.0 and deliverable_overlap <= 0.0 and artifact_overlap <= 0.0:
            return None
        return (
            overlap
            + cfg.deliverable_weight * deliverable_overlap
            + (cfg.strong_end_bonus if b.msg_id in self._strong else 0.0)
            + cfg.artifact_weight * artifact_overlap
            - cfg.span_penalty * (b.ts - a.ts) / 86400.0
        )

    def _activity_days(self, a: Message, b: Message, episode_terms: frozenset[str]) -> int:

        a_speaker = a.speaker_key
        speaker_days = self._by_speaker_day.get(a_speaker)
        if speaker_days is None:
            return 0
        lo = bisect.bisect_left(self.corpus_days, a.date)
        hi = bisect.bisect_right(self.corpus_days, b.date)
        terms_of = self._terms
        activity = 0
        for di in range(lo, hi):
            for mid in speaker_days.get(self.corpus_days[di], ()):
                if terms_of[mid] & episode_terms:
                    activity += 1
                    break
        return activity

    def _make_episode(self, a: Message, b: Message, score: float) -> Episode:
        name, aliases = self._start_name_cache.get(a.msg_id, (None, None))
        if name is None:
            name, aliases = self._name(a)
            self._start_name_cache[a.msg_id] = (name, aliases)
        start_boundary = self._start_boundary_cache.get(a.msg_id)
        if start_boundary is None:
            start_boundary = self._boundary(a, "start", "declared")
            self._start_boundary_cache[a.msg_id] = start_boundary
        episode_terms = self._terms[a.msg_id] | self._terms[b.msg_id]
        return Episode(
            name=name,
            aliases=aliases,
            actor=a.speaker_key,
            channel=a.channel_key,
            start=start_boundary,
            end=self._boundary(b, "end",
                               "declared" if b.msg_id in self._strong else "reported"),
            interval=interval_between(a.date, b.date, self.corpus_days,
                                      activity_days=self._activity_days(a, b, episode_terms)),
            pair_score=score,
            terms=episode_terms,
        )

    def _select_topk(self, a: Message, scored: list[tuple[float, Message]]) -> list[Message]:

        scored.sort(key=lambda r: (-r[0], self._seq.get(r[1].msg_id, r[1].seq)))
        return [b for _score, b in scored[: self.config.ends_per_start]]

    def _commit_start(self, a: Message, ends: list[Message],
                      scored: list[tuple[float, Message]]) -> None:

        if not ends:
            self.stats.unpaired_starts += 1
            self.by_start[a.msg_id] = []
            self._ends_of_start[a.msg_id] = []
            return
        bucket: list[Episode] = []
        for b in ends:
            score = next(s for s, m in scored if m.msg_id == b.msg_id)
            episode = self._make_episode(a, b, score)
            bucket.append(episode)
            self.episodes.append(episode)
        self.by_start[a.msg_id] = bucket
        self._ends_of_start[a.msg_id] = [b.msg_id for b in ends]


    def _build(self) -> None:
        assert self.log is not None
        for m in self.log:
            self._observe_message(m)

        ends_by_channel: dict[str, list[Message]] = defaultdict(list)
        for m in self.log:
            if is_start(m):
                self.starts.append(m)
            if not is_end(m):
                continue
            self.ends.append(m)
            ends_by_channel[m.channel_key].append(m)
            if is_strong_end(m):
                self._strong.add(m.msg_id)
        self.stats.start_declarations = len(self.starts)
        self.stats.end_declarations = len(self.ends)
        self.stats.strong_ends = len(self._strong)

        for a in self.starts:
            rows: list[tuple[float, Message]] = []
            for b in ends_by_channel.get(a.channel_key, ()):
                score = self._pair_score(a, b)
                if score is None:
                    continue
                rows.append((score, b))
            if not rows:
                self.stats.unpaired_starts += 1
                self.by_start[a.msg_id] = []
                self._ends_of_start[a.msg_id] = []
                continue
            self._commit_start(a, self._select_topk(a, rows), rows)
        self.stats.episodes = len(self.episodes)


    @classmethod
    def incremental(cls, *, config: EpisodeConfig = DEFAULT_CONFIG) -> "EpisodeTable":
        return cls(None, config=config, build=False)

    @classmethod
    def from_stream(
        cls,
        messages: Iterable[Message],
        *,
        config: EpisodeConfig = DEFAULT_CONFIG,
    ) -> "EpisodeTable":

        table = cls.incremental(config=config)
        for message in messages:
            table.append(message)
        return table

    def append(self, message: Message) -> list[Episode]:

        if not self._observe_message(message):
            return []
        m = message
        is_s = is_start(m)
        is_e = is_end(m)
        if is_s:
            self.starts.append(m)
            self._start_seq[m.msg_id] = m.seq
            self._starts_by_channel[m.channel_key].append(m.msg_id)
        if is_e:
            self.ends.append(m)
            if is_strong_end(m):
                self._strong.add(m.msg_id)
            self._pending_ends_by_channel[m.channel_key].append(m.msg_id)
        self.stats.start_declarations = len(self.starts)
        self.stats.end_declarations = len(self.ends)
        self.stats.strong_ends = len(self._strong)

        touched: list[Episode] = []
        if is_s:

            for end_id in self._pending_ends_by_channel.get(m.channel_key, ()):
                b = self._messages.get(end_id)
                if b is None:
                    continue
                score = self._pair_score(m, b)
                if score is not None:
                    self._candidate_scores[m.msg_id][end_id] = score
            touched.extend(self._rebuild_bucket(m.msg_id))
        if is_e:


            for start_id in self._starts_by_channel.get(m.channel_key, ()):
                if start_id == m.msg_id:
                    continue
                a = self._messages.get(start_id)
                if a is None:
                    continue
                score = self._pair_score(a, m)
                if score is None:
                    continue
                self._candidate_scores[start_id][m.msg_id] = score
                old_ends = list(self._ends_of_start.get(start_id, ()))
                touched_eps = self._rebuild_bucket(start_id)
                new_ends = list(self._ends_of_start.get(start_id, ()))
                if old_ends != new_ends:
                    self._topk_decisions.append({
                        "start": start_id, "trigger_end": m.msg_id,
                        "dropped": sorted(set(old_ends) - set(new_ends)),
                        "added": sorted(set(new_ends) - set(old_ends)),
                        "candidates": len(self._candidate_scores[start_id]),
                    })
                touched.extend(touched_eps)


        new_terms = self._terms[m.msg_id]
        for e in self.episodes:
            if e.actor != m.speaker_key:
                continue
            if not (e.start.date <= m.date <= e.end.date):
                continue
            if not (new_terms & e.terms):
                continue
            fresh = self._activity_days(
                self._messages[e.start.msg_id], self._messages[e.end.msg_id], e.terms)
            if fresh != e.interval.activity_days:
                idx = self.episodes.index(e)
                updated = Episode(
                    name=e.name, aliases=e.aliases, actor=e.actor, channel=e.channel,
                    start=e.start, end=e.end,
                    interval=interval_between(e.start.date, e.end.date, self.corpus_days,
                                              activity_days=fresh),
                    pair_score=e.pair_score, terms=e.terms,
                )
                self.episodes[idx] = updated
                bucket = self.by_start.get(e.start.msg_id)
                if bucket:
                    for bi, be in enumerate(bucket):
                        if be.end.msg_id == e.end.msg_id:
                            bucket[bi] = updated
                            break
        self.stats.episodes = len(self.episodes)
        return touched

    def _rebuild_bucket(self, start_id: str) -> list[Episode]:

        a = self._messages.get(start_id)
        scores = self._candidate_scores.get(start_id, {})
        current = list(self._ends_of_start.get(start_id, ()))
        if a is None or not scores:
            if current or start_id not in self.by_start:
                self.episodes = [e for e in self.episodes if e.start.msg_id != start_id]
                self.by_start[start_id] = []
                self._ends_of_start[start_id] = []
            return self.by_start[start_id]
        scored = [(s, self._messages[eid]) for eid, s in scores.items()
                  if eid in self._messages]
        chosen = self._select_topk(a, scored)
        chosen_ids = [b.msg_id for b in chosen]
        if chosen_ids == current:
            return self.by_start.get(start_id, [])
        self.episodes = [e for e in self.episodes if e.start.msg_id != start_id]
        bucket: list[Episode] = []
        for b in chosen:
            episode = self._make_episode(a, b, scores[b.msg_id])
            bucket.append(episode)
            self.episodes.append(episode)
        self.by_start[start_id] = bucket
        self._ends_of_start[start_id] = chosen_ids
        return bucket

    def fragility(self, *, band: float = 0.5) -> dict[str, Any]:

        flips = [d for d in self._topk_decisions if d["dropped"] or d["added"]]
        return {
            "topk_decisions": len(self._topk_decisions),
            "selection_changes": len(flips),
            "band": band,
            "records": flips,
        }


    def _wov(self, a: Iterable[str], b: Iterable[str]) -> float:
        sa, sb = set(a), set(b)
        return sum(self.idf.get(w, 0.0) for w in (sa & sb))

    def _boundary(self, m: Message, kind: str, strength: str) -> Boundary:
        return Boundary(
            msg_id=m.msg_id, kind=kind, strength=strength, date=m.date, ts=m.ts,
            speaker_key=m.speaker_key, channel_key=m.channel_key,
            quote=self._quote(m, kind),
        )

    @staticmethod
    def _quote(m: Message, kind: str) -> str:

        pattern = START_DECL if kind == "start" else STRONG_END
        hit = pattern.search(m.text) or (
            None if kind == "start" else WEAK_END.search(m.text)
        )
        if hit is None:
            return " ".join(m.text.split())[:160]
        lo = max(0, hit.start() - 60)
        hi = min(len(m.text), hit.end() + 100)
        excerpt = " ".join(m.text[lo:hi].split())
        return ("…" if lo else "") + excerpt + ("…" if hi < len(m.text) else "")

    def _name(self, m: Message) -> tuple[str, tuple[str, ...]]:

        if m.tags.deliverables:
            return (" ".join(m.tags.deliverables[0].split())[:120],
                    tuple(m.tags.deliverables[:4]))
        named = _NAMED.search(m.text)
        if named:
            label = (named.group(1) or named.group(2) or "").strip()
            if label:
                return " ".join(label.split())[:120], tuple(m.tags.artifacts[:4])
        if m.tags.artifacts:
            return " ".join(str(m.tags.artifacts[0]).split())[:120], tuple(m.tags.artifacts[:4])


        hit = START_DECL.search(m.text)
        if hit:
            tail = m.text[hit.start(): hit.start() + 90]
            return " ".join(tail.split()).rstrip(" .,;:") or "unnamed task", ()
        return " ".join(m.text.split())[:90], ()


    def score_terms(self, text: str) -> frozenset[str]:
        return _content_terms(text)

    def overlap(self, a: Iterable[str], b: Iterable[str]) -> float:
        return self._wov(a, b)

    def terms_of(self, msg_id: str) -> frozenset[str]:
        return self._terms.get(msg_id, frozenset())

    def to_dict(self) -> dict[str, Any]:
        return {"config": {f: getattr(self.config, f)
                           for f in ("max_span_days", "ends_per_start",
                                     "strong_end_bonus", "span_penalty",
                                     "artifact_weight", "deliverable_weight")},
                "stats": self.stats.to_dict(),
                "corpus_fingerprint": self.corpus_fingerprint}


    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for episode in sorted(self.episodes, key=lambda e: (e.start.msg_id, e.end.msg_id)):
            digest.update(json.dumps(
                episode.to_dict(), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for episode in sorted(self.episodes,
                                  key=lambda e: (e.start.msg_id, e.end.msg_id))
        )
        records_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        generation_path = path.with_name(f"{path.stem}.{records_digest[:16]}{path.suffix}")
        if not generation_path.is_file():
            atomic_write(generation_path, payload)
        atomic_write(path, payload)
        state = {
            "n_docs": self._n_docs,
            "df": dict(sorted(self._df.items())),
            "seen_msg_ids": sorted(self._seen_msg_ids),
            "corpus_days": list(self.corpus_days),
            "strong": sorted(self._strong),
            "starts": [m.msg_id for m in self.starts],
            "ends": [m.msg_id for m in self.ends],
            "ends_of_start": {k: list(v) for k, v in sorted(self._ends_of_start.items())},
            "pending_ends_by_channel": {
                k: list(v) for k, v in sorted(self._pending_ends_by_channel.items())
            },
            "starts_by_channel": {
                k: list(v) for k, v in sorted(self._starts_by_channel.items())
            },
            "candidate_scores": {
                k: {eid: s for eid, s in sorted(v.items())}
                for k, v in sorted(self._candidate_scores.items())
            },
            "config": {
                "max_span_days": self.config.max_span_days,
                "ends_per_start": self.config.ends_per_start,
                "strong_end_bonus": self.config.strong_end_bonus,
                "span_penalty": self.config.span_penalty,
                "artifact_weight": self.config.artifact_weight,
                "deliverable_weight": self.config.deliverable_weight,
            },
        }
        metadata = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "policy_version": EPISODE_POLICY_VERSION,
            "records_file": generation_path.name,
            "records_sha256": records_digest,
            "fingerprint": self.fingerprint(),
            "corpus_fingerprint": self.corpus_fingerprint,
            "state": state,
            "state_sha256": hashlib.sha256(json.dumps(
                state, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        atomic_write(meta_path,
                     json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        log: MessageLog,
        *,
        expect_corpus: str | None = None,
    ) -> "EpisodeTable":

        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != EPISODE_SCHEMA_VERSION:
            raise ValueError(f"unsupported episode schema: {metadata.get('schema_version')}")
        if metadata.get("policy_version") != EPISODE_POLICY_VERSION:
            raise ValueError(f"unsupported episode policy: {metadata.get('policy_version')}")
        records_path = path.parent / str(metadata.get("records_file") or path.name)
        payload = records_path.read_text(encoding="utf-8")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != metadata.get("records_sha256"):
            raise ValueError("episode table record checksum mismatch")
        corpus = str(metadata.get("corpus_fingerprint") or "")
        if expect_corpus is not None and corpus != expect_corpus:
            raise ValueError("episode table corpus fingerprint mismatch")
        state = metadata.get("state") or {}
        state_digest = hashlib.sha256(json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if state_digest != metadata.get("state_sha256"):
            raise ValueError("episode table continuation-state checksum mismatch")
        raw_config = state.get("config") or {}
        table = cls.incremental(config=EpisodeConfig(
            max_span_days=int(raw_config.get("max_span_days", 30)),
            ends_per_start=int(raw_config.get("ends_per_start", 3)),
            strong_end_bonus=float(raw_config.get("strong_end_bonus", 3.0)),
            span_penalty=float(raw_config.get("span_penalty", 0.05)),
            artifact_weight=float(raw_config.get("artifact_weight", 1.0)),
            deliverable_weight=float(raw_config.get("deliverable_weight", 3.0)),
        ))
        table.log = log


        for m in log:
            table._observe_message(m)

        episodes = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            episodes.append(Episode.from_dict(json.loads(line)))
        table.episodes = episodes

        by_start: dict[str, list[Episode]] = {}
        for e in episodes:
            by_start.setdefault(e.start.msg_id, []).append(e)

        ends_of_start = {str(k): [str(x) for x in v]
                         for k, v in (state.get("ends_of_start") or {}).items()}
        for start_id, bucket in by_start.items():
            order = {eid: i for i, eid in enumerate(ends_of_start.get(start_id, ()))}
            bucket.sort(key=lambda e: order.get(e.end.msg_id, 1 << 60))
        table.by_start = by_start
        table._ends_of_start = ends_of_start


        for start_id, end_ids in ends_of_start.items():
            if not end_ids:
                table.by_start.setdefault(start_id, [])
        table._strong = {str(x) for x in state.get("strong", ())}
        start_ids = {str(x) for x in state.get("starts", ())}
        end_ids = {str(x) for x in state.get("ends", ())}
        table.starts = [m for m in log if m.msg_id in start_ids]
        table.ends = [m for m in log if m.msg_id in end_ids]
        table._start_seq = {mid: table._seq.get(mid, 0) for mid in start_ids}
        table._pending_ends_by_channel = defaultdict(list, {
            str(k): [str(x) for x in v]
            for k, v in (state.get("pending_ends_by_channel") or {}).items()
        })
        table._starts_by_channel = defaultdict(list, {
            str(k): [str(x) for x in v]
            for k, v in (state.get("starts_by_channel") or {}).items()
        })
        table._candidate_scores = defaultdict(dict, {
            str(k): {str(eid): float(s) for eid, s in v.items()}
            for k, v in (state.get("candidate_scores") or {}).items()
        })
        table._topk_decisions = []

        table.stats = EpisodeStats(
            messages=table._n_docs,
            start_declarations=len(table.starts),
            end_declarations=len(table.ends),
            strong_ends=len(table._strong),
            episodes=len(table.episodes),
            corpus_days=len(table.corpus_days),
        )
        table.stats.unpaired_starts = sum(
            1 for s in table.starts if not table.by_start.get(s.msg_id)
        )
        if table.fingerprint() != metadata.get("fingerprint"):
            raise ValueError("episode table semantic fingerprint mismatch")
        return table


def build(log: MessageLog, *, config: EpisodeConfig = DEFAULT_CONFIG) -> EpisodeTable:
    return EpisodeTable(log, config=config)
