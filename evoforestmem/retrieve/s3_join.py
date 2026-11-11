from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Sequence
from ..log.store import MessageLog
from .s4_assemble import Span
JoinMode = Literal['none', 'event_pair', 'pivot_same', 'pivot_other', 'task_span']
AnchorRole = Literal['a', 'b']
Origin = Literal['seed', 'anchor_a', 'anchor_b']
CHANNEL_VIEW = 'channel'
ACTOR_VIEW = 'actor'
_PIVOT = re.compile('\\b(?:next|new)\\s+(?:independent\\s+)?(?:task|work|assignment|job)\\b', re.IGNORECASE)
_SCOPE_OTHER = re.compile('\\b(?:in|within)\\s+(?:other|another|a different|different)\\s+project\\s+(?:group|cluster|team)s?\\b', re.IGNORECASE)
_SCOPE_SAME = re.compile('\\b(?:in|within)\\s+(?:this|that|the same|the)\\s+project\\s+(?:group|cluster|team)\\b', re.IGNORECASE)
_DONE = re.compile('\\b(?:complet\\w+|finish\\w+|finali[sz]\\w+|deliver\\w+|conclud\\w+|closed|wrapp?ed\\s+up|was\\s+developed|were\\s+developed|has\\s+been\\s+done)\\b', re.IGNORECASE)
_BEGAN = re.compile('\\b(?:start\\w*|begin\\w*|began|begun|kick(?:ed)?[-\\s]?off|launch\\w*|initiat\\w+|commenc\\w+|took\\s+on|take\\s+on|draft\\w*)\\b', re.IGNORECASE)
_COMBINED_SPLIT = re.compile('\\bdid\\b', re.IGNORECASE)
_SCAFFOLD = re.compile('\\b(?:project\\s+(?:group|cluster)s?|group)\\b|\\bfor\\s+the\\b|\\bin\\s+the\\b|\\bdid\\s+the\\b|\\bto\\s+the\\b', re.IGNORECASE)
_SUBORDINATORS = frozenset({'after', 'once', 'following', 'since', 'prior to'})
_COMBINED = frozenset({'how long after', 'how long before'})

def lifecycle_hint(text: str) -> str | None:

    done, began = (_DONE.search(text), _BEGAN.search(text))
    if done and began:
        return 'complete' if done.start() < began.start() else 'start'
    if done:
        return 'complete'
    if began:
        return 'start'
    return None

@dataclass(frozen=True, slots=True)
class Anchor:

    role: AnchorRole
    text: str
    query: str
    channel: str | None = None
    exclude_channel: bool = False
    lifecycle: str | None = None
    pivot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {'role': self.role, 'text': self.text, 'query': self.query, 'channel': self.channel, 'exclude_channel': self.exclude_channel, 'lifecycle': self.lifecycle, 'pivot': self.pivot}

@dataclass(frozen=True, slots=True)
class Decomposition:

    mode: JoinMode
    anchor_a: Anchor | None = None
    anchor_b: Anchor | None = None
    tasks: tuple[tuple[str, str | None], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def fires(self) -> bool:
        return self.mode != 'none' and (self.anchor_a is not None or bool(self.tasks))

    def to_dict(self) -> dict[str, Any]:
        return {'mode': self.mode, 'anchor_a': self.anchor_a.to_dict() if self.anchor_a else None, 'anchor_b': self.anchor_b.to_dict() if self.anchor_b else None, 'tasks': [{'query': q, 'speaker': s} for q, s in self.tasks], 'evidence': self.evidence}

@dataclass(frozen=True, slots=True)
class JoinedPair:

    a_msg_id: str
    b_msg_id: str
    score: float
    a_ts: float = 0.0
    b_ts: float = 0.0
    a_channel: str = ''
    b_channel: str = ''
    speaker: str | None = None
    task: str = ''

    @property
    def cross_channel(self) -> bool:
        return bool(self.a_channel) and self.a_channel != self.b_channel

    @property
    def days(self) -> float:

        return (self.b_ts - self.a_ts) / 86400.0 + 1.0

    @property
    def working_days(self) -> int:

        start = datetime.fromtimestamp(self.a_ts, tz=timezone.utc).date()
        end = datetime.fromtimestamp(self.b_ts, tz=timezone.utc).date()
        if end < start:
            return 0
        days = 0
        cursor = start
        while cursor <= end:
            days += cursor.weekday() < 5
            cursor += timedelta(days=1)
        return days

    def to_dict(self) -> dict[str, Any]:
        return {'a': self.a_msg_id, 'b': self.b_msg_id, 'score': round(self.score, 6), 'a_channel': self.a_channel, 'b_channel': self.b_channel, 'cross_channel': self.cross_channel, 'days': round(self.days, 2), 'working_days': self.working_days, 'speaker': self.speaker, 'task': self.task}

@dataclass
class JoinStats:

    mode: JoinMode = 'none'
    applied: bool = False
    n_seeds: int = 0
    n_emitted: int = 0
    n_join_only: int = 0
    n_pairs: int = 0
    n_cross_channel_pairs: int = 0
    n_anchor_a: int = 0
    n_anchor_b: int = 0
    walk_steps: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def placement_join_hit_rate(self) -> float:
        return self.n_join_only / self.n_emitted if self.n_emitted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {'mode': self.mode, 'applied': self.applied, 'n_seeds': self.n_seeds, 'n_emitted': self.n_emitted, 'n_join_only': self.n_join_only, 'placement_join_hit_rate': round(self.placement_join_hit_rate, 4), 'n_pairs': self.n_pairs, 'n_cross_channel_pairs': self.n_cross_channel_pairs, 'n_anchor_a': self.n_anchor_a, 'n_anchor_b': self.n_anchor_b, 'walk_steps': self.walk_steps, **({'notes': self.notes} if self.notes else {})}

@dataclass
class JoinResult:

    spans: list[Span] = field(default_factory=list)
    pairs: list[JoinedPair] = field(default_factory=list)
    selected_pair: JoinedPair | None = None
    decomposition: Decomposition | None = None
    origin: dict[str, Origin] = field(default_factory=dict)
    stats: JoinStats = field(default_factory=JoinStats)

    @property
    def applied(self) -> bool:
        return self.stats.applied

    @property
    def msg_ids(self) -> list[str]:
        return [s.msg_id for s in self.spans]

    def to_dict(self) -> dict[str, Any]:
        return {'spans': [s.msg_id for s in self.spans], 'pairs': [p.to_dict() for p in self.pairs], 'selected_pair': self.selected_pair.to_dict() if self.selected_pair else None, 'decomposition': self.decomposition.to_dict() if self.decomposition else None, 'stats': self.stats.to_dict()}

@dataclass(frozen=True, slots=True)
class JoinConfig:

    k_anchor: int = 8
    k_pivot: int = 3
    max_pairs: int = 12
    pivot_lifecycle: str | None = 'start'
    strip_channel_phrases: bool = True
    require_forward: bool = True
    task_pair_margin: float = 1.15
    task_max_days: int = 90
    episode_pair_select: bool = False
DEFAULT_CONFIG = JoinConfig()

def episode_pair_selector(log: MessageLog) -> Callable[[str, Sequence[JoinedPair], str], int | None]:


    def select(question: str, cands: Sequence[JoinedPair], tag: str) -> int | None:
        best_key: tuple[int, float, int] | None = None
        best_i: int | None = None
        for i, p in enumerate(cands):
            a = log.get(p.a_msg_id) if p.a_msg_id in log else None
            b = log.get(p.b_msg_id) if p.b_msg_id in log else None
            if a is None or b is None:
                continue
            a_lc = a.tags.lifecycle or ''
            b_lc = b.tags.lifecycle or ''
            if not (a_lc in ('complete', 'archive') and b_lc == 'start'):
                continue
            key = (0 if p.a_ts <= p.b_ts else 1, -p.score, i)
            if best_key is None or key < best_key:
                best_key, best_i = (key, i)
        return best_i
    return select
