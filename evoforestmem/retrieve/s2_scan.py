from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
from .s4_assemble import Span
Mode = Literal['scan', 'beam']
Aggregation = Literal['max', 'mean', 'sum']
NEG_INF = -math.inf
DEFAULT_EPSILON = 0.05
_CURRENT_STATE = re.compile('\\b(current(?:ly)?|now|nowadays|latest|newest|most\\s+recent(?:ly)?|as\\s+of\\s+(?:now|today|this)|at\\s+present|presently|up[-\\s]to[-\\s]date|still|in\\s+effect|finally|eventually|ultimately|in\\s+the\\s+end|end(?:ed|s)?\\s+up|finali[sz]ed?|final\\s+(?:version|state|value|decision|status|form|result|answer|conclusion|design))\\b', re.IGNORECASE)

@dataclass(frozen=True, slots=True)
class NodePath:

    view: str
    tree_id: str
    node_id: str
    level: int
    chain: tuple[str, ...]
    msg_ids: tuple[str, ...]
    headers: tuple[str, ...]
    score: float
    base_score: float
    penalty: float
    ts_start: float
    ts_end: float
    matched_node_id: str | None = None
    matched_level: int | None = None

    @property
    def leaf_count(self) -> int:
        return len(self.msg_ids)

    @property
    def depth(self) -> int:
        return len(self.chain)

    def to_spans(self, *, max_leaves: int | None=None) -> list[Span]:

        ids = self.msg_ids if max_leaves is None else self.msg_ids[:max_leaves]
        return [Span(msg_id=mid, score=self.score, view=self.view, headers=self.headers) for mid in ids]

    def to_dict(self) -> dict[str, Any]:
        return {'view': self.view, 'tree_id': self.tree_id, 'node_id': self.node_id, 'level': self.level, 'chain': list(self.chain), 'n_msgs': len(self.msg_ids), 'score': self.score, 'base_score': self.base_score, 'penalty': self.penalty, 'matched_node_id': self.matched_node_id}

@dataclass
class S2Result:
    paths: list[NodePath] = field(default_factory=list)
    mode: Mode = 'scan'
    nodes_scored: int = 0
    nodes_selected: int = 0
    trees_searched: int = 0
    penalised: int = 0
    asks_current: bool = False
    elapsed_ms: float = 0.0

    def spans(self, *, max_leaves: int | None=None, max_spans: int | None=None, dedupe: bool=True) -> list[Span]:

        out: list[Span] = []
        best: dict[str, Span] = {}
        for path in self.paths:
            for span in path.to_spans(max_leaves=max_leaves):
                if not dedupe:
                    out.append(span)
                    continue
                prev = best.get(span.msg_id)
                if prev is None or span.score > prev.score:
                    best[span.msg_id] = span
        if dedupe:
            out = sorted(best.values(), key=lambda s: -s.score)
        return out if max_spans is None else out[:max_spans]

    def to_dict(self) -> dict[str, Any]:
        return {'mode': self.mode, 'n_paths': len(self.paths), 'nodes_scored': self.nodes_scored, 'nodes_selected': self.nodes_selected, 'trees_searched': self.trees_searched, 'penalised': self.penalised, 'asks_current': self.asks_current, 'elapsed_ms': round(self.elapsed_ms, 3), 'paths': [p.to_dict() for p in self.paths]}

@dataclass(frozen=True, slots=True)
class S2Config:

    mode: Mode = 'scan'
    m: int = 8
    beam_width: int | None = None
    refine_to: int | None = 1
    max_leaves_per_path: int | None = None
    missing_score: float = 0.0
    min_level: int = 1
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        if self.mode not in ('scan', 'beam'):
            raise ValueError(f"unknown s2 mode {self.mode!r}; expected 'scan' or 'beam'")
        if self.m < 1:
            raise ValueError(f'm must be >= 1, got {self.m}')
        if self.beam_width is not None and self.beam_width < 1:
            raise ValueError(f'beam_width must be >= 1, got {self.beam_width}')
        if self.refine_to is not None and self.refine_to < 1:
            raise ValueError(f'refine_to must be >= 1 (L0 holds no summary), got {self.refine_to}')

    @property
    def effective_beam(self) -> int:
        return self.m if self.beam_width is None else self.beam_width

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> 'S2Config':
        d = d or {}
        base = cls()
        return cls(mode=str(d.get('mode', base.mode)), m=int(d.get('m', base.m)), beam_width=None if d.get('beam_width') is None else int(d['beam_width']), refine_to=None if 'refine_to' in d and d['refine_to'] is None else int(d.get('refine_to', base.refine_to)), max_leaves_per_path=None if d.get('max_leaves_per_path') is None else int(d['max_leaves_per_path']), missing_score=float(d.get('missing_score', base.missing_score)), min_level=int(d.get('min_level', base.min_level)), epsilon=float(d.get('epsilon', base.epsilon)))
