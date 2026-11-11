from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
from .fusion import DEFAULT_RRF_K, Aggregation, Budget
from .s4_assemble import Span
BindMode = Literal['signal', 'pin', 'filter']

@dataclass(frozen=True, slots=True)
class S1Config:

    msg_pool: int = 200
    msg_rrf_k: int = DEFAULT_RRF_K
    sparse_weight: float = 1.0
    dense_weight: float = 0.5
    dense_weight_temporal: float | None = None
    dense_weight_skill: float | None = None
    w_root: float = 0.8
    w_leaf2tree: float = 2.0
    w_bind: float = 1.2
    w_collapsed: float = 0.0
    collapsed_recall: bool = False
    collapsed_k: int = 32
    collapsed_dedup: bool = True
    tree_rrf_k: int = 0
    leaf2tree_rrf_k: int = DEFAULT_RRF_K
    max_hits_per_tree: Mapping[str, int | None] | int | None = field(default_factory=lambda: {'channel': 3, 'actor': 1, 'artifact': 1})
    require_summaries_for_root: bool = True
    variant_decay: float = 0.5
    trees_per_view: Mapping[str, int] = field(default_factory=lambda: {'channel': 2, 'actor': 4, 'artifact': 2})
    spans_per_view: Mapping[str, int | float] = field(default_factory=lambda: {'channel': 14, 'actor': 8, 'artifact': 0.2})
    persona_spans_per_view: Mapping[str, int | float] = field(default_factory=lambda: {'channel': 8, 'actor': 14, 'artifact': 0.1})
    default_trees_per_view: int = 2
    default_spans_per_view: int | float = 6
    max_spans: int = 20
    span_rrf_k: int = 4
    aggregation: Aggregation = 'max'
    bind_mode: BindMode = 'pin'

    def hits_cap_for(self, view: str) -> int | None:
        if isinstance(self.max_hits_per_tree, Mapping):
            return self.max_hits_per_tree.get(view, 1)
        return self.max_hits_per_tree

    def tree_budget(self, views: Sequence[str]) -> Budget:
        return Budget(per_view={v: int(self.trees_per_view.get(v, self.default_trees_per_view)) for v in views}, default=self.default_trees_per_view)

    def span_budget(self, views: Sequence[str], *, persona: bool=False, planned: Mapping[str, int] | None=None) -> Budget:
        table = self.persona_spans_per_view if persona else self.spans_per_view

        def resolve_span_count(value: int | float) -> int:

            if isinstance(value, int) or value > 1.0:
                return int(value)
            if value <= 0.0:
                return 0
            return max(1, round(value * self.max_spans))
        planned = planned or {}
        return Budget(per_view={v: max(0, int(planned[v])) if v in planned else resolve_span_count(table.get(v, self.default_spans_per_view)) for v in views}, default=resolve_span_count(self.default_spans_per_view), total=self.max_spans)

    def to_dict(self) -> dict[str, Any]:
        return {'msg_pool': self.msg_pool, 'message_weights': {'sparse': self.sparse_weight, 'dense': self.dense_weight, 'dense_temporal': self.dense_weight_temporal, 'dense_skill': self.dense_weight_skill}, 'weights': {'root': self.w_root, 'leaf2tree': self.w_leaf2tree, 'bind': self.w_bind, 'collapsed': self.w_collapsed}, 'collapsed_recall': self.collapsed_recall, 'collapsed_k': self.collapsed_k, 'collapsed_dedup': self.collapsed_dedup, 'rrf_k': {'msg': self.msg_rrf_k, 'leaf2tree': self.leaf2tree_rrf_k, 'tree': self.tree_rrf_k, 'span': self.span_rrf_k}, 'max_hits_per_tree': dict(self.max_hits_per_tree) if isinstance(self.max_hits_per_tree, Mapping) else self.max_hits_per_tree, 'require_summaries_for_root': self.require_summaries_for_root, 'trees_per_view': dict(sorted(self.trees_per_view.items())), 'spans_per_view': dict(sorted(self.spans_per_view.items())), 'max_spans': self.max_spans, 'aggregation': self.aggregation, 'bind_mode': self.bind_mode}

@dataclass(frozen=True, slots=True)
class TreeHit:

    tree_id: str
    view: str
    score: float
    rank: int
    signals: tuple[str, ...] = ()
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {'tree_id': self.tree_id, 'view': self.view, 'score': round(self.score, 6), 'rank': self.rank, 'signals': list(self.signals), 'pinned': self.pinned}

@dataclass(frozen=True, slots=True)
class SpanHit:

    msg_id: str
    score: float
    view: str
    tree_id: str
    rank: int
    views: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {'msg_id': self.msg_id, 'score': round(self.score, 6), 'view': self.view, 'tree_id': self.tree_id, 'rank': self.rank, 'views': list(self.views)}

@dataclass(frozen=True, slots=True)
class S1Plan:

    question: str
    trees: tuple[TreeHit, ...] = ()
    spans: tuple[SpanHit, ...] = ()
    messages: tuple[tuple[str, float], ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)

    def tree_ids(self, view: str | None=None) -> tuple[str, ...]:
        return tuple((t.tree_id for t in self.trees if view is None or t.view == view))

    def to_spans(self, headers_of=None) -> list[Span]:

        out: list[Span] = []
        for s in self.spans:
            headers = tuple(headers_of(s.tree_id, s.msg_id)) if headers_of else ()
            out.append(Span(msg_id=s.msg_id, score=s.score, view=s.tree_id, headers=headers))
        return out

    def fallback_spans(self, n: int=20) -> list[Span]:

        return [Span(msg_id=m, score=s) for m, s in self.messages[:n]]

    def to_dict(self) -> dict[str, Any]:
        return {'trees': [t.to_dict() for t in self.trees], 'spans': [s.to_dict() for s in self.spans], 'stats': self.stats}
