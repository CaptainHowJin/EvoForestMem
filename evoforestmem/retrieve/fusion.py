from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence


DEFAULT_RRF_K = 60


LOGSUMEXP_TEMPERATURE = 1.0 / (DEFAULT_RRF_K + 1)

Aggregation = Literal["max", "sum", "logsumexp"]


def rrf(rank: int, k: int = DEFAULT_RRF_K) -> float:

    if rank < 1:
        raise ValueError(f"rank is 1-based, got {rank}")
    if k < 0:
        raise ValueError(f"rrf_k must be >= 0, got {k}")
    return 1.0 / (k + rank)


def dedupe_ranked(items: Iterable[str]) -> tuple[str, ...]:

    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Signal:


    name: str
    items: tuple[str, ...]
    view: str | None = None
    weight: float = 1.0
    rrf_k: int = DEFAULT_RRF_K

    @classmethod
    def of(
        cls,
        name: str,
        items: Iterable[str],
        *,
        view: str | None = None,
        weight: float = 1.0,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> "Signal":
        return cls(name=name, items=dedupe_ranked(items), view=view,
                   weight=weight, rrf_k=rrf_k)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class Fused:


    item: str
    score: float
    view: str | None = None
    rank: int = 0
    best_rank: int = 0
    signals: tuple[str, ...] = ()
    views: tuple[str, ...] = ()
    contributions: dict[str, float] = field(default_factory=dict)
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "score": round(self.score, 6),
            "view": self.view,
            "rank": self.rank,
            "signals": list(self.signals),
            "pinned": self.pinned,
        }


def fuse(signals: Sequence[Signal], *, limit: int | None = None) -> list[Fused]:

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    contributions: dict[str, dict[str, float]] = {}
    view_mass: dict[str, dict[str, float]] = {}
    order: list[str] = []

    for sig in signals:
        for rank, item in enumerate(dedupe_ranked(sig.items), start=1):
            mass = sig.weight * rrf(rank, sig.rrf_k)
            if item not in scores:
                scores[item] = 0.0
                best_rank[item] = rank
                contributions[item] = {}
                view_mass[item] = {}
                order.append(item)
            scores[item] += mass
            best_rank[item] = min(best_rank[item], rank)
            contributions[item][sig.name] = contributions[item].get(sig.name, 0.0) + mass
            if sig.view is not None:
                view_mass[item][sig.view] = view_mass[item].get(sig.view, 0.0) + mass

    ranked = sorted(order, key=lambda i: (-scores[i], best_rank[i], i))
    if limit is not None:
        ranked = ranked[:limit]

    out: list[Fused] = []
    for pos, item in enumerate(ranked, start=1):
        vm = view_mass[item]


        top_view = max(vm, key=lambda v: (vm[v], v)) if vm else None
        out.append(
            Fused(
                item=item,
                score=scores[item],
                view=top_view,
                rank=pos,
                best_rank=best_rank[item],
                signals=tuple(sorted(contributions[item])),
                views=tuple(sorted(vm)),
                contributions=dict(contributions[item]),
            )
        )
    return out


def aggregate(
    values: Sequence[float],
    mode: Aggregation = "max",
    *,
    temperature: float = LOGSUMEXP_TEMPERATURE,
) -> float:

    if not values:
        return 0.0
    if mode == "max":
        return max(values)
    if mode == "sum":
        return float(sum(values))
    if mode == "logsumexp":
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        top = max(values)

        return top + temperature * math.log(
            sum(math.exp((v - top) / temperature) for v in values)
        )
    raise ValueError(f"unknown aggregation {mode!r}")


@dataclass(frozen=True, slots=True)
class CrossView:


    item: str
    score: float
    view: str | None
    views: tuple[str, ...]
    per_view: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def owner(self) -> str:

        return max(self.per_view, key=lambda v: (self.per_view[v], v))

    @property
    def exclusive(self) -> bool:
        return len(self.views) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "score": round(self.score, 6),
            "view": self.view,
            "views": list(self.views),
            "per_view": {v: round(s, 6) for v, s in sorted(self.per_view.items())},
        }


def aggregate_views(
    per_view: Mapping[str, Sequence[tuple[str, float]]],
    *,
    mode: Aggregation = "max",
    temperature: float = LOGSUMEXP_TEMPERATURE,
    payloads: Mapping[tuple[str, str], Any] | None = None,
    charge_shared: bool = True,
) -> list[CrossView]:

    collected: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for view, items in per_view.items():
        best: dict[str, float] = {}
        for item, score in items:
            if item not in best or score > best[item]:
                best[item] = score
        for item, score in best.items():
            if item not in collected:
                collected[item] = {}
                order.append(item)
            collected[item][view] = score

    out: list[CrossView] = []
    for item in order:
        scores = collected[item]


        owner = max(scores, key=lambda v: (scores[v], v))
        charged: str | None = owner
        if not charge_shared and len(scores) > 1:
            charged = None
        payload = {}
        if payloads is not None:
            payload = {v: payloads.get((v, item)) for v in scores if (v, item) in payloads}
        out.append(
            CrossView(
                item=item,
                score=aggregate(list(scores.values()), mode, temperature=temperature),
                view=charged,
                views=tuple(sorted(scores)),
                per_view=dict(scores),
                payload=payload,
            )
        )
    out.sort(key=lambda c: (-c.score, c.item))
    return out


class Scored(Protocol):


    item: str
    score: float
    view: str | None


@dataclass(frozen=True, slots=True)
class Budget:


    per_view: Mapping[str, int] = field(default_factory=dict)
    default: int = 2
    total: int | None = None


    SHARED = "*"

    def limit_for(self, view: str | None) -> int:
        if view is None:


            return self.total if self.total is not None else 1 << 30
        return int(self.per_view.get(view, self.default))

    @classmethod
    def uniform(cls, views: Iterable[str], n: int, *, total: int | None = None) -> "Budget":
        return cls(per_view={v: n for v in views}, default=n, total=total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_view": dict(sorted(self.per_view.items())),
            "default": self.default,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class Allocation:


    selected: tuple[Any, ...]
    per_view: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    injected: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    displaced: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.selected)

    def __iter__(self):
        return iter(self.selected)

    @property
    def items(self) -> tuple[str, ...]:
        return tuple(s.item for s in self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": len(self.selected),
            "per_view": dict(sorted(self.per_view.items())),
            "dropped": dict(sorted(self.dropped.items())),
            "injected": list(self.injected),
            "refused": list(self.refused),
            "displaced": list(self.displaced),
        }


def allocate(
    candidates: Sequence[Any],
    budget: Budget,
    *,
    pinned: Sequence[str] = (),
) -> Allocation:

    pins = dedupe_ranked(pinned)
    pinset = set(pins)
    ordered = sorted(candidates, key=lambda c: (-c.score, c.item))

    selected: list[Any] = []
    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}
    seen: set[str] = set()

    for cand in ordered:
        if cand.item in seen:
            continue
        view = cand.view if cand.view is not None else Budget.SHARED
        limit = budget.limit_for(cand.view)
        at_total = budget.total is not None and len(selected) >= budget.total
        if counts.get(view, 0) >= limit or at_total:
            dropped[view] = dropped.get(view, 0) + 1
            continue
        seen.add(cand.item)
        counts[view] = counts.get(view, 0) + 1
        selected.append(replace(cand, pinned=cand.item in pinset)
                        if hasattr(cand, "pinned") else cand)


    injected: list[str] = []
    refused: list[str] = []
    displaced: list[str] = []
    by_item = {c.item: c for c in ordered}

    for pin in pins:
        if pin in seen:
            continue
        cand = by_item.get(pin)
        if cand is None:
            refused.append(pin)
            continue
        view = cand.view if cand.view is not None else Budget.SHARED
        limit = budget.limit_for(cand.view)
        if limit <= 0:
            refused.append(pin)
            continue
        pinned_cand = replace(cand, pinned=True) if hasattr(cand, "pinned") else cand
        if counts.get(view, 0) < limit and not (
            budget.total is not None and len(selected) >= budget.total
        ):
            selected.append(pinned_cand)
            counts[view] = counts.get(view, 0) + 1
            seen.add(pin)
            injected.append(pin)
            continue


        victims = [
            (i, s) for i, s in enumerate(selected)
            if (s.view if s.view is not None else Budget.SHARED) == view
            and s.item not in pinset
        ]
        if not victims:
            refused.append(pin)
            continue
        idx = min(victims, key=lambda pair: (pair[1].score, pair[1].item))[0]
        displaced.append(selected[idx].item)
        seen.discard(selected[idx].item)
        selected[idx] = pinned_cand
        seen.add(pin)
        injected.append(pin)
        dropped[view] = dropped.get(view, 0) + 1


    selected.sort(key=lambda s: (-s.score, getattr(s, "pinned", False), s.item))
    return Allocation(
        selected=tuple(selected),
        per_view=dict(sorted(counts.items())),
        dropped=dict(sorted(dropped.items())),
        injected=tuple(injected),
        refused=tuple(refused),
        displaced=tuple(displaced),
    )
