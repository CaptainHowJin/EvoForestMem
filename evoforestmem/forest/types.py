from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from evoforestmem.forest.btree import empty_tree
from evoforestmem.forest.tree_types import MemTree, MemTreeNode, TreeCard


View = Literal["channel", "actor", "artifact"]

VIEWS: tuple[str, ...] = get_args(View)
PHASE1_VIEWS: tuple[str, ...] = ("channel", "actor")


PolyNode = MemTreeNode


PolyTree = MemTree

__all__ = [
    "View",
    "VIEWS",
    "PHASE1_VIEWS",
    "PolyNode",
    "PolyTree",
    "TreeCard",
    "ViewConfig",
    "ForestConfig",
    "DEFAULT_FOREST_CONFIG",
    "tree_id_for",
    "split_tree_id",
    "new_tree",
]


def tree_id_for(view: str, key: str) -> str:

    if ":" in view:
        raise ValueError(f"view must not contain ':': {view!r}")
    return f"{view}:{key}"


def split_tree_id(tree_id: str) -> tuple[str, str]:
    view, _, key = tree_id.partition(":")
    if not key:
        raise ValueError(f"malformed tree_id: {tree_id!r}")
    return view, key


@dataclass(frozen=True, slots=True)
class ViewConfig:


    view: str
    k: int = 16
    rrf_k: int = 60
    enabled: bool = True
    grouping: str = "time"
    min_thread: int = 2


    seal_on_append: bool = True

    def __post_init__(self) -> None:
        if self.view not in VIEWS:
            raise ValueError(f"unknown view {self.view!r}; expected one of {VIEWS}")
        if self.k < 2:
            raise ValueError(f"k must be >= 2 for view {self.view!r}, got {self.k}")
        if self.grouping not in {"time", "thread"}:
            raise ValueError(f"unknown grouping {self.grouping!r} for view {self.view!r}")
        if self.grouping != "time" and self.view != "channel":
            raise ValueError("thread grouping is valid only for the channel view")
        if self.min_thread < 2:
            raise ValueError("min_thread must be >= 2")


@dataclass(frozen=True, slots=True)
class ForestConfig:


    views: tuple[ViewConfig, ...]

    def get(self, view: str) -> ViewConfig:
        for cfg in self.views:
            if cfg.view == view:
                return cfg
        raise KeyError(f"view {view!r} is not configured")

    def k_for(self, view: str) -> int:
        return self.get(view).k

    def grouping_for(self, view: str) -> str:
        return self.get(view).grouping

    def min_thread_for(self, view: str) -> int:
        return self.get(view).min_thread

    def seal_on_append_for(self, view: str) -> bool:
        return self.get(view).seal_on_append

    @property
    def enabled_views(self) -> tuple[str, ...]:
        return tuple(cfg.view for cfg in self.views if cfg.enabled)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ForestConfig":

        d = d or {}
        out: list[ViewConfig] = []
        for base in DEFAULT_FOREST_CONFIG.views:
            over = d.get(base.view) or {}
            out.append(
                ViewConfig(
                    view=base.view,
                    k=int(over.get("k", base.k)),
                    rrf_k=int(over.get("rrf_k", base.rrf_k)),
                    enabled=bool(over.get("enabled", base.enabled)),
                    grouping=str(over.get("grouping", base.grouping)),
                    min_thread=int(over.get("min_thread", base.min_thread)),
                    seal_on_append=bool(over.get("seal_on_append", base.seal_on_append)),
                )
            )
        return cls(views=tuple(out))


DEFAULT_FOREST_CONFIG = ForestConfig(
    views=(

        ViewConfig(view="channel", k=16),

        ViewConfig(view="actor", k=16),

        ViewConfig(view="artifact", k=16),
    )
)


def new_tree(
    view: str, key: str, *, k: int, label: str | None = None,
    seal_on_append: bool = False,
) -> PolyTree:

    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    tree_id = tree_id_for(view, key)
    return empty_tree(
        tree_id, view, key, label or key, k, append_sealed=seal_on_append,
    )
