from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

DEFAULT_ROOT = Path("data/index")


DEFAULT_RRF_K = 60


DEFAULT_MAX_HITS_PER_TREE = 3

Weighting = Literal["rrf", "count"]


@dataclass(frozen=True, slots=True)
class Placement:


    view: str
    tree_id: str
    leaf_node_id: str
    leaf_pos: int

    def to_list(self) -> list[Any]:
        return [self.view, self.tree_id, self.leaf_node_id, self.leaf_pos]

    @classmethod
    def from_list(cls, row: Sequence[Any]) -> "Placement":
        return cls(view=str(row[0]), tree_id=str(row[1]), leaf_node_id=str(row[2]),
                   leaf_pos=int(row[3]))


class PlacementMap:


    def __init__(self) -> None:
        self._by_msg: dict[str, list[Placement]] = {}


        self._order: dict[str, list[str]] = {}
        self._view_of_tree: dict[str, str] = {}
        self._stale = False
        self._out_of_order = 0
        self._reindexes = 0

    def replace_with(self, other: "PlacementMap") -> None:

        self._by_msg = {msg_id: list(rows) for msg_id, rows in other._by_msg.items()}
        self._order = {tree_id: list(msg_ids) for tree_id, msg_ids in other._order.items()}
        self._view_of_tree = dict(other._view_of_tree)
        self._stale = other._stale
        self._out_of_order = other._out_of_order
        self._reindexes = other._reindexes


    def on_place(self, tree, msg_id: str, leaf_node_id: str, leaf_pos: int) -> None:

        self.add(
            view=tree.view,
            tree_id=tree.tree_id,
            msg_id=msg_id,
            leaf_node_id=leaf_node_id,
            leaf_pos=leaf_pos,
        )

    def on_unplace(self, tree, msg_id: str, leaf_node_id: str) -> None:

        self.remove(tree.tree_id, msg_id)

    def add(
        self,
        *,
        view: str,
        tree_id: str,
        msg_id: str,
        leaf_node_id: str,
        leaf_pos: int,
    ) -> Placement:

        order = self._order.setdefault(tree_id, [])
        for existing in self._by_msg.get(msg_id, ()):
            if existing.tree_id == tree_id:
                return existing

        pos = max(0, min(int(leaf_pos), len(order)))
        if pos == len(order):
            order.append(msg_id)
        else:


            order.insert(pos, msg_id)
            self._out_of_order += 1
            self._stale = True

        placement = Placement(view=view, tree_id=tree_id, leaf_node_id=leaf_node_id, leaf_pos=pos)
        self._by_msg.setdefault(msg_id, []).append(placement)
        self._view_of_tree[tree_id] = view
        return placement

    def remove(self, tree_id: str, msg_id: str) -> bool:
        placements = self._by_msg.get(msg_id)
        if not placements:
            return False
        kept = [p for p in placements if p.tree_id != tree_id]
        if len(kept) == len(placements):
            return False
        if kept:
            self._by_msg[msg_id] = kept
        else:
            del self._by_msg[msg_id]
        order = self._order.get(tree_id)
        if order is not None and msg_id in order:
            order.remove(msg_id)
            self._stale = True
        return True

    def drop_tree(self, tree_id: str) -> int:

        n = 0
        for msg_id in list(self._by_msg):
            kept = [p for p in self._by_msg[msg_id] if p.tree_id != tree_id]
            n += len(self._by_msg[msg_id]) - len(kept)
            if kept:
                self._by_msg[msg_id] = kept
            else:
                del self._by_msg[msg_id]
        self._order.pop(tree_id, None)
        self._view_of_tree.pop(tree_id, None)
        return n

    def index_tree(self, tree) -> int:

        self.drop_tree(tree.tree_id)
        n = 0
        for pos, msg_id in enumerate(tree.payload_ids_ordered):
            leaf_node_id = tree.leaf_by_payload.get(msg_id)
            if leaf_node_id is None:
                raise ValueError(
                    f"tree {tree.tree_id} lists payload {msg_id!r} in its leaf order but "
                    f"has no leaf node for it -- the tree is inconsistent, not the map"
                )
            self.add(
                view=tree.view,
                tree_id=tree.tree_id,
                msg_id=msg_id,
                leaf_node_id=leaf_node_id,
                leaf_pos=pos,
            )
            n += 1
        return n

    @classmethod
    def from_trees(cls, trees: Iterable) -> "PlacementMap":
        pm = cls()
        for tree in trees:
            pm.index_tree(tree)
        return pm


    def reindex(self) -> int:

        pos_by_tree = {
            tree_id: {msg_id: i for i, msg_id in enumerate(order)}
            for tree_id, order in self._order.items()
        }
        fixed = 0
        for msg_id, placements in self._by_msg.items():
            for i, p in enumerate(placements):
                correct = pos_by_tree.get(p.tree_id, {}).get(msg_id)
                if correct is None or correct == p.leaf_pos:
                    continue
                placements[i] = replace(p, leaf_pos=correct)
                fixed += 1
        self._stale = False
        self._reindexes += 1
        return fixed

    def _ensure_fresh(self) -> None:
        if self._stale:
            self.reindex()


    def __len__(self) -> int:
        return len(self._by_msg)

    def __contains__(self, msg_id: object) -> bool:
        return msg_id in self._by_msg

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_msg)

    @property
    def n_placements(self) -> int:
        return sum(len(v) for v in self._by_msg.values())

    @property
    def tree_ids(self) -> list[str]:
        return sorted(self._order)

    def views(self) -> list[str]:
        return sorted(set(self._view_of_tree.values()))

    def view_of(self, tree_id: str) -> str | None:
        return self._view_of_tree.get(tree_id)

    def placements_of(self, msg_id: str) -> tuple[Placement, ...]:
        self._ensure_fresh()
        return tuple(self._by_msg.get(msg_id, ()))

    def in_view(self, msg_id: str, view: str) -> tuple[Placement, ...]:
        return tuple(p for p in self.placements_of(msg_id) if p.view == view)

    def leaf_order(self, tree_id: str) -> tuple[str, ...]:

        self._ensure_fresh()
        return tuple(self._order.get(tree_id, ()))

    def as_dict(self) -> dict[str, tuple[Placement, ...]]:

        self._ensure_fresh()
        return {
            msg_id: tuple(sorted(ps, key=lambda p: (p.view, p.tree_id)))
            for msg_id, ps in self._by_msg.items()
        }


    def trees_of(
        self,
        msg_ids_ranked: Sequence[str],
        view: str | None = None,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        rrf_k_by_view: Mapping[str, int] | None = None,
        max_hits_per_tree: int | None = DEFAULT_MAX_HITS_PER_TREE,
        weighting: Weighting = "rrf",
        top_n: int | None = None,
    ) -> dict[str, float]:

        scores: dict[str, float] = {}
        used: Counter[str] = Counter()
        seen: set[str] = set()
        rank = 0
        for msg_id in msg_ids_ranked:
            if msg_id in seen:
                continue
            seen.add(msg_id)
            rank += 1
            for p in self._by_msg.get(msg_id, ()):
                if view is not None and p.view != view:
                    continue
                if weighting == "count":
                    scores[p.tree_id] = scores.get(p.tree_id, 0.0) + 1.0
                    continue
                if max_hits_per_tree is not None and used[p.tree_id] >= max_hits_per_tree:
                    continue
                used[p.tree_id] += 1
                k = rrf_k if rrf_k_by_view is None else rrf_k_by_view.get(p.view, rrf_k)
                scores[p.tree_id] = scores.get(p.tree_id, 0.0) + 1.0 / (k + rank)


        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if top_n is not None:
            ordered = ordered[:top_n]
        return dict(ordered)


    def window(
        self,
        msg_id: str,
        view: str,
        w: int,
        *,
        include_self: bool = True,
    ) -> list[str]:

        if w < 0:
            raise ValueError(f"window width must be >= 0, got {w}")
        self._ensure_fresh()
        out: list[str] = []
        seen: set[str] = set()
        for p in self._by_msg.get(msg_id, ()):
            if p.view != view:
                continue
            order = self._order.get(p.tree_id) or []
            lo = max(0, p.leaf_pos - w)
            hi = min(len(order), p.leaf_pos + w + 1)
            for neighbour in order[lo:hi]:
                if neighbour == msg_id and not include_self:
                    continue
                if neighbour not in seen:
                    seen.add(neighbour)
                    out.append(neighbour)
        return out

    def cotenants(self, msg_id: str, exclude_view: str | None = None) -> tuple[Placement, ...]:

        self._ensure_fresh()
        return tuple(
            p for p in self._by_msg.get(msg_id, ())
            if exclude_view is None or p.view != exclude_view
        )


    @staticmethod
    def path_for(batch: str, root: str | Path = DEFAULT_ROOT) -> Path:
        return Path(root) / batch / "placement.jsonl"

    def save(self, path: str | Path) -> Path:

        self._ensure_fresh()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for msg_id, placements in self._by_msg.items():
                fh.write(json.dumps(
                    {"msg_id": msg_id, "placements": [p.to_list() for p in placements]},
                    ensure_ascii=False,
                ))
                fh.write("\n")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PlacementMap":
        pm = cls()
        rows: list[tuple[str, int, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msg_id = str(d["msg_id"])
                placements = [Placement.from_list(row) for row in d.get("placements", ())]
                pm._by_msg[msg_id] = placements
                for p in placements:
                    pm._view_of_tree[p.tree_id] = p.view
                    rows.append((p.tree_id, p.leaf_pos, msg_id))


        rows.sort()
        for tree_id, _pos, msg_id in rows:
            pm._order.setdefault(tree_id, []).append(msg_id)
        for tree_id, order in pm._order.items():
            expected = list(range(len(order)))
            actual = sorted(p.leaf_pos for m in order for p in pm._by_msg[m] if p.tree_id == tree_id)
            if actual != expected:


                pm._stale = True
        pm._ensure_fresh()
        return pm


    def stats(self) -> dict[str, Any]:
        self._ensure_fresh()
        per_view: dict[str, dict[str, int]] = {}
        for tree_id, order in self._order.items():
            view = self._view_of_tree.get(tree_id, "?")
            v = per_view.setdefault(view, {"trees": 0, "placements": 0})
            v["trees"] += 1
            v["placements"] += len(order)
        sizes = [len(o) for o in self._order.values()]


        in_degree = Counter(len(ps) for ps in self._by_msg.values())
        return {
            "messages": len(self._by_msg),
            "placements": self.n_placements,
            "trees": len(self._order),
            "per_view": dict(sorted(per_view.items())),
            "in_degree": {str(k): v for k, v in sorted(in_degree.items())},
            "leaves_min": min(sizes) if sizes else 0,
            "leaves_median": int(median(sizes)) if sizes else 0,
            "leaves_max": max(sizes) if sizes else 0,
            "out_of_order_inserts": self._out_of_order,
            "reindexes": self._reindexes,
        }
