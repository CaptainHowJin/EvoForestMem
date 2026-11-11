from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

from evoforestmem.forest.btree import (
    rebuild_dirty_queues_from_flags,
    validate_tree_structure,
)
from evoforestmem.forest.tree_types import MemTree, MemTreeNode, TreeCard

from .types import PolyTree, split_tree_id

DEFAULT_ROOT = Path("data/forest")


def _safe_name(tree_id: str) -> str:

    return quote(tree_id, safe="")


def node_to_dict(node: MemTreeNode) -> dict[str, Any]:
    d: dict[str, Any] = {
        "node_id": node.node_id,
        "level": node.level,
        "ts": [node.time_start, node.time_end],
        "children": node.child_ids,
        "item_count": node.item_count,
    }
    if node.parent_id is not None:
        d["parent_id"] = node.parent_id
    if node.summary:
        d["summary"] = node.summary
    if node.summary_dirty:
        d["summary_dirty"] = True
    if node.prev_leaf_id is not None:
        d["prev"] = node.prev_leaf_id
    if node.next_leaf_id is not None:
        d["next"] = node.next_leaf_id
    return d


def node_from_dict(d: dict[str, Any], tree_id: str) -> MemTreeNode:
    ts = d.get("ts") or [0.0, 0.0]
    return MemTreeNode(
        node_id=str(d["node_id"]),
        tree_id=tree_id,
        level=int(d.get("level", 0)),
        time_start=float(ts[0]),
        time_end=float(ts[1]),
        summary=str(d.get("summary", "")),
        summary_dirty=bool(d.get("summary_dirty", False)),
        child_ids=[str(c) for c in (d.get("children") or [])],
        item_count=int(d.get("item_count", 0)),
        parent_id=str(d["parent_id"]) if d.get("parent_id") is not None else None,
        prev_leaf_id=str(d["prev"]) if d.get("prev") is not None else None,
        next_leaf_id=str(d["next"]) if d.get("next") is not None else None,
    )


def tree_to_dict(tree: PolyTree) -> dict[str, Any]:
    d: dict[str, Any] = {
        "tree_id": tree.tree_id,
        "view": tree.view,
        "key": tree.key,
        "label": tree.label,
        "k": tree.k,
        "root_node_id": tree.root_node_id,
        "version": tree.version,
        "created_at": tree.created_at,
        "updated_at": tree.updated_at,
        "lifecycle_state": tree.lifecycle_state,
        "summaries_built": tree.summaries_built,
        "append_sealed": tree.append_sealed,
        "next_index_by_level": {str(lv): n for lv, n in sorted(tree.next_index_by_level.items())},

        "payload_ids_ordered": tree.payload_ids_ordered,
        "nodes": [node_to_dict(n) for n in tree.nodes.values()],
    }
    if tree.centroid is not None:
        d["centroid"] = tree.centroid
    if tree.deactivated:
        d["deactivated"] = True
        d["replacement_tree_ids"] = tree.replacement_tree_ids
    return d


def tree_from_dict(d: dict[str, Any]) -> PolyTree:
    tree_id = str(d["tree_id"])
    view = str(d.get("view") or split_tree_id(tree_id)[0])
    nodes = {}
    for nd in d.get("nodes") or []:
        node = node_from_dict(nd, tree_id)
        nodes[node.node_id] = node
    leaf_by_payload = {
        node.child_ids[0]: node.node_id
        for node in nodes.values()
        if node.level == 0 and node.child_ids
    }
    tree = MemTree(
        tree_id=tree_id,
        view=view,
        key=str(d.get("key") or split_tree_id(tree_id)[1]),
        label=str(d.get("label", "")),
        k=int(d.get("k", 16)),
        root_node_id=str(d.get("root_node_id", "")),
        nodes=nodes,
        payload_ids_ordered=[str(p) for p in (d.get("payload_ids_ordered") or [])],
        leaf_by_payload=leaf_by_payload,
        centroid=d.get("centroid"),
        version=int(d.get("version", 1)),
        lifecycle_state=str(d.get("lifecycle_state", "active")),
        deactivated=bool(d.get("deactivated", False)),
        replacement_tree_ids=[str(x) for x in (d.get("replacement_tree_ids") or [])],
        created_at=float(d.get("created_at", 0.0)),
        updated_at=float(d.get("updated_at", 0.0)),
        summaries_built=bool(d.get("summaries_built", False)),
        append_sealed=bool(d.get("append_sealed", False)),
        next_index_by_level={
            int(lv): int(n) for lv, n in (d.get("next_index_by_level") or {}).items()
        },
    )
    rebuild_dirty_queues_from_flags(tree)
    return tree


class TreeStore:


    def __init__(self, batch: str, root: str | Path = DEFAULT_ROOT) -> None:
        self.batch = batch
        self.root = Path(root) / batch
        self._trees: dict[str, PolyTree] = {}
        self._dirty_tree_ids: set[str] = set()


    def path_for(self, tree: PolyTree) -> Path:
        return self.root / tree.view / f"{_safe_name(tree.tree_id)}.json"

    def register(self, tree: PolyTree) -> PolyTree:

        self._trees[tree.tree_id] = tree
        self._dirty_tree_ids.add(tree.tree_id)
        return tree

    def mark_dirty(self, tree_ids: Iterable[str]) -> None:
        self._dirty_tree_ids.update(tree_id for tree_id in tree_ids if tree_id in self._trees)

    def unregister(self, tree_id: str, *, delete_file: bool = False) -> bool:
        tree = self._trees.pop(tree_id, None)
        self._dirty_tree_ids.discard(tree_id)
        if tree is None:
            return False
        if delete_file:
            try:
                self.path_for(tree).unlink()
            except FileNotFoundError:
                pass
        return True

    @property
    def dirty_tree_ids(self) -> frozenset[str]:
        return frozenset(self._dirty_tree_ids)

    def _follow_replacement(self, tree: PolyTree) -> PolyTree:

        seen: set[str] = set()
        current = tree
        while current.deactivated and current.replacement_tree_ids and current.tree_id not in seen:
            seen.add(current.tree_id)
            nxt_id = current.replacement_tree_ids[0]
            nxt = self._trees.get(nxt_id)
            if nxt is None:
                view, _ = split_tree_id(nxt_id)
                path = self.root / view / f"{_safe_name(nxt_id)}.json"
                if not path.exists():
                    break
                nxt = self._load_path(path, validate=False)
            if nxt is None:
                break
            current = nxt
        return current

    def get(self, tree_id: str, *, follow_replacement: bool = False) -> PolyTree | None:
        tree = self._trees.get(tree_id)
        if tree is None:
            return None
        if follow_replacement:
            return self._follow_replacement(tree)
        return tree

    def __len__(self) -> int:
        return len(self._trees)

    def __contains__(self, tree_id: object) -> bool:
        return tree_id in self._trees

    def __iter__(self) -> Iterator[PolyTree]:
        return iter(self._trees.values())

    @property
    def trees(self) -> dict[str, PolyTree]:
        return self._trees

    def by_view(self, view: str) -> list[PolyTree]:
        return [t for t in self._trees.values() if t.view == view]

    def views(self) -> list[str]:
        return sorted({t.view for t in self._trees.values()})


    def save_tree(self, tree: PolyTree) -> Path:
        path = self.path_for(tree)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(tree_to_dict(tree), ensure_ascii=False, separators=(",", ":"))
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
        self._trees[tree.tree_id] = tree
        self._dirty_tree_ids.discard(tree.tree_id)
        return path

    def save_dirty(self, tree_ids: Iterable[str] | None = None) -> int:

        wanted = set(self._dirty_tree_ids if tree_ids is None else tree_ids)
        saved = 0
        for tree_id in sorted(wanted):
            tree = self._trees.get(tree_id)
            if tree is None:
                continue
            self.save_tree(tree)
            saved += 1
        return saved

    def save_all(self) -> int:
        for tree in self._trees.values():
            self.save_tree(tree)
        return len(self._trees)

    def load_tree(self, tree_id: str, *, validate: bool = True,
                  follow_replacement: bool = False) -> PolyTree | None:
        cached = self._trees.get(tree_id)
        if cached is None:
            view, _ = split_tree_id(tree_id)
            path = self.root / view / f"{_safe_name(tree_id)}.json"
            if not path.exists():
                return None
            cached = self._load_path(path, validate=validate)
        if cached is not None and follow_replacement:
            return self._follow_replacement(cached)
        return cached

    def load_all(self, *, validate: bool = True) -> dict[str, PolyTree]:

        if not self.root.exists():
            return self._trees
        for view_dir in sorted(self.root.iterdir()):
            if not view_dir.is_dir():
                continue
            for path in sorted(view_dir.glob("*.json")):
                self._load_path(path, validate=validate)
        return self._trees

    def _load_path(self, path: Path, *, validate: bool) -> PolyTree:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to read tree file {path}: {exc}") from exc
        tree = tree_from_dict(data)
        if validate:


            validate_tree_structure(tree)
        self._trees[tree.tree_id] = tree
        self._dirty_tree_ids.discard(tree.tree_id)
        return tree


    def cards(self, *, include_suppressed: bool = False) -> list[TreeCard]:

        out: list[TreeCard] = []
        for tree in self._trees.values():
            if not include_suppressed and tree.lifecycle_state == "suppressed":
                continue
            root = tree.nodes.get(tree.root_node_id)
            out.append(
                TreeCard(
                    tree_id=tree.tree_id,
                    view=tree.view,
                    label=tree.label,
                    time_start=root.time_start if root else 0.0,
                    time_end=root.time_end if root else 0.0,
                    root_summary=root.summary if root else "",
                    item_count=root.item_count if root else 0,
                )
            )
        return sorted(out, key=lambda c: c.tree_id)

    def stats(self) -> dict[str, Any]:
        per_view: dict[str, dict[str, int]] = {}
        for tree in self._trees.values():
            v = per_view.setdefault(tree.view, {"trees": 0, "leaves": 0, "internal": 0})
            v["trees"] += 1
            v["leaves"] += tree.leaf_count
            v["internal"] += sum(1 for n in tree.nodes.values() if n.level > 0)
        leaves = [t.leaf_count for t in self._trees.values()]
        return {
            "batch": self.batch,
            "trees": len(self._trees),
            "per_view": dict(sorted(per_view.items())),

            "leaves_min": min(leaves) if leaves else 0,
            "leaves_max": max(leaves) if leaves else 0,

            "summary_nodes": sum(
                1 for t in self._trees.values() for n in t.nodes.values() if n.level > 0
            ),
        }
