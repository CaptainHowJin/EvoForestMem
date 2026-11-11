from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from evoforestmem.log.message import Message
from evoforestmem.routers import Router, default_routers
from evoforestmem.forest import btree
from evoforestmem.forest.btree import (
    build_tree_from_payloads,
    insert_payload,
    rebuild_hierarchy_from_level0,
)

from .placement import PlacementMap
from .store import TreeStore
from .types import DEFAULT_FOREST_CONFIG, ForestConfig, PolyTree, new_tree, tree_id_for


@dataclass(frozen=True, slots=True)
class BuildStats:
    messages: int
    routes: int
    placements: int
    trees: int
    elapsed_sec: float
    per_view: dict[str, int] = field(default_factory=dict)
    unrouted: dict[str, int] = field(default_factory=dict)
    fastpath_hit: int = 0
    fastpath_walk: int = 0
    fastpath_walk_steps: int = 0

    @property
    def fastpath_hit_rate(self) -> float:
        total = self.fastpath_hit + self.fastpath_walk
        return self.fastpath_hit / total if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "routes": self.routes,
            "placements": self.placements,
            "trees": self.trees,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "per_view": dict(sorted(self.per_view.items())),
            "unrouted": dict(sorted(self.unrouted.items())),

            "fastpath": {
                "hit": self.fastpath_hit,
                "walk": self.fastpath_walk,
                "walk_steps": self.fastpath_walk_steps,
                "hit_rate": round(self.fastpath_hit_rate, 6),
            },
        }


class ForestBuilder:


    def __init__(
        self,
        *,
        store: TreeStore,
        placement: PlacementMap | None = None,
        routers: Sequence[Router] | None = None,
        config: ForestConfig = DEFAULT_FOREST_CONFIG,
        per_tree_k: Mapping[str, int] | None = None,
    ) -> None:
        self.store = store
        self.placement = placement if placement is not None else PlacementMap()
        self.config = config


        self.per_tree_k: dict[str, int] = dict(per_tree_k or {})
        self.routers: tuple[Router, ...] = (
            tuple(routers) if routers is not None else default_routers(config.enabled_views)
        )
        unknown = [r.view for r in self.routers if r.view not in config.enabled_views]
        if unknown:
            raise ValueError(
                f"routers supplied for views that are not enabled in the config: {unknown}; "
                f"enabled = {list(config.enabled_views)}"
            )
        self._messages = 0
        self._routes = 0
        self._per_view: dict[str, int] = {r.view: 0 for r in self.routers}
        self._unrouted: dict[str, int] = {r.view: 0 for r in self.routers}


        self._ts: dict[str, float] = {}
        self._elapsed = 0.0
        self.last_add_timings = {"route_s": 0.0, "insert_s": 0.0, "placement_s": 0.0}

        self.arrival_order = None
        self.last_insertions = []


    def tree_for(self, view: str, key: str, label: str | None = None) -> PolyTree:

        tree_id = tree_id_for(view, key)
        tree = self.store.get(tree_id, follow_replacement=True)
        if tree is None:
            k = self.per_tree_k.get(tree_id, self.config.k_for(view))
            tree = new_tree(
                view, key, k=k, label=label or key,
                seal_on_append=self.config.seal_on_append_for(view),
            )
            self.store.register(tree)
        return tree

    def add(self, msg: Message) -> int:

        timings = {"route_s": 0.0, "insert_s": 0.0, "placement_s": 0.0}
        self.last_insertions = []
        self._ts[msg.msg_id] = msg.ts
        touched = 0
        for router in self.routers:
            started = time.perf_counter()
            keys = router.route(msg)
            timings["route_s"] += time.perf_counter() - started
            if not keys:
                self._unrouted[router.view] += 1
                continue
            for key in keys:
                started = time.perf_counter()
                tree = self.tree_for(router.view, key, router.label_for(key))
                placement_elapsed = 0.0

                def on_place(*args, **kwargs):
                    nonlocal placement_elapsed
                    placement_started = time.perf_counter()
                    try:
                        return self.placement.on_place(*args, **kwargs)
                    finally:
                        placement_elapsed += time.perf_counter() - placement_started


                take_backfill = getattr(router, "take_backfill", None)
                backfill = take_backfill(key) if take_backfill is not None else ()
                for member_id, member_ts in (*backfill, (msg.msg_id, msg.ts)):
                    leaf = insert_payload(tree, member_id, member_ts, on_place=on_place,
                                          arrival_order=self.arrival_order)
                    self.last_insertions.append((tree.tree_id, leaf))
                elapsed = time.perf_counter() - started
                timings["placement_s"] += placement_elapsed
                timings["insert_s"] += max(0.0, elapsed - placement_elapsed)
                self._per_view[router.view] += len(backfill) + 1
                touched += len(backfill) + 1
        self._messages += 1
        self._routes += touched
        self.last_add_timings = timings
        return touched

    def add_all(self, messages: Iterable[Message]) -> BuildStats:
        btree.reset_fastpath_stats()
        t0 = time.perf_counter()
        for msg in messages:
            self.add(msg)
        self._elapsed += time.perf_counter() - t0
        return self.stats()

    def stats(self) -> BuildStats:


        return BuildStats(
            messages=self._messages,
            routes=self._routes,
            placements=self.placement.n_placements,
            trees=len(self.store),
            elapsed_sec=self._elapsed,
            per_view=dict(self._per_view),
            unrouted={v: n for v, n in self._unrouted.items() if n},
            fastpath_hit=btree.FASTPATH_STATS["hit"],
            fastpath_walk=btree.FASTPATH_STATS["walk"],
            fastpath_walk_steps=btree.FASTPATH_STATS["walk_steps"],
        )


    def repack(self, ts_of: "Callable[[str], float] | None" = None) -> dict[str, Any]:

        summarised = [t.tree_id for t in self.store
                      if getattr(t, "summaries_built", False)
                      or any(getattr(n, "summary", "") for n in t.nodes.values())]
        if summarised:
            raise ValueError(
                f"repack: {len(summarised)} tree(s) carry generated summaries "
                f"({', '.join(sorted(summarised)[:3])}...); a rebuild changes which "
                f"leaves each node covers, so those summaries would describe the wrong "
                f"messages. Repack BEFORE flushing summaries, not after."
            )
        resolve = ts_of if ts_of is not None else self._ts.get
        t0 = time.perf_counter()
        before = self._node_counts()
        for tree in list(self.store):
            payloads = []
            for msg_id in tree.payload_ids_ordered:
                ts = resolve(msg_id)
                if ts is None:
                    raise ValueError(
                        f"repack: no timestamp for {msg_id!r} in {tree.tree_id}. This "
                        f"builder did not ingest it; pass ts_of=log.get(...).ts"
                    )
                payloads.append((msg_id, float(ts), float(ts)))
            rebuilt, _requests = build_tree_from_payloads(
                tree.tree_id, tree.view, tree.key, tree.label,
                self.config.k_for(tree.view), payloads,
                append_sealed=tree.append_sealed,
            )
            if rebuilt.payload_ids_ordered != tree.payload_ids_ordered:
                raise ValueError(
                    f"repack changed the leaf order of {tree.tree_id}; the leaf set and "
                    f"its order are the one thing a rebuild must preserve"
                )
            rebuilt.created_at = tree.created_at
            self.store.register(rebuilt)
            self.placement.index_tree(rebuilt)
        after = self._node_counts()
        return {
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "internal_nodes_before": before["internal"],
            "internal_nodes_after": after["internal"],
            "max_depth_before": before["max_depth"],
            "max_depth_after": after["max_depth"],
        }

    def regroup_channel_by_thread(
        self,
        log,
        *,
        min_thread: int | None = None,
        allow_summarised: bool = False,
    ) -> dict[str, Any]:

        from .threads import reply_roots, thread_groups

        channel_trees = [tree for tree in self.store if tree.view == "channel"]
        summarised = [tree.tree_id for tree in channel_trees if tree.summaries_built or any(
            node.summary for node in tree.nodes.values() if node.level > 0
        )]
        if summarised and not allow_summarised:
            raise ValueError(
                f"regroup would discard summaries from {len(summarised)} channel tree(s); "
                "pass allow_summarised=True only on an isolated copied batch"
            )

        threshold = min_thread or self.config.min_thread_for("channel")
        replies = {message.msg_id: message.reply_to for message in log if message.reply_to}
        started = time.perf_counter()
        before = self._node_counts()
        reply_edges = same_node_edges = pure_nodes = level1_nodes = 0
        changed = 0

        for tree in channel_trees:
            payloads = tree.payload_ids_ordered
            groups = thread_groups(payloads, replies, k=tree.k, min_thread=threshold)
            if not groups or not any(replies.get(msg_id) in tree.leaf_by_payload for msg_id in payloads):
                continue
            old_l0 = {
                node_id: (node.parent_id, node.prev_leaf_id, node.next_leaf_id,
                          node.time_start, node.time_end, tuple(node.child_ids))
                for node_id, node in tree.nodes.items() if node.level == 0
            }
            rebuilt, _requests = rebuild_hierarchy_from_level0(tree, level1_groups=groups)
            new_l0 = {
                node_id: (node.parent_id, node.prev_leaf_id, node.next_leaf_id,
                          node.time_start, node.time_end, tuple(node.child_ids))
                for node_id, node in rebuilt.nodes.items() if node.level == 0
            }

            if {k: v[1:] for k, v in old_l0.items()} != {k: v[1:] for k, v in new_l0.items()}:
                raise ValueError(f"regroup changed L0 data in {tree.tree_id}")
            if rebuilt.payload_ids_ordered != tree.payload_ids_ordered:
                raise ValueError(f"regroup changed leaf order in {tree.tree_id}")
            for position, msg_id in enumerate(rebuilt.payload_ids_ordered):
                placements = [p for p in self.placement.placements_of(msg_id)
                              if p.tree_id == rebuilt.tree_id]
                if len(placements) != 1 or placements[0].leaf_pos != position or \
                        placements[0].leaf_node_id != rebuilt.leaf_by_payload[msg_id]:
                    raise ValueError(f"regroup would change placement for {msg_id}")

            roots = reply_roots(payloads, replies)
            for group in groups:
                level1_nodes += 1
                if len({roots[payloads[index]] for index in group}) == 1:
                    pure_nodes += 1
            parent_by_payload = {
                msg_id: rebuilt.nodes[rebuilt.leaf_by_payload[msg_id]].parent_id
                for msg_id in payloads
            }
            for child in payloads:
                parent = replies.get(child)
                if parent not in tree.leaf_by_payload:
                    continue
                reply_edges += 1
                if parent_by_payload[child] == parent_by_payload[parent]:
                    same_node_edges += 1

            rebuilt.created_at = tree.created_at
            self.store.register(rebuilt)
            changed += 1

        after = self._node_counts()
        return {
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "channel_trees": changed,
            "min_thread": threshold,
            "internal_nodes_before": before["internal"],
            "internal_nodes_after": after["internal"],
            "max_depth_before": before["max_depth"],
            "max_depth_after": after["max_depth"],
            "level1_nodes": level1_nodes,
            "reply_edges": reply_edges,
            "same_l1_reply_edges": same_node_edges,
            "same_l1_reply_rate": same_node_edges / reply_edges if reply_edges else 0.0,
            "thread_pure_l1_nodes": pure_nodes,
            "thread_pure_l1_rate": pure_nodes / level1_nodes if level1_nodes else 0.0,
        }

    def _node_counts(self) -> dict[str, int]:
        internal = 0
        max_depth = 0
        for tree in self.store:
            internal += sum(1 for n in tree.nodes.values() if n.level > 0)
            root = tree.nodes.get(tree.root_node_id)
            if root is not None:
                max_depth = max(max_depth, root.level)
        return {"internal": internal, "max_depth": max_depth}


    def manifest(self) -> dict[str, Any]:

        stats = self.stats()
        nodes_by_level: dict[int, int] = {}
        depths: dict[int, int] = {}
        per_view: dict[str, dict[str, Any]] = {}
        for tree in self.store:
            root = tree.nodes.get(tree.root_node_id)
            depth = root.level if root is not None else 0
            depths[depth] = depths.get(depth, 0) + 1
            for node in tree.nodes.values():
                nodes_by_level[node.level] = nodes_by_level.get(node.level, 0) + 1
            v = per_view.setdefault(
                tree.view, {"trees": 0, "leaves": 0, "internal": 0, "k": self.config.k_for(tree.view),
                            "_sizes": []}
            )
            v["trees"] += 1
            v["leaves"] += tree.leaf_count
            v["internal"] += sum(1 for n in tree.nodes.values() if n.level > 0)
            v["_sizes"].append(tree.leaf_count)
        for v in per_view.values():
            sizes = sorted(v.pop("_sizes"))
            v["leaves_min"] = sizes[0] if sizes else 0
            v["leaves_median"] = int(median(sizes)) if sizes else 0
            v["leaves_max"] = sizes[-1] if sizes else 0

        internal_total = sum(n for lv, n in nodes_by_level.items() if lv > 0)
        return {
            "build": stats.to_dict(),
            "routers": {r.view: r.stats() for r in self.routers},
            "trees": len(self.store),
            "nodes_by_level": {str(lv): n for lv, n in sorted(nodes_by_level.items())},
            "depth_distribution": {str(d): n for d, n in sorted(depths.items())},

            "summary_nodes": internal_total,
            "per_view": dict(sorted(per_view.items())),
            "placement": self.placement.stats(),
        }
