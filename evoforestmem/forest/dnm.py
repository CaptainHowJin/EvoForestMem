from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ._persist import atomic_write

SCHEMA_VERSION = "evoforestmem.dnm.v1"
StateKey = tuple[str, str, str]
VectorLookup = Callable[[str], np.ndarray]


def _hash_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def cosine_drift(current: np.ndarray, base: np.ndarray) -> float:
    current_norm = float(np.linalg.norm(current))
    base_norm = float(np.linalg.norm(base))
    if current_norm <= 1e-12 and base_norm <= 1e-12:
        return 0.0
    if current_norm <= 1e-12 or base_norm <= 1e-12:
        return 1.0
    similarity = float(np.dot(current, base) / (current_norm * base_norm))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def source_changes(current_ids, base_ids):

    current, base = set(current_ids), set(base_ids)
    added = [mid for mid in current_ids if mid not in base]
    removed = [mid for mid in base_ids if mid not in current]
    return added, removed, (len(added) + len(removed)) / max(1, len(base_ids))


@dataclass(frozen=True, slots=True)
class DnmConfig:


    mode: str = "shadow"
    drift_delta: float = 0.05
    churn_delta: float = 0.05
    drift_compact: float = 0.10
    churn_compact: float = 0.20
    max_deltas: int = 4
    max_delta_bytes: int = 4096
    thresholds: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    atol: float = 1e-6
    rtol: float = 1e-5

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "apply"}:
            raise ValueError("DNM mode must be 'shadow' or 'apply'")
        if self.drift_compact < self.drift_delta:
            raise ValueError("drift_compact must be >= drift_delta")
        if self.churn_compact < self.churn_delta:
            raise ValueError("churn_compact must be >= churn_delta")

    def threshold(self, name: str, *, view: str, level: int) -> float:
        layer = self.thresholds.get(f"{view}/L{level}", {})
        return float(layer.get(name, getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "drift_delta": self.drift_delta,
            "churn_delta": self.churn_delta,
            "drift_compact": self.drift_compact,
            "churn_compact": self.churn_compact,
            "max_deltas": self.max_deltas,
            "max_delta_bytes": self.max_delta_bytes,
            "thresholds": {key: dict(value) for key, value in self.thresholds.items()},
            "atol": self.atol,
            "rtol": self.rtol,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DnmConfig":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class DeltaSummary:
    delta_id: str
    evidence_msg_ids: tuple[str, ...]
    summary: str | None
    generation: int
    effective_from: float
    effective_to: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "evidence_msg_ids": list(self.evidence_msg_ids),
            "summary": self.summary,
            "generation": self.generation,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeltaSummary":
        return cls(
            delta_id=str(value["delta_id"]),
            evidence_msg_ids=tuple(str(item) for item in value.get("evidence_msg_ids", ())),
            summary=None if value.get("summary") is None else str(value["summary"]),
            generation=int(value["generation"]),
            effective_from=float(value.get("effective_from", 0.0)),
            effective_to=(None if value.get("effective_to") is None
                          else float(value["effective_to"])),
        )


@dataclass(frozen=True, slots=True)
class SupersedeAnnotation:
    old_msg_id: str
    new_msg_id: str
    relation: str
    actor_scope: str | None
    effective_from: float
    valid_to: float | None
    relation_source: str
    generation: int

    def render(self) -> str:
        return f"{self.old_msg_id} {self.relation} by {self.new_msg_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_msg_id": self.old_msg_id,
            "new_msg_id": self.new_msg_id,
            "relation": self.relation,
            "actor_scope": self.actor_scope,
            "effective_from": self.effective_from,
            "valid_to": self.valid_to,
            "relation_source": self.relation_source,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SupersedeAnnotation":
        return cls(
            old_msg_id=str(value["old_msg_id"]),
            new_msg_id=str(value["new_msg_id"]),
            relation=str(value.get("relation") or "supersede"),
            actor_scope=(None if value.get("actor_scope") is None
                         else str(value["actor_scope"])),
            effective_from=float(value.get("effective_from", 0.0)),
            valid_to=None if value.get("valid_to") is None else float(value["valid_to"]),
            relation_source=str(value.get("relation_source") or "validity"),
            generation=int(value.get("generation", 0)),
        )


@dataclass(frozen=True, slots=True)
class DnmSummaryRequest:


    view: str
    tree_id: str
    node_id: str
    level: int
    previous_summary: str
    tree_label: str
    time_start: float
    time_end: float
    evidence_msg_ids: tuple[str, ...]
    invalidated_ids: tuple[str, ...]
    input_text: str
    generation: int
    is_root: bool = False
    channel_thread: bool = False


@dataclass(slots=True)
class NodeMaintenanceState:
    view: str
    tree_id: str
    node_id: str
    level: int
    parent_id: str | None
    child_semantic_hash: str
    l0_count: int
    l0_sum_f32: np.ndarray
    centroid_f32: np.ndarray
    structural_generation: int = 0
    current_msg_ids: tuple[str, ...] = ()
    base_summary: str = ""
    base_vector_f32: np.ndarray | None = None
    base_l0_count: int = 0
    base_centroid_f32: np.ndarray | None = None
    base_semantic_hash: str = ""
    base_generation: int = 0
    base_msg_ids: tuple[str, ...] = ()
    delta_ids: list[str] = field(default_factory=list)
    invalidated_ids: list[str] = field(default_factory=list)
    delta_summaries: list[DeltaSummary] = field(default_factory=list)
    annotations: list[SupersedeAnnotation] = field(default_factory=list)
    delta_bytes: int = 0
    centroid_drift: float = 0.0
    churn_ratio: float = 0.0
    invalidation_count: int = 0
    needs_delta: bool = False
    needs_compaction: bool = False
    dirty_state: bool = True
    dirty_summary_row: bool = False
    dirty_tree: bool = False
    updated_at: float = 0.0

    @property
    def key(self) -> StateKey:
        return self.view, self.tree_id, self.node_id

    def visible_header(
        self, *, as_of: float | None = None,
        visible_msg_ids: set[str] | None = None,
    ) -> str:
        parts = [self.base_summary.strip()]
        parts.extend(
            delta.summary.strip()
            for delta in sorted(self.delta_summaries, key=lambda item: item.generation)
            if delta.summary and (as_of is None or delta.effective_from <= as_of)
            and (visible_msg_ids is None or set(delta.evidence_msg_ids) <= visible_msg_ids)
            and (delta.effective_to is None or as_of is None or as_of < delta.effective_to)
        )
        parts.extend(
            annotation.render()
            for annotation in sorted(self.annotations, key=lambda item: item.generation)
            if as_of is None or annotation.effective_from <= as_of
            if visible_msg_ids is None or {
                annotation.old_msg_id, annotation.new_msg_id} <= visible_msg_ids
        )
        return " | ".join(part for part in parts if part)


class DnmController:


    def __init__(
        self,
        *,
        vector_of: VectorLookup,
        dim: int,
        model: str = "",
        config: DnmConfig = DnmConfig(),
    ) -> None:
        if dim < 1:
            raise ValueError("DNM vector dimension must be positive")
        self.vector_of = vector_of
        self.dim = int(dim)
        self.model = model
        self.config = config
        self.states: dict[StateKey, NodeMaintenanceState] = {}
        self.generation = 0

    def state_for(self, view: str, tree_id: str, node_id: str) -> NodeMaintenanceState | None:
        return self.states.get((view, tree_id, node_id))

    def _leaf_value(self, tree: Any, node: Any) -> tuple[np.ndarray, tuple[str, ...], str]:
        msg_ids = tuple(str(item) for item in node.child_ids)
        total = np.zeros(self.dim, dtype=np.float64)
        for msg_id in msg_ids:
            vector = np.asarray(self.vector_of(msg_id), dtype=np.float32).reshape(-1)
            if vector.size != self.dim:
                raise ValueError(
                    f"DNM vector for {msg_id!r} has {vector.size} dims, expected {self.dim}"
                )
            total += vector.astype(np.float64)
        semantic_hash = _hash_parts((node.node_id, *msg_ids))
        return total, msg_ids, semantic_hash

    def _reconcile_node(
        self, tree: Any, node_id: str, memo: dict[str, tuple[np.ndarray, tuple[str, ...], str]],
        force: set[str],
    ) -> tuple[np.ndarray, tuple[str, ...], str]:
        cached = memo.get(node_id)
        if cached is not None:
            return cached
        node = tree.nodes[node_id]
        existing = self.states.get((str(tree.view), str(tree.tree_id), str(node.node_id)))
        if node.level > 0 and node_id not in force and existing is not None:
            value = (
                existing.l0_sum_f32.astype(np.float64),
                existing.current_msg_ids,
                existing.child_semantic_hash,
            )
            memo[node_id] = value
            return value
        if node.level == 0:
            value = self._leaf_value(tree, node)
            memo[node_id] = value
            return value
        total = np.zeros(self.dim, dtype=np.float64)
        msg_ids: list[str] = []
        child_hashes: list[str] = []
        for child_id in node.child_ids:
            child_sum, child_ids, child_hash = self._reconcile_node(
                tree, child_id, memo, force)
            total += child_sum
            msg_ids.extend(child_ids)
            child_hashes.extend((child_id, child_hash))
        semantic_hash = _hash_parts(child_hashes)
        current_ids = tuple(msg_ids)
        key = (str(tree.view), str(tree.tree_id), str(node.node_id))
        previous = self.states.get(key)
        total_f32 = total.astype(np.float32)
        if previous is not None and previous.child_semantic_hash == semantic_hash:
            changed = previous.l0_count != len(current_ids) or not np.allclose(
                previous.l0_sum_f32, total_f32,
                atol=self.config.atol, rtol=self.config.rtol,
            )
            if changed:
                raise ValueError(
                    f"DNM invariant violation for {tree.tree_id}/{node.node_id}: "
                    "semantic hash unchanged while sum/count changed"
                )
        centroid = _unit(total_f32)
        if previous is None:
            state = NodeMaintenanceState(
                view=str(tree.view), tree_id=str(tree.tree_id), node_id=str(node.node_id),
                level=int(node.level), parent_id=node.parent_id,
                child_semantic_hash=semantic_hash, l0_count=len(current_ids),
                l0_sum_f32=total_f32, centroid_f32=centroid,
                structural_generation=1, current_msg_ids=current_ids,
                base_summary=str(node.summary or ""),
                base_l0_count=len(current_ids) if node.summary else 0,
                base_centroid_f32=centroid.copy(),
                base_semantic_hash=semantic_hash if node.summary else "",
                base_generation=1 if node.summary else 0,
                base_msg_ids=current_ids if node.summary else (),
                updated_at=time.time(),
            )
            self.states[key] = state
        else:
            state = previous
            if state.child_semantic_hash != semantic_hash:
                state.structural_generation += 1
                state.dirty_state = True
                state.dirty_tree = True
            state.level = int(node.level)
            state.parent_id = node.parent_id
            state.child_semantic_hash = semantic_hash
            state.l0_count = len(current_ids)
            state.l0_sum_f32 = total_f32
            state.centroid_f32 = centroid
            state.current_msg_ids = current_ids
            state.updated_at = time.time()
        self._evaluate(state)
        memo[node_id] = total, current_ids, semantic_hash
        return memo[node_id]

    def _evaluate(self, state: NodeMaintenanceState) -> None:
        base_centroid = state.base_centroid_f32
        state.centroid_drift = (
            1.0 if base_centroid is None
            else cosine_drift(state.centroid_f32, base_centroid)
        )
        added, removed, churn = source_changes(state.current_msg_ids, state.base_msg_ids)
        state.delta_ids = added
        state.invalidated_ids = list(dict.fromkeys((*state.invalidated_ids, *removed)))
        state.invalidation_count = len(state.invalidated_ids)
        state.churn_ratio = churn
        drift_delta = self.config.threshold(
            "drift_delta", view=state.view, level=state.level)
        churn_delta = self.config.threshold(
            "churn_delta", view=state.view, level=state.level)
        drift_compact = self.config.threshold(
            "drift_compact", view=state.view, level=state.level)
        churn_compact = self.config.threshold(
            "churn_compact", view=state.view, level=state.level)
        covered = {
            msg_id for delta in state.delta_summaries for msg_id in delta.evidence_msg_ids
        }
        pending_delta = [msg_id for msg_id in state.delta_ids if msg_id not in covered]
        state.needs_delta = bool(pending_delta) and (
            state.centroid_drift > drift_delta or state.churn_ratio > churn_delta
        )
        state.needs_compaction = (
            state.invalidation_count > 0
            or state.centroid_drift > drift_compact
            or state.churn_ratio > churn_compact
            or len(state.delta_summaries) > self.config.max_deltas
            or state.delta_bytes > self.config.max_delta_bytes
            or not state.base_summary
        )

    def reconcile_tree(self, tree: Any, node_ids: Iterable[str] | None = None) -> set[StateKey]:

        targets = set(node_ids or ())
        if node_ids is None:
            targets = {
                node_id for level, ids in tree.dirty_internal_node_ids_by_level.items()
                if level > 0 for node_id in ids
            }
        if node_ids is None and not targets and tree.root_node_id:
            targets.add(tree.root_node_id)
        expanded = set(targets)
        for node_id in tuple(targets):
            node = tree.nodes.get(node_id)
            while node is not None and node.parent_id is not None:
                expanded.add(node.parent_id)
                node = tree.nodes.get(node.parent_id)
        memo: dict[str, tuple[np.ndarray, tuple[str, ...], str]] = {}
        changed: set[StateKey] = set()
        for node_id in sorted(
            expanded, key=lambda item: (tree.nodes[item].level, item)
            if item in tree.nodes else (-1, item),
        ):
            node = tree.nodes.get(node_id)
            if node is None or node.level == 0:
                continue
            before = self.states.get((tree.view, tree.tree_id, node_id))
            old_hash = before.child_semantic_hash if before is not None else None
            self._reconcile_node(tree, node_id, memo, expanded)
            state = self.states[(tree.view, tree.tree_id, node_id)]
            if old_hash != state.child_semantic_hash:
                changed.add(state.key)
        live = set(tree.nodes)
        for key in tuple(self.states):
            if key[1] == tree.tree_id and key[2] not in live:
                del self.states[key]
        if changed:
            self.generation += 1
        return changed

    def observe_supersessions(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        placement: Any,
        store: Any,
    ) -> set[StateKey]:
        affected: set[StateKey] = set()
        for event in events:
            old_id = str(event.get("target_msg_id") or "")
            new_id = str(event.get("trigger_msg_id") or "")
            if not old_id or not new_id:
                continue
            generation = self.generation + 1
            annotation = SupersedeAnnotation(
                old_msg_id=old_id, new_msg_id=new_id,
                relation=str(event.get("relationship") or "supersede"),
                actor_scope=(str(event["subject"]) if event.get("subject") else None),
                effective_from=float(event.get("effective_from") or 0.0),
                valid_to=None,
                relation_source=str(event.get("event_id") or event.get("_fmc_event_id") or "validity"),
                generation=generation,
            )
            for msg_id in (old_id, new_id):
                for row in placement.placements_of(msg_id):
                    tree = store.get(row.tree_id)
                    node = tree.nodes.get(row.leaf_node_id) if tree is not None else None
                    while node is not None and node.parent_id is not None:
                        node = tree.nodes.get(node.parent_id)
                        if node is None:
                            break
                        state = self.states.get((tree.view, tree.tree_id, node.node_id))
                        if state is None:
                            continue
                        identity = (annotation.old_msg_id, annotation.new_msg_id,
                                    annotation.relation_source)
                        if identity not in {
                            (item.old_msg_id, item.new_msg_id, item.relation_source)
                            for item in state.annotations
                        }:
                            state.annotations.append(annotation)
                        if old_id in state.current_msg_ids and old_id not in state.invalidated_ids:
                            state.invalidated_ids.append(old_id)
                        state.invalidation_count = len(state.invalidated_ids)
                        state.needs_compaction = True
                        state.dirty_state = True
                        state.dirty_tree = True
                        state.updated_at = time.time()
                        affected.add(state.key)
            self.generation = generation
        return affected

    def add_delta(
        self, key: StateKey, summary: str | None, *, evidence_msg_ids: Sequence[str],
        effective_from: float,
    ) -> DeltaSummary:
        state = self.states[key]
        evidence = tuple(dict.fromkeys(str(item) for item in evidence_msg_ids))
        generation = self.generation + 1
        delta = DeltaSummary(
            delta_id=_hash_parts((*key, str(generation), *evidence))[:24],
            evidence_msg_ids=evidence, summary=summary, generation=generation,
            effective_from=float(effective_from),
        )
        state.delta_summaries.append(delta)
        state.delta_bytes += len((summary or "").encode("utf-8"))
        state.dirty_state = True
        state.dirty_summary_row = True
        state.dirty_tree = True
        state.updated_at = time.time()
        self.generation = generation
        self._evaluate(state)
        return delta

    def delta_request(
        self, key: StateKey, *, text_of: Callable[[str], str], tree: Any | None = None,
    ) -> DnmSummaryRequest | None:
        state = self.states[key]
        if not state.needs_delta:
            return None
        covered = {
            msg_id for delta in state.delta_summaries for msg_id in delta.evidence_msg_ids
        }
        evidence = tuple(msg_id for msg_id in state.delta_ids if msg_id not in covered)
        if not evidence:
            return None
        node = tree.nodes.get(state.node_id) if tree is not None else None
        if state.level <= 1 or tree is None:
            sections = [text_of(msg_id) for msg_id in evidence]
        else:
            sections = []
            for child_id in node.child_ids if node is not None else ():
                child = self.state_for(state.view, state.tree_id, child_id)
                if child is None:
                    continue
                changed = [
                    delta.summary for delta in child.delta_summaries
                    if delta.summary and set(delta.evidence_msg_ids) & set(evidence)
                ]
                sections.extend(changed)
                sections.extend(annotation.render() for annotation in child.annotations)
            if not sections:
                sections = [text_of(msg_id) for msg_id in evidence]
        sections.extend(f"INVALIDATED: {msg_id}" for msg_id in state.invalidated_ids)
        return DnmSummaryRequest(
            view=state.view, tree_id=state.tree_id, node_id=state.node_id,
            level=state.level, previous_summary=state.visible_header(),
            tree_label=str(getattr(tree, "label", "") or "") if tree is not None else "",
            time_start=float(getattr(node, "time_start", 0.0) or 0.0)
            if tree is not None and node is not None else 0.0,
            time_end=float(getattr(node, "time_end", 0.0) or 0.0)
            if tree is not None and node is not None else 0.0,
            evidence_msg_ids=evidence,
            invalidated_ids=tuple(state.invalidated_ids),
            input_text="\n\n".join(sections), generation=self.generation + 1,
            is_root=bool(tree is not None and tree.root_node_id == state.node_id),
        )

    def compact(
        self, key: StateKey, *, summary: str, vector: np.ndarray | None = None,
        resolve_invalidations: bool = False,
    ) -> None:
        state = self.states[key]
        state.base_summary = str(summary)
        state.base_vector_f32 = None if vector is None else _unit(vector)
        state.base_l0_count = state.l0_count
        state.base_centroid_f32 = state.centroid_f32.copy()
        state.base_semantic_hash = state.child_semantic_hash
        state.base_msg_ids = state.current_msg_ids
        state.base_generation += 1
        state.delta_ids.clear()
        state.delta_summaries.clear()
        if resolve_invalidations:
            state.invalidated_ids.clear()
        state.delta_bytes = 0
        state.dirty_summary_row = True
        state.dirty_tree = True
        state.dirty_state = True
        state.updated_at = time.time()
        self.generation += 1
        self._evaluate(state)

    def tree_needs_paid_work(self, tree: Any) -> bool:
        candidates = [state for state in self.states.values() if state.tree_id == tree.tree_id]
        if not candidates:
            return any(tree.dirty_internal_node_ids_by_level.values()) or not tree.summaries_built
        return any(state.needs_delta or state.needs_compaction for state in candidates)

    def stats(self) -> dict[str, Any]:
        values = list(self.states.values())
        return {
            "mode": self.config.mode,
            "generation": self.generation,
            "nodes": len(values),
            "needs_delta": sum(state.needs_delta for state in values),
            "needs_compaction": sum(state.needs_compaction for state in values),
            "invalidations": sum(state.invalidation_count for state in values),
            "dirty_trees": sorted({state.tree_id for state in values if state.dirty_tree}),
        }

    def validate(self, *, available_msg_ids: set[str]) -> None:

        for state in self.states.values():
            foreign = set(state.current_msg_ids) - available_msg_ids
            if foreign:
                raise ValueError(
                    f"DNM checkpoint prefix mismatch at {state.tree_id}/{state.node_id}: "
                    f"{sorted(foreign)[:3]} not in L0"
                )
            if state.l0_count != len(state.current_msg_ids):
                raise ValueError(
                    f"DNM count mismatch at {state.tree_id}/{state.node_id}"
                )

    def _metadata(self, state: NodeMaintenanceState, row: int) -> dict[str, Any]:
        return {
            "view": state.view, "tree_id": state.tree_id, "node_id": state.node_id,
            "level": state.level, "parent_id": state.parent_id,
            "child_semantic_hash": state.child_semantic_hash,
            "l0_count": state.l0_count, "structural_generation": state.structural_generation,
            "current_msg_ids": list(state.current_msg_ids), "base_summary": state.base_summary,
            "base_l0_count": state.base_l0_count,
            "base_semantic_hash": state.base_semantic_hash,
            "base_generation": state.base_generation, "base_msg_ids": list(state.base_msg_ids),
            "delta_ids": list(state.delta_ids),
            "invalidated_ids": list(state.invalidated_ids),
            "delta_summaries": [item.to_dict() for item in state.delta_summaries],
            "annotations": [item.to_dict() for item in state.annotations],
            "delta_bytes": state.delta_bytes, "centroid_drift": state.centroid_drift,
            "churn_ratio": state.churn_ratio,
            "invalidation_count": state.invalidation_count,
            "needs_delta": state.needs_delta, "needs_compaction": state.needs_compaction,
            "dirty_state": False, "dirty_summary_row": state.dirty_summary_row,
            "dirty_tree": False, "updated_at": state.updated_at,
            "vector_row": row,
            "has_base_vector": state.base_vector_f32 is not None,
        }

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.states.values(), key=lambda state: state.key)
        vectors = np.zeros((len(ordered), 4, self.dim), dtype=np.float32)
        rows = []
        for row, state in enumerate(ordered):
            vectors[row, 0] = state.l0_sum_f32
            vectors[row, 1] = state.centroid_f32
            if state.base_centroid_f32 is not None:
                vectors[row, 2] = state.base_centroid_f32
            if state.base_vector_f32 is not None:
                vectors[row, 3] = state.base_vector_f32
            rows.append(self._metadata(state, row))
        state_payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
        state_digest = hashlib.sha256(state_payload).hexdigest()
        state_name = f"state-{self.generation:012d}-{state_digest[:12]}.jsonl"
        state_path = root / state_name
        if not state_path.exists():
            atomic_write(state_path, state_payload.decode("utf-8"))
        with tempfile.NamedTemporaryFile(dir=root, prefix=".vectors.", suffix=".tmp.npy",
                                         delete=False) as handle:
            np.save(handle, vectors)
            handle.flush()
            os.fsync(handle.fileno())
            vector_tmp = Path(handle.name)
        vector_digest = hashlib.sha256(vector_tmp.read_bytes()).hexdigest()
        vector_name = f"vectors-{self.generation:012d}-{vector_digest[:12]}.npy"
        vector_path = root / vector_name
        if vector_path.exists():
            vector_tmp.unlink()
        else:
            os.replace(vector_tmp, vector_path)
        checksums = {
            state_name: state_digest,
            vector_name: vector_digest,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION, "generation": self.generation,
            "model": self.model, "dim": self.dim, "row_count": len(ordered),
            "config": self.config.to_dict(), "checksums": checksums,
            "state_file": state_name, "vectors_file": vector_name,
        }
        manifest_text = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        manifest_digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        generation_manifest = root / (
            f"manifest-{self.generation:012d}-{manifest_digest[:12]}.json")
        if not generation_manifest.exists():
            atomic_write(generation_manifest, manifest_text)
        atomic_write(
            root / "manifest.json", manifest_text,
        )
        for state in ordered:
            state.dirty_state = False
            state.dirty_tree = False
        return generation_manifest

    @classmethod
    def load(
        cls, directory: str | Path, *, vector_of: VectorLookup,
        config: DnmConfig | None = None, manifest_name: str = "manifest.json",
    ) -> "DnmController":
        root = Path(directory)
        manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported DNM schema: {manifest.get('schema_version')}")
        for name, expected in manifest.get("checksums", {}).items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"DNM {name} checksum mismatch")
        vectors_file = str(manifest.get("vectors_file") or "vectors.npy")
        state_file = str(manifest.get("state_file") or "state.jsonl")
        vectors = np.load(root / vectors_file)
        dim = int(manifest["dim"])
        count = int(manifest["row_count"])
        if vectors.shape != (count, 4, dim):
            raise ValueError("DNM vector shape does not match manifest")
        rows = [json.loads(line) for line in
                (root / state_file).read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != count:
            raise ValueError("DNM state row count does not match manifest")
        controller = cls(
            vector_of=vector_of, dim=dim, model=str(manifest.get("model") or ""),
            config=config or DnmConfig.from_dict(manifest.get("config") or {}),
        )
        controller.generation = int(manifest.get("generation", 0))
        for row in rows:
            index = int(row["vector_row"])
            state = NodeMaintenanceState(
                view=str(row["view"]), tree_id=str(row["tree_id"]),
                node_id=str(row["node_id"]), level=int(row["level"]),
                parent_id=row.get("parent_id"),
                child_semantic_hash=str(row["child_semantic_hash"]),
                l0_count=int(row["l0_count"]), l0_sum_f32=vectors[index, 0].copy(),
                centroid_f32=vectors[index, 1].copy(),
                structural_generation=int(row.get("structural_generation", 0)),
                current_msg_ids=tuple(row.get("current_msg_ids", ())),
                base_summary=str(row.get("base_summary") or ""),
                base_vector_f32=(vectors[index, 3].copy()
                                 if row.get("has_base_vector", False) else None),
                base_l0_count=int(row.get("base_l0_count", 0)),
                base_centroid_f32=vectors[index, 2].copy(),
                base_semantic_hash=str(row.get("base_semantic_hash") or ""),
                base_generation=int(row.get("base_generation", 0)),
                base_msg_ids=tuple(row.get("base_msg_ids", ())),
                delta_ids=list(row.get("delta_ids", ())),
                invalidated_ids=list(row.get("invalidated_ids", ())),
                delta_summaries=[DeltaSummary.from_dict(item)
                                 for item in row.get("delta_summaries", ())],
                annotations=[SupersedeAnnotation.from_dict(item)
                             for item in row.get("annotations", ())],
                delta_bytes=int(row.get("delta_bytes", 0)),
                centroid_drift=float(row.get("centroid_drift", 0.0)),
                churn_ratio=float(row.get("churn_ratio", 0.0)),
                invalidation_count=int(row.get("invalidation_count", 0)),
                needs_delta=bool(row.get("needs_delta", False)),
                needs_compaction=bool(row.get("needs_compaction", False)),
                dirty_state=bool(row.get("dirty_state", False)),
                dirty_summary_row=bool(row.get("dirty_summary_row", False)),
                dirty_tree=bool(row.get("dirty_tree", False)),
                updated_at=float(row.get("updated_at", 0.0)),
            )
            controller.states[state.key] = state
        return controller


__all__ = [
    "DeltaSummary", "DnmConfig", "DnmController", "DnmSummaryRequest",
    "NodeMaintenanceState",
    "SCHEMA_VERSION", "SupersedeAnnotation", "cosine_drift",
]
