from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..log.message import Message
from .dnm import DnmController

SCHEMA_VERSION = "evoforestmem.fmc-shadow.v1"
_ACTIONS = frozenset({"mint", "admit", "reorganize", "refresh", "delta", "compact"})


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fsync_directory(path: Path) -> None:

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class FmcConfig:


    mode: str = "shadow"
    policy_version: str = "fmc-shadow-v1"
    code_revision: str = "unknown"
    environment_hash: str = "unspecified"

    def __post_init__(self) -> None:
        if self.mode != "shadow":
            raise ValueError("FMC currently supports shadow mode only")

    def to_dict(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "policy_version": self.policy_version,
            "code_revision": self.code_revision,
            "environment_hash": self.environment_hash,
        }

    def fingerprint(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FmcEvent:


    event_id: str
    epoch_id: int
    msg_id: str
    msg_seq: int
    committed_at: float
    action: str
    component: str
    input_msg_ids: tuple[str, ...]
    affected_trees: tuple[str, ...]
    candidate_scores: tuple[tuple[str, float], ...]
    thresholds: tuple[tuple[str, float], ...]
    attributes: tuple[tuple[str, str], ...]
    decision: str
    reason: str
    prior_generation: int
    output_generation: int
    config_hash: str

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError(f"unknown FMC action: {self.action}")
        if self.output_generation != self.prior_generation + 1:
            raise ValueError("FMC generation must advance by exactly one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "epoch_id": self.epoch_id,
            "msg_id": self.msg_id,
            "msg_seq": self.msg_seq,
            "committed_at": self.committed_at,
            "action": self.action,
            "component": self.component,
            "input_msg_ids": list(self.input_msg_ids),
            "affected_trees": list(self.affected_trees),
            "candidate_scores": dict(self.candidate_scores),
            "thresholds": dict(self.thresholds),
            "attributes": dict(self.attributes),
            "decision": self.decision,
            "reason": self.reason,
            "prior_generation": self.prior_generation,
            "output_generation": self.output_generation,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FmcEvent":
        return cls(
            event_id=str(value["event_id"]),
            epoch_id=int(value["epoch_id"]),
            msg_id=str(value.get("msg_id") or ""),
            msg_seq=int(value.get("msg_seq", -1)),
            committed_at=float(value.get("committed_at", 0.0)),
            action=str(value["action"]),
            component=str(value["component"]),
            input_msg_ids=tuple(str(x) for x in value.get("input_msg_ids", ())),
            affected_trees=tuple(str(x) for x in value.get("affected_trees", ())),
            candidate_scores=tuple(sorted(
                (str(key), float(number))
                for key, number in (value.get("candidate_scores") or {}).items()
            )),
            thresholds=tuple(sorted(
                (str(key), float(number))
                for key, number in (value.get("thresholds") or {}).items()
            )),
            attributes=tuple(sorted(
                (str(key), str(item))
                for key, item in (value.get("attributes") or {}).items()
            )),
            decision=str(value.get("decision") or "observe"),
            reason=str(value.get("reason") or ""),
            prior_generation=int(value["prior_generation"]),
            output_generation=int(value["output_generation"]),
            config_hash=str(value["config_hash"]),
        )


class ForestMaintenanceController:


    def __init__(
        self, *, config: FmcConfig = FmcConfig(), dnm: DnmController | None = None,
    ) -> None:
        self.config = config
        self._config_hash = config.fingerprint()
        self._events: list[FmcEvent] = []
        self._by_id: dict[str, FmcEvent] = {}
        self._generation = 0
        self._persisted_count = 0
        self._segments: list[dict[str, Any]] = []
        self.dnm = dnm

    @property
    def events(self) -> tuple[FmcEvent, ...]:
        return tuple(self._events)

    @property
    def generation(self) -> int:
        return self._generation

    def fingerprint(self) -> str:
        return _sha256([event.to_dict() for event in self._events])

    def stats(self) -> dict[str, Any]:
        by_action: dict[str, int] = {}
        by_component: dict[str, int] = {}
        for event in self._events:
            by_action[event.action] = by_action.get(event.action, 0) + 1
            by_component[event.component] = by_component.get(event.component, 0) + 1
        result = {
            "mode": self.config.mode,
            "generation": self._generation,
            "events": len(self._events),
            "by_action": dict(sorted(by_action.items())),
            "by_component": dict(sorted(by_component.items())),
            "fingerprint": self.fingerprint(),
            "config_hash": self._config_hash,
        }
        if self.dnm is not None:
            result["dnm"] = self.dnm.stats()
        return result

    @staticmethod
    def _semantic_id(payload: Mapping[str, Any]) -> str:
        return _sha256(payload)[:24]

    def _record(
        self,
        *,
        epoch_id: int,
        msg_id: str,
        msg_seq: int,
        committed_at: float,
        action: str,
        component: str,
        input_msg_ids: Iterable[str] = (),
        affected_trees: Iterable[str] = (),
        candidate_scores: Mapping[str, float] | None = None,
        thresholds: Mapping[str, float] | None = None,
        attributes: Mapping[str, Any] | None = None,
        identity_attributes: Mapping[str, Any] | None = None,
        decision: str = "observe",
        reason: str = "shadow_only",
        available_msg_ids: set[str],
    ) -> bool:
        refs = tuple(sorted({str(value) for value in input_msg_ids if value}))
        future = set(refs) - available_msg_ids
        if future:
            raise ValueError(
                "FMC no-lookahead violation: event references unavailable messages "
                f"{sorted(future)[:3]}"
            )
        trees = tuple(sorted({str(value) for value in affected_trees if value}))
        scores = tuple(sorted(
            (str(key), float(value)) for key, value in (candidate_scores or {}).items()
        ))
        gates = tuple(sorted(
            (str(key), float(value)) for key, value in (thresholds or {}).items()
        ))
        facts = tuple(sorted(
            (str(key), str(value)) for key, value in (attributes or {}).items()
        ))
        identity_facts = tuple(sorted(
            (str(key), str(value))
            for key, value in (identity_attributes or attributes or {}).items()
        ))


        identity = {
            "epoch_id": int(epoch_id),
            "msg_id": str(msg_id),
            "msg_seq": int(msg_seq),
            "action": action,
            "component": component,
            "input_msg_ids": refs,
            "attributes": identity_facts,
            "config_hash": self._config_hash,
        }
        event_id = self._semantic_id(identity)
        if event_id in self._by_id:
            return False
        prior = self._generation
        event = FmcEvent(
            event_id=event_id,
            epoch_id=int(epoch_id),
            msg_id=str(msg_id),
            msg_seq=int(msg_seq),
            committed_at=float(committed_at),
            action=action,
            component=component,
            input_msg_ids=refs,
            affected_trees=trees,
            candidate_scores=scores,
            thresholds=gates,
            attributes=facts,
            decision=decision,
            reason=reason,
            prior_generation=prior,
            output_generation=prior + 1,
            config_hash=self._config_hash,
        )
        self._events.append(event)
        self._by_id[event_id] = event
        self._generation += 1
        return True

    def observe_message(
        self,
        *,
        epoch_id: int,
        message: Message,
        placement: Any,
        minted_tree_ids: Iterable[str],
        available_msg_ids: set[str],
        store: Any | None = None,
    ) -> int:

        before = self._generation
        for tree_id in sorted(set(minted_tree_ids)):
            self._record(
                epoch_id=epoch_id,
                msg_id=message.msg_id,
                msg_seq=message.seq,
                committed_at=message.ts,
                action="mint",
                component="view_tree",
                input_msg_ids=(message.msg_id,),
                affected_trees=(tree_id,),
                attributes={"tree_id": tree_id},
                decision="observe",
                reason="first_routed_key",
                available_msg_ids=available_msg_ids,
            )
        placements = placement.placements_of(message.msg_id)
        self._record(
            epoch_id=epoch_id,
            msg_id=message.msg_id,
            msg_seq=message.seq,
            committed_at=message.ts,
            action="admit",
            component="forest_ingest",
            input_msg_ids=(message.msg_id,),
            affected_trees=(row.tree_id for row in placements),
            candidate_scores={"placements": float(len(placements))},
            decision="observe",
            reason="message_committed_and_routed",
            available_msg_ids=available_msg_ids,
        )
        if self.dnm is not None and store is not None:
            for tree_id in sorted({row.tree_id for row in placements}):
                tree = store.get(tree_id)
                if tree is not None:
                    paths: set[str] = set()
                    for row in placements:
                        if row.tree_id != tree_id:
                            continue
                        node = tree.nodes.get(row.leaf_node_id)
                        while node is not None and node.parent_id is not None:
                            paths.add(node.parent_id)
                            node = tree.nodes.get(node.parent_id)
                    changed = self.dnm.reconcile_tree(tree, node_ids=paths)
                    for key in sorted(changed):
                        state = self.dnm.states[key]
                        action = "compact" if state.needs_compaction else (
                            "delta" if state.needs_delta else "refresh"
                        )
                        self._record(
                            epoch_id=epoch_id, msg_id=message.msg_id,
                            msg_seq=message.seq, committed_at=message.ts,
                            action=action, component="dnm_summary",
                            input_msg_ids=state.current_msg_ids,
                            affected_trees=(tree_id,),
                            candidate_scores={
                                "centroid_drift": state.centroid_drift,
                                "churn_ratio": state.churn_ratio,
                                "invalidation_count": float(state.invalidation_count),
                            },
                            thresholds={
                                "drift_delta": self.dnm.config.threshold(
                                    "drift_delta", view=state.view, level=state.level),
                                "churn_delta": self.dnm.config.threshold(
                                    "churn_delta", view=state.view, level=state.level),
                                "drift_compact": self.dnm.config.threshold(
                                    "drift_compact", view=state.view, level=state.level),
                                "churn_compact": self.dnm.config.threshold(
                                    "churn_compact", view=state.view, level=state.level),
                            },
                            attributes={"node_id": state.node_id, "level": state.level},
                            identity_attributes={
                                "node_id": state.node_id,
                                "structural_generation": state.structural_generation,
                            },
                            decision=("defer" if self.dnm.config.mode == "shadow"
                                      else "apply"),
                            reason="dnm_gate",
                            available_msg_ids=available_msg_ids,
                        )
        return self._generation - before

    def observe_norm(
        self,
        *,
        epoch_id: int,
        message: Message,
        norm: Any,
        available_msg_ids: set[str],
    ) -> int:

        predecessors = tuple(sorted({value for _, value in norm.supersedes_by_subject}))
        refs = (*predecessors, message.msg_id)
        return int(self._record(
            epoch_id=epoch_id,
            msg_id=message.msg_id,
            msg_seq=message.seq,
            committed_at=message.ts,
            action="admit",
            component="norm_relation",
            input_msg_ids=refs,
            attributes={
                "kind": norm.kind,
                "subjects": "|".join(norm.subjects),
                "supersedes": norm.supersedes or "",
                "subject_edges": json.dumps(
                    dict(norm.supersedes_by_subject), sort_keys=True, separators=(",", ":")
                ),
            },
            decision="observe",
            reason="incremental_norm_extraction",
            available_msg_ids=available_msg_ids,
        ))

    def observe_epoch(
        self,
        *,
        epoch_id: int,
        messages: Sequence[Message],
        explicit_events: Sequence[Mapping[str, Any]],
        placement: Any,
        store: Any,
        touched_tree_ids: Iterable[str],
        meaning_dirty_tree_ids: Iterable[str],
        forked_tree_ids: Iterable[str],
        available_msg_ids: set[str],
        message_lookup: Mapping[str, Message] | None = None,
    ) -> int:

        before = self._generation
        ordered = sorted(messages, key=lambda message: (message.seq, message.msg_id))
        fallback = ordered[-1] if ordered else None
        if fallback is None and message_lookup is not None:
            trigger_messages = [
                message_lookup.get(str(event.get("trigger_msg_id") or ""))
                for event in explicit_events
            ]
            fallback = max(
                (message for message in trigger_messages if message is not None),
                key=lambda message: (message.seq, message.msg_id),
                default=None,
            )
        for relation in sorted(
            explicit_events,
            key=lambda row: (
                str(row.get("trigger_msg_id") or ""),
                str(row.get("target_msg_id") or ""),
                str(row.get("subject") or ""),
            ),
        ):
            trigger = str(relation.get("trigger_msg_id") or "")
            target = str(relation.get("target_msg_id") or "")
            message = (message_lookup or {}).get(trigger)
            if message is None:
                message = next((row for row in ordered if row.msg_id == trigger), fallback)
            refs = tuple(value for value in (target, trigger) if value)
            relation_trees = {
                row.tree_id for ref in refs for row in placement.placements_of(ref)
            }
            self._record(
                epoch_id=epoch_id,
                msg_id=trigger or (message.msg_id if message else ""),
                msg_seq=message.seq if message else -1,
                committed_at=(
                    message.ts if message else 0.0
                ),
                action="admit",
                component="update_relation",
                input_msg_ids=refs,
                affected_trees=relation_trees,
                attributes={
                    "source_event_id": relation.get("event_id") or relation.get("_fmc_event_id", ""),
                    "source_payload_hash": relation.get("_fmc_payload_hash", ""),
                    "subject": relation.get("subject", ""),
                    "relationship": relation.get("relationship", ""),
                    "effective_from": relation.get("effective_from", ""),
                    "ledger_applied": relation.get("_ledger_applied", False),
                },
                identity_attributes={
                    "source_event_id": relation.get("event_id") or relation.get("_fmc_event_id", ""),
                    "source_payload_hash": relation.get("_fmc_payload_hash", ""),
                    "subject": relation.get("subject", ""),
                    "relationship": relation.get("relationship", ""),
                    "effective_from": relation.get("effective_from", ""),
                },
                decision="observe",
                reason=str(relation.get("relationship") or "explicit_update_event"),
                available_msg_ids=available_msg_ids,
            )

        if self.dnm is not None:
            self.dnm.observe_supersessions(explicit_events, placement=placement, store=store)

        dirty = set(meaning_dirty_tree_ids)
        current_msg = fallback
        current_refs = tuple(sorted({
            *(message.msg_id for message in ordered),
            *(str(event.get(key) or "") for event in explicit_events
              for key in ("target_msg_id", "trigger_msg_id")),
        } - {""}))
        for tree_id in sorted(set(touched_tree_ids) | dirty):
            tree = store.get(tree_id)
            if tree is None:
                continue
            dirty_nodes = sum(
                len(node_ids)
                for level, node_ids in tree.dirty_internal_node_ids_by_level.items()
                if level > 0
            )
            self._record(
                epoch_id=epoch_id,
                msg_id=current_msg.msg_id if current_msg else "",
                msg_seq=current_msg.seq if current_msg else -1,
                committed_at=current_msg.ts if current_msg else 0.0,
                action="reorganize",
                component="summary_maintenance",
                input_msg_ids=current_refs,
                affected_trees=(tree_id,),
                candidate_scores={
                    "dirty_internal_nodes": float(dirty_nodes),
                    "leaf_count": float(tree.leaf_count),
                    "meaning_dirty": 1.0 if tree_id in dirty else 0.0,
                },
                attributes={"tree_id": tree_id},
                decision="defer",
                reason="shadow_candidate_only",
                available_msg_ids=available_msg_ids,
            )
            if self.dnm is not None:
                changed = (
                    self.dnm.reconcile_tree(tree)
                    if tree_id in set(forked_tree_ids)
                    else set()
                )
                for key in sorted(changed):
                    state = self.dnm.states[key]
                    action = "compact" if state.needs_compaction else (
                        "delta" if state.needs_delta else "refresh"
                    )
                    self._record(
                        epoch_id=epoch_id,
                        msg_id=current_msg.msg_id if current_msg else "",
                        msg_seq=current_msg.seq if current_msg else -1,
                        committed_at=current_msg.ts if current_msg else 0.0,
                        action=action,
                        component="dnm_summary",
                        input_msg_ids=state.current_msg_ids,
                        affected_trees=(tree_id,),
                        candidate_scores={
                            "centroid_drift": state.centroid_drift,
                            "churn_ratio": state.churn_ratio,
                            "invalidation_count": float(state.invalidation_count),
                        },
                        thresholds={
                            "drift_delta": self.dnm.config.threshold(
                                "drift_delta", view=state.view, level=state.level),
                            "churn_delta": self.dnm.config.threshold(
                                "churn_delta", view=state.view, level=state.level),
                            "drift_compact": self.dnm.config.threshold(
                                "drift_compact", view=state.view, level=state.level),
                            "churn_compact": self.dnm.config.threshold(
                                "churn_compact", view=state.view, level=state.level),
                        },
                        attributes={"node_id": state.node_id, "level": state.level},
                        identity_attributes={
                            "node_id": state.node_id,
                            "structural_generation": state.structural_generation,
                        },
                        decision="defer" if self.dnm.config.mode == "shadow" else "apply",
                        reason="dnm_gate",
                        available_msg_ids=available_msg_ids,
                    )

        for successor in sorted(set(forked_tree_ids)):
            self._record(
                epoch_id=epoch_id,
                msg_id=current_msg.msg_id if current_msg else "",
                msg_seq=current_msg.seq if current_msg else -1,
                committed_at=current_msg.ts if current_msg else 0.0,
                action="reorganize",
                component="tree_lifecycle",
                input_msg_ids=current_refs,
                affected_trees=(successor,),
                attributes={"successor_tree_id": successor},
                decision="observe",
                reason="fork_already_applied_by_legacy_path",
                available_msg_ids=available_msg_ids,
            )
        return self._generation - before

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        pending = self._events[self._persisted_count:]
        if pending:
            payload = "".join(
                json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for event in pending
            )
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            first = self._persisted_count + 1
            last = self._generation
            events_name = f"events-{first:012d}-{last:012d}-{digest[:12]}.jsonl"
            events_path = root / events_name
            if not events_path.exists():
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{events_name}.", suffix=".tmp", dir=root,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, events_path)
                    _fsync_directory(root)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                    raise
            elif events_path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"FMC segment collision at {events_path}")
            self._segments.append({
                "file": events_name,
                "first_generation": first,
                "last_generation": last,
                "event_count": len(pending),
                "sha256": digest,
            })
            self._persisted_count = len(self._events)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.config.mode,
            "policy_version": self.config.policy_version,
            "code_revision": self.config.code_revision,
            "environment_hash": self.config.environment_hash,
            "config_hash": self._config_hash,
            "generation": self._generation,
            "event_count": len(self._events),
            "segments": list(self._segments),
            "fingerprint": self.fingerprint(),
            "dnm": self.dnm is not None,
        }
        if self.dnm is not None:
            dnm_manifest = self.dnm.save(root / "dnm")
            manifest["dnm_manifest"] = dnm_manifest.name
        manifest_path = root / "manifest.json"
        fd, temporary = tempfile.mkstemp(
            prefix=".manifest.", suffix=".tmp", dir=root,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
            _fsync_directory(root)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return manifest_path

    @classmethod
    def load(
        cls, directory: str | Path, *, config: FmcConfig | None = None,
        dnm_vector_of: Any | None = None,
    ) -> "ForestMaintenanceController":
        root = Path(directory)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return cls(config=config or FmcConfig())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported FMC schema: {manifest.get('schema_version')}")
        stored_config = FmcConfig(
            mode=str(manifest.get("mode") or "shadow"),
            policy_version=str(manifest.get("policy_version") or "fmc-shadow-v1"),
            code_revision=str(manifest.get("code_revision") or "unknown"),
            environment_hash=str(manifest.get("environment_hash") or "unspecified"),
        )
        dnm = None
        if manifest.get("dnm"):
            if dnm_vector_of is None:
                raise ValueError("loading FMC with DNM state requires dnm_vector_of")
            dnm_manifest = json.loads((root / "dnm" / "manifest.json").read_text(encoding="utf-8"))
            dnm = DnmController.load(
                root / "dnm", vector_of=dnm_vector_of,
                manifest_name=str(manifest.get("dnm_manifest") or "manifest.json"),
            )
            if dnm.dim != int(dnm_manifest["dim"]):
                raise ValueError("DNM dimension mismatch")
        controller = cls(config=config or stored_config, dnm=dnm)
        if controller._config_hash != str(manifest.get("config_hash")):
            raise ValueError("FMC config hash mismatch")
        segments = manifest.get("segments") or []
        for segment_number, segment in enumerate(segments, 1):
            events_path = root / str(segment["file"])
            payload = events_path.read_text(encoding="utf-8")
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if digest != segment.get("sha256"):
                raise ValueError(f"FMC segment {segment_number} checksum mismatch")
            count_before = len(controller._events)
            for line_number, line in enumerate(payload.splitlines(), 1):
                if not line.strip():
                    continue
                event = FmcEvent.from_dict(json.loads(line))
                if event.config_hash != controller._config_hash:
                    raise ValueError(
                        f"FMC config mismatch at segment {segment_number} line {line_number}"
                    )
                if event.prior_generation != controller._generation:
                    raise ValueError(
                        f"non-contiguous FMC generation at segment {segment_number} "
                        f"line {line_number}"
                    )
                if event.event_id in controller._by_id:
                    raise ValueError(
                        f"duplicate FMC event id at segment {segment_number} line {line_number}"
                    )
                controller._events.append(event)
                controller._by_id[event.event_id] = event
                controller._generation = event.output_generation
            if len(controller._events) - count_before != int(segment.get("event_count", -1)):
                raise ValueError(f"FMC segment {segment_number} event count mismatch")
            controller._segments.append(dict(segment))
        controller._persisted_count = len(controller._events)
        if controller._generation != int(manifest.get("generation", -1)):
            raise ValueError("FMC manifest generation mismatch")
        if len(controller._events) != int(manifest.get("event_count", -1)):
            raise ValueError("FMC manifest event count mismatch")
        if controller.fingerprint() != manifest.get("fingerprint"):
            raise ValueError("FMC semantic fingerprint mismatch")
        return controller


__all__ = ["FmcConfig", "FmcEvent", "ForestMaintenanceController", "SCHEMA_VERSION"]
