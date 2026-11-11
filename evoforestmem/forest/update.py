from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from evoforestmem.forest.btree import build_tree_from_payloads

from ..log.message import Message
from ..log.store import MessageLog
from ..log.validity import ValidityLedger
from .builder import ForestBuilder
from .maintenance import ForestMaintenanceController
from .placement import PlacementMap
from .store import TreeStore
from .types import PolyTree, split_tree_id


_FORK_RELATIONSHIPS: frozenset[str] = frozenset({"supersede", "archive", "revise"})


def _update_event_id(event: Mapping[str, Any]) -> str:
    supplied = str(event.get("event_id") or "").strip()
    if supplied:
        return supplied
    identity = {
        key: event.get(key)
        for key in (
            "subject", "relationship", "target_msg_id", "trigger_msg_id",
            "target_step", "effective_from",
        )
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _update_event_payload_hash(event: Mapping[str, Any]) -> str:
    payload = {
        key: event.get(key)
        for key in (
            "event_id", "subject", "relationship", "target_msg_id", "trigger_msg_id",
            "target_step", "effective_from",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ForkConfig:


    domain: str
    fork_threshold: float = 0.5
    fork_on_marker: bool = True

    @classmethod
    def from_dict(
        cls, d: Mapping[str, Any] | None, *, domain: str,
    ) -> "ForkConfig | None":

        if not d:
            return None
        thresholds = d.get("fork_threshold") or {}
        threshold = thresholds.get(domain)
        if threshold is None:

            threshold = thresholds.get("ever", 0.5) if domain.startswith("ever") else 0.5
        return cls(
            domain=domain,
            fork_threshold=float(threshold),
            fork_on_marker=bool(d.get("fork_on_marker", True)),
        )


@dataclass(frozen=True, slots=True)
class UpdateReport:
    epoch_id: int
    messages: int
    trees_touched: int
    claims_added: int
    meaning_dirty_trees: int
    checkpointed: bool
    forks: int = 0
    forked_trees: tuple[str, ...] = ()
    maintenance_events: int = 0
    norms_added: int = 0
    episodes_touched: int = 0
    structural_compactions: int = 0
    timings: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "messages": self.messages,
            "trees_touched": self.trees_touched,
            "claims_added": self.claims_added,
            "meaning_dirty_trees": self.meaning_dirty_trees,
            "forks": self.forks,
            "forked_trees": list(self.forked_trees),
            "maintenance_events": self.maintenance_events,
            "norms_added": self.norms_added,
            "episodes_touched": self.episodes_touched,
            "structural_compactions": self.structural_compactions,
            "timings": dict(self.timings),
            "checkpointed": self.checkpointed,
        }


class SummaryFlushInLoopError(RuntimeError):
    pass


class _AvailableIds:


    __slots__ = ("_log", "_added")

    def __init__(self, log) -> None:
        self._log = log
        self._added: set[str] = set()

    def __contains__(self, msg_id: object) -> bool:
        return msg_id in self._added or msg_id in self._log

    def add(self, msg_id: str) -> None:
        self._added.add(msg_id)

    def __iter__(self):
        yield from (m.msg_id for m in self._log)
        yield from (mid for mid in self._added if mid not in self._log)

    def __len__(self) -> int:
        return len(self._log) + sum(1 for mid in self._added if mid not in self._log)


    def __sub__(self, other):
        return set(self) - set(other)

    def __rsub__(self, other):
        return {mid for mid in other if mid not in self}

    def __or__(self, other):
        return set(self) | set(other)

    def __ror__(self, other):
        return set(other) | set(self)

    def __and__(self, other):
        return {mid for mid in other if mid in self}

    def __rand__(self, other):
        return {mid for mid in other if mid in self}


class _LazyLookup(Mapping):


    __slots__ = ("_log", "_extra")

    def __init__(self, log, extra: Mapping[str, Message]) -> None:
        self._log = log
        self._extra = extra

    def __getitem__(self, msg_id: str) -> Message:
        if msg_id in self._extra:
            return self._extra[msg_id]
        return self._log.get(msg_id)

    def __iter__(self):
        yield from self._extra
        yield from (m.msg_id for m in self._log if m.msg_id not in self._extra)

    def __len__(self) -> int:
        return len(self._extra) + sum(1 for m in self._log if m.msg_id not in self._extra)


class PeriodUpdater:


    def __init__(
        self,
        *,
        log: MessageLog,
        store: TreeStore,
        placement: PlacementMap,
        ledger: ValidityLedger,
        builder: ForestBuilder | None = None,
        checkpoint_dir: str | Path | None = None,
        summariser: Any | None = None,
        flush_on_checkpoint: bool = True,
        fork_config: ForkConfig | None = None,
        fork_embedder: Any = None,
        maintenance: ForestMaintenanceController | None = None,
        norms: Any = None,
        episodes: Any = None,
        node_index_checkpoint: Any = None,
    ) -> None:

        self.log = log
        self.store = store
        self.placement = placement
        self.ledger = ledger
        self.builder = builder or ForestBuilder(store=store, placement=placement)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.summariser = summariser
        self.flush_on_checkpoint = flush_on_checkpoint


        self._fork_config = fork_config
        self._fork_embedder = fork_embedder


        self.maintenance = maintenance
        self.norms = norms
        self.episodes = episodes
        self.node_index_checkpoint = node_index_checkpoint
        self.periods: list[dict[str, Any]] = []
        self._processed_update_event_ids: set[str] = {
            dict(event.attributes).get("source_event_id", "")
            for event in (maintenance.events if maintenance is not None else ())
            if event.component == "update_relation"
        } - {""}
        self._processed_update_event_payloads: dict[str, str] = {
            dict(event.attributes).get("source_event_id", ""): dict(event.attributes).get(
                "source_payload_hash", ""
            )
            for event in (maintenance.events if maintenance is not None else ())
            if event.component == "update_relation"
            and dict(event.attributes).get("source_event_id")
        }

        self.summary_report: dict[str, Any] | None = None
        self.last_checkpoint_timings: dict[str, float] = {}

    def _append_exact(self, message: Message) -> tuple[Message, bool]:
        if message.msg_id in self.log:
            return self.log.get(message.msg_id), False
        stored = self.log.append(
            text=message.text,
            speaker_key=message.speaker_key,
            channel_key=message.channel_key,
            ts=message.ts,
            reply_to=message.reply_to,
            tags=message.tags,
        )
        if stored.msg_id != message.msg_id:
            raise ValueError(
                f"incremental L0 id drift: expected {message.msg_id}, got {stored.msg_id}"
            )
        return stored, True

    def _mark_meaning_dirty(self, msg_id: str) -> set[str]:
        touched: set[str] = set()
        for placement in self.placement.placements_of(msg_id):
            tree = self.store.get(placement.tree_id)
            if tree is None:
                continue
            node = tree.nodes.get(placement.leaf_node_id)
            parent_id = node.parent_id if node is not None else None
            while parent_id is not None:
                parent = tree.nodes.get(parent_id)
                if parent is None:
                    break
                parent.summary_dirty = True
                tree.dirty_internal_node_ids_by_level.setdefault(parent.level, set()).add(parent.node_id)
                parent_id = parent.parent_id
            touched.add(tree.tree_id)
        return touched


    def _artifact_tree_of(self, target: str) -> PolyTree | None:

        for p in self.placement.placements_of(target):
            if p.view == "artifact":
                tree = self.store.get(p.tree_id)
                if tree is not None:
                    return tree
        return None

    def _has_fork_marker(
        self, event: Mapping[str, Any], lookup: Mapping[str, Message],
    ) -> bool:

        rel = str(event.get("relationship") or "").lower()
        if rel in _FORK_RELATIONSHIPS:
            return True
        if not self._fork_config or not self._fork_config.fork_on_marker:
            return False
        trg = lookup.get(str(event.get("trigger_msg_id") or ""))
        if trg is None:
            return False
        tags = getattr(trg, "tags", None)
        if tags is None:
            return False
        if "supersede" in (getattr(tags, "markers", ()) or ()):
            return True
        return getattr(tags, "lifecycle", None) in {"archive", "revise"}

    def _cosine(self, a: str, b: str) -> float | None:

        try:
            import numpy as np

            vecs = self._fork_embedder.encode([a, b], kind="document")
            arr = np.asarray(vecs, dtype="float64")
            if arr.ndim != 2 or arr.shape[0] < 2:
                return None
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = arr / norms
            return float(unit[0] @ unit[1])
        except Exception:
            return None

    def _should_fork(
        self, event: Mapping[str, Any], lookup: Mapping[str, Message],
    ) -> bool:

        cfg = self._fork_config
        if cfg is None or cfg.fork_threshold <= 0.0:
            return False
        target = str(event.get("target_msg_id") or "")
        trigger = str(event.get("trigger_msg_id") or "")
        if not target or not trigger or target == trigger:
            return False
        old = self._artifact_tree_of(target)
        if old is None or old.deactivated:
            return False
        if self._fork_embedder is not None:
            tgt = lookup.get(target)
            trg = lookup.get(trigger)
            if tgt is None or trg is None:
                return False
            cos = self._cosine(tgt.text, trg.text)
            return cos is not None and cos < cfg.fork_threshold
        return self._has_fork_marker(event, lookup)

    @staticmethod
    def _base_tree_id(tree_id: str) -> str:

        idx = tree_id.rfind("#v")
        if idx > 0 and tree_id[idx + 2:].isdigit():
            return tree_id[:idx]
        return tree_id

    def _next_version_id(self, base_tree_id: str) -> str:

        highest = 0
        for tid in self.store.trees:
            if self._base_tree_id(tid) != base_tree_id:
                continue
            suffix = tid[len(base_tree_id):]
            if suffix.startswith("#v") and suffix[2:].isdigit():
                highest = max(highest, int(suffix[2:]))
            else:
                highest = max(highest, 1)
        return f"{base_tree_id}#v{highest + 1}"

    def _fork_on_revision(
        self, event: Mapping[str, Any], lookup: Mapping[str, Message],
    ) -> str | None:

        target = str(event.get("target_msg_id") or "")
        trigger = str(event.get("trigger_msg_id") or "")
        old = self._artifact_tree_of(target)
        if old is None or old.deactivated:
            return None
        trigger_trees = {
            p.tree_id for p in self.placement.placements_of(trigger)
            if p.view == "artifact"
        }
        if old.tree_id not in trigger_trees:
            return None
        order = list(old.payload_ids_ordered)
        try:
            split_idx = order.index(trigger)
        except ValueError:
            return None
        tail = order[split_idx:]
        if not tail:
            return None
        view, key = split_tree_id(old.tree_id)
        base = self._base_tree_id(old.tree_id)
        successor_id = self._next_version_id(base)

        payloads = []
        for mid in tail:
            msg = self.log.get(mid)
            ts = msg.ts if msg is not None else 0.0
            payloads.append((mid, ts, ts))
        successor, _summary_requests = build_tree_from_payloads(
            successor_id, view, key, old.label, old.k, payloads,
            append_sealed=old.append_sealed,
        )
        successor.lifecycle_state = "active"


        leaf_pos = {mid: i for i, mid in enumerate(successor.payload_ids_ordered)}
        for mid in tail:
            self.placement.remove(old.tree_id, mid)
            leaf_node_id = successor.leaf_by_payload.get(mid)
            if leaf_node_id is None:
                continue
            self.placement.add(
                view=view,
                tree_id=successor_id,
                msg_id=mid,
                leaf_node_id=leaf_node_id,
                leaf_pos=leaf_pos.get(mid, 0),
            )


        old.lifecycle_state = "suppressed"
        old.deactivated = True
        old.replacement_tree_ids = [successor_id]
        old.updated_at = time.time()
        self.store.register(successor)
        return successor_id

    def update(
        self,
        epoch_id: int,
        messages: Iterable[Message],
        *,
        explicit_events: Iterable[Mapping[str, Any]] = (),
        message_lookup: Mapping[str, Message] | None = None,
        checkpoint: bool = False,
    ) -> UpdateReport:
        timings = {
            "l0_write_s": 0.0, "route_s": 0.0, "insert_s": 0.0,
            "maintenance_s": 0.0, "placement_s": 0.0, "validity_s": 0.0,
            "summary_wait_s": 0.0, "summary_service_s": 0.0,
            "node_index_encode_s": 0.0, "node_index_write_s": 0.0,
            "bridge_s": 0.0, "checkpoint_s": 0.0,
        }
        ordered = sorted(messages, key=lambda m: (m.ts, m.seq, m.msg_id))
        unique_events: list[dict[str, Any]] = []
        seen_this_call: dict[str, str] = {}
        for raw_event in explicit_events:
            event = dict(raw_event)
            event_id = _update_event_id(event)
            payload_hash = _update_event_payload_hash(event)
            prior_hash = self._processed_update_event_payloads.get(event_id)
            if prior_hash and prior_hash != payload_hash:
                raise ValueError(f"conflicting payload for processed update event {event_id}")
            call_hash = seen_this_call.get(event_id)
            if call_hash is not None:
                if call_hash != payload_hash:
                    raise ValueError(f"conflicting payload for duplicate update event {event_id}")
                continue
            seen_this_call[event_id] = payload_hash
            if event_id in self._processed_update_event_ids:
                continue
            event["_fmc_event_id"] = event_id
            event["_fmc_payload_hash"] = payload_hash
            unique_events.append(event)
        events = tuple(unique_events)


        preexisting_msg_ids = (
            {message.msg_id for message in self.log} if unique_events else frozenset()
        )
        touched: set[str] = set()
        stored_messages: list[Message] = []
        maintenance_events = 0
        norms_added = 0
        episodes_touched = 0
        structural_compactions = 0
        available_msg_ids = _AvailableIds(self.log)
        for message in ordered:
            started = time.perf_counter()
            stored, is_new = self._append_exact(message)
            timings["l0_write_s"] += time.perf_counter() - started
            if not is_new:
                continue
            stored_messages.append(stored)
            trees_before = set(self.store.trees) if self.maintenance is not None else set()
            started = time.perf_counter()
            self.builder.add(stored)
            elapsed = time.perf_counter() - started
            measured = getattr(self.builder, "last_add_timings", None)
            if measured:
                for key in ("route_s", "insert_s", "placement_s"):
                    timings[key] += float(measured.get(key, 0.0))
            else:
                timings["insert_s"] += elapsed
            placed_tree_ids = {
                placement.tree_id for placement in self.placement.placements_of(stored.msg_id)
            }
            touched.update(placed_tree_ids)
            available_msg_ids.add(stored.msg_id)
            if self.maintenance is not None:
                started = time.perf_counter()
                maintenance_events += self.maintenance.observe_message(
                    epoch_id=int(epoch_id),
                    message=stored,
                    placement=self.placement,
                    minted_tree_ids=set(self.store.trees) - trees_before,
                    available_msg_ids=available_msg_ids,
                    store=self.store,
                )
                timings["maintenance_s"] += time.perf_counter() - started
            if self.norms is not None:
                norm = self.norms.append(stored)
                if norm is not None:
                    norms_added += 1
                    if self.maintenance is not None:
                        maintenance_events += self.maintenance.observe_norm(
                            epoch_id=int(epoch_id),
                            message=stored,
                            norm=norm,
                            available_msg_ids=available_msg_ids,
                        )
            if self.episodes is not None:
                episodes_touched += len(self.episodes.append(stored))


        lookup = _LazyLookup(self.log, dict(message_lookup or {}))
        should_compact = getattr(self.builder, "should_compact", None)
        compact_structure = getattr(self.builder, "compact", None)
        if should_compact is not None and compact_structure is not None and should_compact():
            compacted = compact_structure()
            structural_compactions = int(compacted.get("trees", 0))
            touched.update(compacted.get("tree_ids", ()))
        stored_ids = {message.msg_id for message in stored_messages}
        stored_positions = {message.msg_id: index for index, message in enumerate(stored_messages)}
        committed_rows: list[dict[str, Any]] = []
        for event in events:
            trigger_id = str(event.get("trigger_msg_id") or "")
            target_id = str(event.get("target_msg_id") or "")
            if (
                trigger_id not in self.log
                or not str(event.get("subject") or "").strip()
                or target_id not in self.log
                or not (
                    target_id in preexisting_msg_ids
                    or stored_positions.get(target_id, 1 << 60)
                    <= stored_positions.get(trigger_id, -1)
                )
            ):
                continue
            committed_rows.append({
                **event,
                "_ledger_applied": True,
            })
        committed_events = tuple(committed_rows)
        ledger_messages = list(stored_messages)
        for event in committed_events:
            trigger_id = str(event.get("trigger_msg_id") or "")
            if trigger_id not in stored_ids:
                trigger = lookup.get(trigger_id)
                if trigger is not None:
                    ledger_messages.append(trigger)
        started = time.perf_counter()
        claims_added = self.ledger.extend(
            ledger_messages,
            int(epoch_id),
            explicit_events=committed_events,
            message_lookup=lookup,
        )
        timings["validity_s"] += time.perf_counter() - started

        meaning_dirty: set[str] = set()
        for event in committed_events:
            target = str(event.get("target_msg_id") or "")
            if target:
                meaning_dirty.update(self._mark_meaning_dirty(target))


        forked_trees: list[str] = []
        if self._fork_config is not None:
            for event in committed_events:
                if self._should_fork(event, lookup):
                    new_id = self._fork_on_revision(event, lookup)
                    if new_id is not None:
                        forked_trees.append(new_id)
        self._processed_update_event_ids.update(
            str(event["_fmc_event_id"]) for event in committed_events
        )
        self._processed_update_event_payloads.update({
            str(event["_fmc_event_id"]): str(event["_fmc_payload_hash"])
            for event in committed_events
        })

        if self.maintenance is not None:
            started = time.perf_counter()
            maintenance_events += self.maintenance.observe_epoch(
                epoch_id=int(epoch_id),
                messages=stored_messages,
                explicit_events=committed_events,
                placement=self.placement,
                store=self.store,
                touched_tree_ids=touched,
                meaning_dirty_tree_ids=(
                    meaning_dirty if stored_messages or committed_events else ()
                ),
                forked_tree_ids=forked_trees,
                available_msg_ids=available_msg_ids,
                message_lookup=lookup,
            )
            timings["maintenance_s"] += time.perf_counter() - started

        self.store.mark_dirty(touched | meaning_dirty | set(forked_trees))

        if stored_messages:
            self.periods.append({
                "epoch_id": int(epoch_id),
                "label": f"message-{int(epoch_id):06d}",
                "t_start": min(m.ts for m in stored_messages),
                "t_end": max(m.ts for m in stored_messages),
                "msg_ids": [m.msg_id for m in stored_messages],
            })
        if checkpoint:
            started = time.perf_counter()
            self.checkpoint()
            timings["checkpoint_s"] += time.perf_counter() - started
            timings.update(self.last_checkpoint_timings)
        return UpdateReport(
            epoch_id=int(epoch_id),
            messages=len(stored_messages),
            trees_touched=len(touched),
            claims_added=claims_added,
            meaning_dirty_trees=len(meaning_dirty),
            forks=len(forked_trees),
            forked_trees=tuple(forked_trees),
            maintenance_events=maintenance_events,
            norms_added=norms_added,
            episodes_touched=episodes_touched,
            structural_compactions=structural_compactions,
            timings=timings,
            checkpointed=checkpoint,
        )


    def _pending_flush_trees(self) -> list[Any]:

        if self.summariser is None or not self.flush_on_checkpoint:
            return []
        needs_flush = getattr(self.summariser, "needs_flush", None)
        if needs_flush is None:
            return []
        dnm = self.maintenance.dnm if self.maintenance is not None else None
        return [
            tree for tree in self.store.trees.values()
            if (needs_flush(tree, dnm) if dnm is not None else needs_flush(tree))
        ]

    def _record_flush(self, report: Any) -> None:

        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        if self.summary_report is None:
            self.summary_report = {"flushes": 0}
        merged = self.summary_report
        merged["flushes"] = int(merged.get("flushes", 0)) + 1
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + value
            elif isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            else:
                merged[key] = value

    async def _aflush(self) -> Any | None:
        pending = self._pending_flush_trees()
        if not pending:
            return None
        report = await self.summariser.flush(pending)
        self.store.mark_dirty(tree.tree_id for tree in pending)
        self._record_flush(report)
        return report

    def _flush_blocking(self) -> Any | None:

        if not self._pending_flush_trees():
            return None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aflush())
        raise SummaryFlushInLoopError(
            "summary flush cannot run inside an active event loop; "
            "await PeriodUpdater.acheckpoint() from async code, or construct the "
            "updater without a summariser and flush separately"
        )


    def _write_checkpoint(self, root: Path) -> Path:

        log_dir = root / "log" / self.store.batch
        index_dir = root / "index" / self.store.batch
        log_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.log.save(log_dir / "messages.jsonl")
        self.ledger.save(index_dir / "validity.jsonl")
        self.store.save_dirty()
        self.placement.save(index_dir / "placement.jsonl")
        save_delta = getattr(self.builder, "save_delta", None)
        if save_delta is not None:
            save_delta(index_dir / "structural_lsm")
        (index_dir / "periods.json").write_text(
            json.dumps(self.periods, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.norms is not None:
            self.norms.save(index_dir / "norms.jsonl")
        if self.episodes is not None:
            self.episodes.save(index_dir / "episodes.jsonl")
        if self.node_index_checkpoint is not None:
            self.node_index_checkpoint(index_dir / "node_index")
        if self.maintenance is not None:
            if self.maintenance.dnm is not None:
                self.maintenance.dnm.validate(
                    available_msg_ids={message.msg_id for message in self.log}
                )
            self.maintenance.save(index_dir / "maintenance")
        return root

    def _checkpoint_root(self, directory: str | Path | None) -> Path:
        root = Path(directory) if directory is not None else self.checkpoint_dir
        if root is None:
            raise ValueError("checkpoint directory is required")
        return root

    def checkpoint(self, directory: str | Path | None = None) -> Path:

        root = self._checkpoint_root(directory)
        started = time.perf_counter()
        report = self._flush_blocking()
        summary_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        result = self._write_checkpoint(root)
        self.last_checkpoint_timings = {
            "summary_wait_s": summary_elapsed,
            "summary_service_s": float(getattr(report, "elapsed_sec", 0.0) or 0.0),
            "checkpoint_write_s": time.perf_counter() - started,
        }
        return result

    async def acheckpoint(self, directory: str | Path | None = None) -> Path:

        root = self._checkpoint_root(directory)
        started = time.perf_counter()
        report = await self._aflush()
        summary_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        result = self._write_checkpoint(root)
        self.last_checkpoint_timings = {
            "summary_wait_s": summary_elapsed,
            "summary_service_s": float(getattr(report, "elapsed_sec", 0.0) or 0.0),
            "checkpoint_write_s": time.perf_counter() - started,
        }
        return result

    def interval_invariant(self) -> dict[str, Any]:

        violations: list[str] = []
        prev_end: float | None = None
        seen_epochs: set[int] = set()
        for period in self.periods:
            epoch = period.get("epoch_id")
            t0 = period.get("t_start")
            t1 = period.get("t_end")
            if t0 is None or t1 is None or t0 > t1:
                violations.append(f"epoch {epoch}: malformed interval [{t0}, {t1}]")
                continue
            if epoch in seen_epochs:
                violations.append(f"epoch {epoch}: duplicate epoch_id")
            seen_epochs.add(epoch)
            if prev_end is not None and t0 < prev_end:
                violations.append(
                    f"epoch {epoch}: t_start {t0} precedes previous t_end {prev_end}"
                )
            for mid in period.get("msg_ids", ()):
                msg = self.log.get(mid) if mid in self.log else None
                if msg is not None and not (t0 <= msg.ts <= t1):
                    violations.append(
                        f"epoch {epoch}: message {mid} ts {msg.ts} outside [{t0}, {t1}]"
                    )
            prev_end = t1
        return {"periods": len(self.periods), "violations": violations, "ok": not violations}
