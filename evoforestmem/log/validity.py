from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .message import Message


_HOST_BLOCK = frozenset({
    "sharepoint", "confluence", "figma", "jira", "github", "gitlab", "gitlabci",
    "swagger", "openapi", "notion", "slack", "teams", "trello", "asana", "monday",
    "clickup", "linear", "airtable", "excel", "word", "powerpoint", "outlook",
    "onedrive", "postman", "wiki", "ado", "azuredevops", "vsts", "jenkins",
    "kubernetes", "docker", "grafana", "datadog", "sentry",
})


_IDENT = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{1,}(?:[_-][A-Z0-9]+)+|[A-Z]{2,}-\d+|`[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+`)"
)
_WORDS = re.compile(r"[a-z0-9_]+")
_WORD = re.compile(r"[a-z0-9]+")
_URL_OR_HOST = re.compile(r"://|^https?|^www\.|[a-z0-9-]+\.[a-z]{2,}(\.[a-z]{2,})?")
_STOP = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "about",
    "must", "should", "will", "would", "could", "system", "team", "project",
    "but", "not", "are", "was", "were", "have", "has", "been", "all", "any",
    "new", "old", "final", "almost", "temporary", "actionable", "confirmed",
    "normal", "creative", "pending", "active", "current", "latest", "completed",
    "delete", "update", "add", "helpfully",
})


def claim_id(subject: str, kind: str, msg_id: str) -> str:
    raw = f"{subject}\0{kind}\0{msg_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _is_subject(subject: str) -> bool:

    s = subject.strip().lower()
    if len(s) < 3:
        return False
    if _URL_OR_HOST.search(s):
        return False
    tokens = _WORD.findall(s)
    if not tokens or not re.search(r"[a-z]", s):
        return False
    if all(tok in _STOP for tok in tokens):
        return False
    if s in _HOST_BLOCK:
        return False
    if any(tok in _HOST_BLOCK for tok in tokens):
        return False
    return True


def extract_subjects(msg: Message) -> tuple[str, ...]:

    found: list[str] = []
    for x in msg.tags.artifacts:
        s = x.strip().lower()
        if 2 < len(s) < 80 and _is_subject(s):
            found.append(s)
    for x in _IDENT.findall(msg.text):
        s = x.strip("`").lower()
        if _is_subject(s):
            found.append(s)
    channel = msg.channel_key
    scoped: list[str] = []
    for s in found:
        key = s if ":" in s else f"{channel}:{s}"
        if key not in scoped:
            scoped.append(key)
    return tuple(scoped)[:8]


def _terms(text: str) -> frozenset[str]:
    return frozenset(x for x in _WORDS.findall(text.lower()) if len(x) > 2 and x not in _STOP)


def infer_kind(msg: Message) -> str:
    markers = set(msg.tags.markers)
    if markers & {"decided", "ratified", "signoff"}:
        return "decision"
    if msg.tags.lifecycle in {"start", "progress", "complete", "archive", "revise"}:
        return "task_state"
    if markers & {"mandate", "effective_now", "supersede"}:
        return "norm"
    return "fact"


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    subject: str
    kind: str
    value_ref: tuple[str, ...]
    asserted_at: float
    recorded_at: float
    effective_from: float
    valid_to: float | None
    valid_to_period: int | None
    superseded_by: str | None
    supersedes: str | None


    relation_source: str | None
    relation_confidence: float | None
    actor_id: str
    channel_key: str
    markers: tuple[str, ...]
    period: int

    @property
    def active(self) -> bool:
        return self.superseded_by is None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["value_ref"] = list(self.value_ref)
        value["markers"] = list(self.markers)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Claim":
        return cls(
            claim_id=str(value["claim_id"]),
            subject=str(value["subject"]),
            kind=str(value["kind"]),
            value_ref=tuple(value.get("value_ref", ())),
            asserted_at=float(value["asserted_at"]),
            recorded_at=float(value.get("recorded_at", value["asserted_at"])),
            effective_from=float(value.get("effective_from", value["asserted_at"])),
            valid_to=(float(value["valid_to"]) if value.get("valid_to") is not None else None),
            valid_to_period=(
                int(value["valid_to_period"])
                if value.get("valid_to_period") is not None else None
            ),
            superseded_by=value.get("superseded_by"),
            supersedes=value.get("supersedes"),
            relation_source=value.get("relation_source"),
            relation_confidence=(
                float(value["relation_confidence"])
                if value.get("relation_confidence") is not None else None
            ),
            actor_id=str(value.get("actor_id", "")),
            channel_key=str(value.get("channel_key", "")),
            markers=tuple(value.get("markers", ())),
            period=int(value.get("period", 0)),
        )


class _LayeredLookup(Mapping):


    __slots__ = ("_front", "_back")

    def __init__(self, front: Mapping[str, Message], back: Mapping[str, Message]) -> None:
        self._front = front
        self._back = back

    def __getitem__(self, msg_id: str) -> Message:
        try:
            return self._front[msg_id]
        except KeyError:
            return self._back[msg_id]

    def __iter__(self):
        yield from self._front
        yield from (mid for mid in self._back if mid not in self._front)

    def __len__(self) -> int:
        return len(self._front) + sum(1 for mid in self._back if mid not in self._front)


@dataclass(frozen=True, slots=True)
class ClaimVersion:


    claim: Claim
    status: str
    head: Claim

    def describe(self) -> str:
        if self.status == "superseded":
            return "SUPERSEDED by " + ", ".join(mid[:8] for mid in self.head.value_ref)
        return {"applicable": "in force", "unknown": "not yet known",
                "pending": "not yet effective"}[self.status]


class ValidityLedger:


    def __init__(self, path: str | Path | None = None, *, supersede_overlap: float = 0.5, infer_replacements: bool = True):
        self.path = Path(path) if path is not None else None
        self.supersede_overlap = float(supersede_overlap)
        self.infer_replacements = infer_replacements
        self.claims: dict[str, Claim] = {}
        self.by_subject: dict[str, list[str]] = defaultdict(list)
        self.by_message: dict[str, list[str]] = defaultdict(list)
        self.coexists: set[tuple[str, str]] = set()
        if self.path is not None and self.path.is_file():
            self.load(self.path)

    def __len__(self) -> int:
        return len(self.claims)

    def __iter__(self) -> Iterator[Claim]:
        return iter(sorted(self.claims.values(), key=lambda c: (c.asserted_at, c.claim_id)))

    def subjects(self) -> Iterator[str]:
        return iter(sorted(self.by_subject))

    def assert_claim(
        self,
        subject: str,
        kind: str,
        value_ref: Iterable[str],
        asserted_at: float,
        effective_from: float | None,
        actor: str,
        channel: str,
        markers: Iterable[str],
        period: int,
        *,
        recorded_at: float | None = None,
    ) -> Claim:
        refs = tuple(dict.fromkeys(str(x) for x in value_ref if x))
        if not refs:
            raise ValueError("claim requires at least one L0 message reference")
        subject = subject.strip().lower()
        if not subject:
            raise ValueError("claim subject cannot be empty")
        cid = claim_id(subject, kind, refs[0])
        existing = self.claims.get(cid)
        if existing is not None:
            return existing
        claim = Claim(
            claim_id=cid,
            subject=subject,
            kind=kind,
            value_ref=refs,
            asserted_at=float(asserted_at),
            recorded_at=float(asserted_at if recorded_at is None else recorded_at),
            effective_from=float(asserted_at if effective_from is None else effective_from),
            valid_to=None,
            valid_to_period=None,
            superseded_by=None,
            supersedes=None,
            relation_source=None,
            relation_confidence=None,
            actor_id=actor,
            channel_key=channel,
            markers=tuple(dict.fromkeys(markers)),
            period=int(period),
        )
        self.claims[cid] = claim
        self.by_subject[subject].append(cid)
        for msg_id in refs:
            self.by_message[msg_id].append(cid)
        return claim

    def supersede(
        self, old_claim_id: str, new_claim_id: str, *,
        relation_source: str | None = None, relation_confidence: float | None = None,
    ) -> None:
        if old_claim_id == new_claim_id:
            return
        old = self.claims[old_claim_id]
        new = self.claims[new_claim_id]
        if old.subject != new.subject:
            raise ValueError(f"cannot supersede across subjects: {old.subject!r} != {new.subject!r}")
        if old.superseded_by not in (None, new_claim_id):
            raise ValueError(f"claim {old_claim_id} already superseded by {old.superseded_by}")
        if new.supersedes not in (None, old_claim_id):
            raise ValueError(f"claim {new_claim_id} already supersedes {new.supersedes}")
        self.claims[old_claim_id] = replace(
            old,
            valid_to=new.effective_from,
            valid_to_period=new.period,
            superseded_by=new_claim_id,
            relation_source=relation_source,
            relation_confidence=relation_confidence,
        )
        self.claims[new_claim_id] = replace(new, supersedes=old_claim_id)

    def coexist(self, claim_a: str, claim_b: str) -> None:
        if claim_a == claim_b:
            return
        if claim_a not in self.claims or claim_b not in self.claims:
            raise KeyError("coexistence requires two existing claims")
        self.coexists.add(tuple(sorted((claim_a, claim_b))))

    def claims_for_message(self, msg_id: str) -> tuple[Claim, ...]:
        return tuple(self.claims[x] for x in self.by_message.get(msg_id, ()))

    def version(self, claim: Claim, *, known_at: float | None = None,
                effective_at: float | None = None, period: int | None = None,
                resolved_only: bool = False) -> ClaimVersion:

        claim = self.claims.get(claim.claim_id, claim)
        if ((known_at is not None and claim.recorded_at > known_at)
                or (period is not None and claim.period > period)):
            return ClaimVersion(claim, "unknown", claim)
        if effective_at is not None and claim.effective_from > effective_at:
            return ClaimVersion(claim, "pending", claim)
        head, seen = claim, {claim.claim_id}
        while (nxt := self.successor(head, known_at=known_at, effective_at=effective_at,
                                    period=period, resolved_only=resolved_only)) is not None:
            if nxt.claim_id in seen:
                raise ValueError("cycle in validity chain")
            seen.add(nxt.claim_id)
            head = nxt
        return ClaimVersion(claim, "applicable" if head.claim_id == claim.claim_id
                            else "superseded", head)

    def message_versions(self, msg_id: str, *, subjects: Iterable[str] = (),
                         channel: str | None = None, **snapshot) -> tuple[ClaimVersion, ...]:
        claims = self.claims_for_message(msg_id)


        keys = {s.lower() for s in subjects}
        if keys:
            scoped = keys | {f"{prefix}:{s}" for prefix in (channel, "policy", "decision")
                             if prefix for s in keys}
            matched = tuple(c for c in claims if c.subject in scoped)
            claims = matched
        return tuple(self.version(c, **snapshot) for c in claims)

    def current(self, subject: str, *, actor: str | None = None) -> Claim | None:
        candidates = self.current_all(subject, actor=actor)
        return max(candidates, key=lambda x: (x.effective_from, x.asserted_at, x.claim_id), default=None)

    def current_all(self, subject: str, *, actor: str | None = None) -> tuple[Claim, ...]:
        candidates = [
            self.claims[x] for x in self.by_subject.get(subject.lower(), ())
            if self.claims[x].active and (actor is None or self.claims[x].actor_id == actor)
        ]
        return self._relation_selection(candidates)

    def _relation_selection(self, candidates: Iterable[Claim]) -> tuple[Claim, ...]:
        rows = list(candidates)
        if not rows:
            return ()
        ids = {claim.claim_id for claim in rows}
        related = {
            claim_id
            for pair in self.coexists if pair[0] in ids and pair[1] in ids
            for claim_id in pair
        }
        if related:
            selected = [claim for claim in rows if claim.claim_id in related]
            latest = max(rows, key=lambda x: (x.effective_from, x.asserted_at, x.claim_id))
            if latest.claim_id not in related:
                selected.append(latest)
            return tuple(sorted(
                selected,
                key=lambda x: (x.actor_id, x.effective_from, x.claim_id),
            ))
        return (max(rows, key=lambda x: (x.effective_from, x.asserted_at, x.claim_id)),)

    def as_of(
        self, subject: str, t: float, *, actor: str | None = None,
        period: int | None = None,
    ) -> Claim | None:
        candidates = self.as_of_all(subject, t, actor=actor, period=period)
        return max(candidates, key=lambda x: (x.effective_from, x.asserted_at, x.claim_id), default=None)

    def as_of_all(
        self, subject: str, t: float | None, *, actor: str | None = None,
        period: int | None = None,
    ) -> tuple[Claim, ...]:
        candidates = [self.claims[x] for x in self.by_subject.get(subject.lower(), ())]
        if actor is not None:
            candidates = [x for x in candidates if x.actor_id == actor]
        valid: list[Claim] = []
        for claim in candidates:


            if t is not None and (claim.recorded_at > t or claim.effective_from > t):
                continue
            if period is not None and claim.period > period:
                continue
            if self.successor(claim, known_at=t, effective_at=t, period=period,
                              resolved_only=False) is not None:
                continue
            valid.append(claim)
        if not valid:
            return ()
        return self._relation_selection(valid)

    def is_superseded(
        self, msg_id: str, *, as_of: float | None = None, period: int | None = None
    ) -> bool:
        claims = self.claims_for_message(msg_id)
        if not claims:
            return False
        visible = [self.version(c, known_at=as_of, effective_at=as_of, period=period)
                   for c in claims]
        visible = [v for v in visible if v.status not in ("unknown", "pending")]
        return bool(visible) and all(v.status == "superseded" for v in visible)

    def successor(self, claim: Claim, *, known_at: float | None,
                  effective_at: float | None, period: int | None = None,
                  resolved_only: bool = True) -> Claim | None:

        if resolved_only and claim.relation_source not in (None, "benchmark_given", "explicit"):
            return None
        new = self.claims.get(claim.superseded_by)
        if new is None:
            return None
        if known_at is not None and new.recorded_at > known_at:
            return None
        if effective_at is not None and new.effective_from > effective_at:
            return None
        if period is not None and new.period > period:
            return None
        return new

    def snapshot(self, subject: str, *, known_at: float,
                 effective_at: float | None = None, period: int | None = None,
                 actor: str | None = None) -> tuple[Claim, ...]:

        effective_at = known_at if effective_at is None else effective_at
        return tuple(c for cid in self.by_subject.get(subject.lower(), ())
                     if (c := self.claims[cid]).recorded_at <= known_at
                     and c.effective_from <= effective_at
                     and (period is None or c.period <= period)
                     and (actor is None or c.actor_id == actor)
                     and self.successor(c, known_at=known_at, effective_at=effective_at,
                                        period=period) is None)

    def revision_sources(self, msg_id: str, *, known_at: float,
                         effective_at: float, period: int | None = None, available=None) -> tuple[str, ...]:

        refs: dict[str, None] = {}
        for claim in self.claims_for_message(msg_id):
            seen = {claim.claim_id}
            while (nxt := self.successor(claim, known_at=known_at,
                                        effective_at=effective_at, period=period)) is not None:
                if available is not None and not all(available(mid) for mid in nxt.value_ref):
                    break
                if nxt.claim_id in seen:
                    raise ValueError("cycle in validity chain")
                seen.add(nxt.claim_id)
                refs.update(dict.fromkeys(nxt.value_ref))
                claim = nxt
        return tuple(refs)

    def linked_message_ids(self, msg_id: str, *, limit: int = 2) -> tuple[str, ...]:
        out: list[str] = []
        for claim in self.claims_for_message(msg_id):
            for linked_id in (claim.supersedes, claim.superseded_by):
                if not linked_id:
                    continue
                for candidate in self.claims[linked_id].value_ref:
                    if candidate != msg_id and candidate not in out:
                        out.append(candidate)
                        if len(out) >= limit:
                            return tuple(out)
        return tuple(out)

    def _claim_for_message(
        self,
        msg: Message,
        *,
        subject: str,
        period: int,
        kind: str | None = None,
        effective_from: float | None = None,
    ) -> Claim:
        return self.assert_claim(
            subject=subject,
            kind=kind or infer_kind(msg),
            value_ref=(msg.msg_id,),
            asserted_at=msg.ts,
            effective_from=effective_from,
            actor=msg.speaker_key,
            channel=msg.channel_key,
            markers=msg.tags.markers,
            period=period,
        )

    def extend(
        self,
        new_msgs: Iterable[Message],
        period: int,
        *,
        explicit_events: Iterable[Mapping[str, Any]] = (),
        message_lookup: Mapping[str, Message] | None = None,
    ) -> int:

        messages = sorted(new_msgs, key=lambda m: (m.ts, m.seq, m.msg_id))
        before = len(self)


        lookup = _LayeredLookup({m.msg_id: m for m in messages}, message_lookup or {})
        events_by_trigger: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for event in explicit_events:
            events_by_trigger[str(event.get("trigger_msg_id", ""))].append(event)

        for msg in messages:
            explicit = events_by_trigger.get(msg.msg_id, ())
            if explicit:
                for event in explicit:
                    subject = str(event.get("subject") or "").strip().lower()
                    target_id = str(event.get("target_msg_id") or "")
                    target_msg = lookup.get(target_id)
                    if not subject or target_msg is None:
                        continue
                    old = self._claim_for_message(
                        target_msg, subject=subject, period=min(period, int(event.get("target_step", period)))
                    )
                    new = self._claim_for_message(
                        msg,
                        subject=subject,
                        period=period,
                        effective_from=float(event.get("effective_from", msg.ts)),
                    )


                    if old.superseded_by not in (None, new.claim_id):
                        heads = [
                            self.claims[cid] for cid in self.by_subject[subject]
                            if cid != new.claim_id and self.claims[cid].active
                        ]
                        if not self.infer_replacements:

                            seen = {old.claim_id}
                            while old.superseded_by and old.superseded_by != new.claim_id:
                                old = self.claims[old.superseded_by]
                                if old.claim_id in seen:
                                    raise ValueError("cycle in validity chain")
                                seen.add(old.claim_id)
                        elif heads:
                            old = max(heads, key=lambda x: (x.effective_from, x.claim_id))
                    self.supersede(
                        old.claim_id, new.claim_id,
                        relation_source="benchmark_given", relation_confidence=1.0,
                    )
                continue

            subjects = extract_subjects(msg)
            if not subjects:
                continue
            is_revision = "supersede" in msg.tags.markers or msg.tags.lifecycle == "revise"
            for subject in subjects:
                new = self._claim_for_message(msg, subject=subject, period=period)
                prior = [
                    self.claims[x] for x in self.by_subject[subject]
                    if x != new.claim_id and self.claims[x].channel_key == msg.channel_key
                    and self.claims[x].asserted_at <= msg.ts and self.claims[x].active
                ]
                if not prior:
                    continue
                previous = max(prior, key=lambda x: (x.asserted_at, x.claim_id))
                if is_revision and self.infer_replacements:
                    self.supersede(
                        previous.claim_id, new.claim_id,
                        relation_source="rule_marker", relation_confidence=0.7,
                    )
                elif previous.actor_id != new.actor_id:
                    shared = _terms(lookup.get(previous.value_ref[0], msg).text) & _terms(msg.text)
                    base = _terms(lookup.get(previous.value_ref[0], msg).text)
                    if base and len(shared) / len(base) >= self.supersede_overlap:
                        self.coexist(previous.claim_id, new.claim_id)
        return len(self) - before

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for claim in self:
            h.update(
                f"{claim.claim_id}\0{claim.supersedes or ''}\0{claim.superseded_by or ''}\0"
                f"{claim.valid_to if claim.valid_to is not None else ''}\0"
                f"{claim.valid_to_period if claim.valid_to_period is not None else ''}\n".encode("utf-8")
            )
        for pair in sorted(self.coexists):
            h.update(f"coexist\0{pair[0]}\0{pair[1]}\n".encode("utf-8"))
        return h.hexdigest()

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("ledger save path is required")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for claim in self:
                fh.write(json.dumps({"type": "claim", **claim.to_dict()}, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
            for a, b in sorted(self.coexists):
                fh.write(json.dumps({"type": "coexist", "claim_a": a, "claim_b": b}, sort_keys=True))
                fh.write("\n")
        self.path = target
        return target

    def load(self, path: str | Path) -> None:
        self.claims.clear()
        self.by_subject.clear()
        self.by_message.clear()
        self.coexists.clear()
        target = Path(path)
        with target.open(encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("type") == "coexist":
                    self.coexists.add(tuple(sorted((row["claim_a"], row["claim_b"]))))
                    continue
                row.pop("type", None)
                claim = Claim.from_dict(row)
                self.claims[claim.claim_id] = claim
                self.by_subject[claim.subject].append(claim.claim_id)
                for msg_id in claim.value_ref:
                    self.by_message[msg_id].append(claim.claim_id)
        self.path = target
