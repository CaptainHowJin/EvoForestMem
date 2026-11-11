from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from ..log.validity import Claim, ValidityLedger
from .s4_assemble import Span

_WHAT_CHANGED = re.compile(
    r"\bwhat\s+(?:has\s+)?changed\b|\bhow\b.{0,100}\bchange(?:d)?\b|\bchange\s+from\b",
    re.I,
)
_TOKEN = re.compile(r"[a-z0-9_]+")
_QUERY_STOP = frozenset({
    "the", "and", "what", "how", "was", "were", "did", "does", "after",
    "before", "during", "immediately", "change", "changed", "decision", "current",
})


def has_update_intent(question: str) -> bool:
    return bool(_WHAT_CHANGED.search(question))


@dataclass(frozen=True, slots=True)
class ValidityBlock:
    claims: tuple[Claim, ...]
    as_of: float | None = None
    as_of_period: int | None = None
    ledger: ValidityLedger | None = None

    @property
    def msg_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for claim in self.claims:
            for msg_id in claim.value_ref:
                if msg_id not in out:
                    out.append(msg_id)
        return tuple(out)

    def render(self) -> list[str]:
        lines: list[str] = []
        for claim in self.claims:
            asserted = datetime.fromtimestamp(claim.asserted_at, tz=timezone.utc).strftime("%Y-%m-%d")
            status = (self.ledger.version(claim, known_at=self.as_of,
                      effective_at=self.as_of, period=self.as_of_period).describe()
                      if self.ledger is not None else "version unverified")
            lines.append(
                f'    (claim "{claim.subject}" -- {claim.kind}, asserted {asserted} by '
                f"{claim.actor_id}; {status})"
            )
        return lines


@dataclass(frozen=True, slots=True)
class SuppressedSiblingBlock:


    sibling_tree_id: str
    replacement_tree_id: str
    current_branch_ids: tuple[str, ...] = ()

    @property
    def msg_ids(self) -> tuple[str, ...]:
        return self.current_branch_ids

    def render(self) -> list[str]:
        return [
            f"    (superseded branch {self.sibling_tree_id} -> see "
            f"{self.replacement_tree_id}; prior decision recovered by audit)"
        ]


@dataclass(frozen=True, slots=True)
class ValidityOverlay:
    spans: tuple[Span, ...]
    block: ValidityBlock | None
    mandatory_msg_ids: frozenset[str]
    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NormEvidence:
    blocks: tuple[Any, ...]
    mandatory_msg_ids: frozenset[str]
    inapplicable_msg_ids: frozenset[str]
    stats: dict[str, Any]


def select_norm_evidence(index, *, question: str, ledger: ValidityLedger | None,
                         log, known_at: float | None = None,
                         effective_at: float | None = None, period: int | None = None,
                         include_history: bool = False) -> NormEvidence:

    from ..forest.norms import NormBlock

    effective_at = known_at if effective_at is None else effective_at
    chosen, annotations, inapplicable = set(), {}, set()
    untracked = superseded = pending = 0

    def available(mid):
        return (mid in log and (known_at is None or log.get(mid).ts <= known_at)
                and (period is None or log.get(mid).seq <= period))

    hits = index.lookup(question, as_of=known_at, available=available)

    def record(mid, versions=(), *, pin=True):
        if not available(mid):
            return
        annotations.setdefault(mid, {}).update((v.claim.claim_id, v) for v in versions)
        if pin:
            chosen.add(mid)

    for norm in hits:
        if not available(norm.msg_id):
            continue
        versions = (ledger.message_versions(norm.msg_id, subjects=norm.subjects,
                    channel=norm.channel_key, known_at=known_at, effective_at=effective_at,
                    period=period) if ledger is not None else ())
        if not versions:
            untracked += 1
            record(norm.msg_id)
            continue

        visible = tuple(v for v in versions if v.status != "unknown")
        record(norm.msg_id, visible, pin=False)
        if visible and not any(v.status == "applicable" for v in visible):
            if not include_history or all(v.status == "pending" for v in visible):
                inapplicable.add(norm.msg_id)
        for version in visible:
            if version.status == "pending":
                pending += 1
                continue
            if version.status == "superseded":
                superseded += 1
                if include_history:
                    record(norm.msg_id, (version,))
            else:
                record(norm.msg_id, (version,))
            head_version = ledger.version(version.head, known_at=known_at,
                effective_at=effective_at, period=period)
            for mid in version.head.value_ref:
                record(mid, (head_version,))

    blocks = []
    for mid, versions in annotations.items():
        status = "; ".join(
            f"{v.claim.subject} " + ("SUPERSEDED; replacement source unavailable"
                if v.status == "superseded" and not all(available(ref) for ref in v.head.value_ref)
                else v.describe()) for v in versions.values())
        status = status or "version unverified"
        norm = index.norms.get(mid)
        if norm is not None:
            blocks.append(NormBlock((norm,), {mid: status}))
        elif versions:


            blocks.append(_ResolvedClaimBlock(mid, status))
    return NormEvidence(tuple(blocks), frozenset(chosen), frozenset(inapplicable), {
        "authority": "validity_ledger" if ledger is not None else "unavailable",
        "hits": len(hits), "untracked": untracked,
        "superseded": superseded, "pending": pending, "mandatory": len(chosen),
    })


@dataclass(frozen=True, slots=True)
class _ResolvedClaimBlock:
    msg_id: str
    status: str

    @property
    def msg_ids(self):
        return (self.msg_id,)

    def render(self):
        return [f"    (claim {self.status})"]


def apply_validity(
    spans: Sequence[Span],
    *,
    question: str,
    ledger: ValidityLedger,
    as_of: float | None,
    as_of_period: int | None = None,
    actor: str | None = None,
    penalty: float = 0.25,
    max_subjects: int = 3,
    qa_subject: str | None = None,
    norm_evidence: NormEvidence | None = None,
    discover_subjects: bool = True,
) -> ValidityOverlay:

    by_id = {span.msg_id: span for span in spans}
    support: Counter[str] = Counter()
    for span in spans:
        support.update(claim.subject for claim in ledger.claims_for_message(span.msg_id))

    qterms = {
        token for token in _TOKEN.findall(question.lower())
        if len(token) > 2 and token not in _QUERY_STOP
    }
    global_overlap: dict[str, int] = {}
    for subject in (ledger.subjects() if discover_subjects else ()):
        overlap = len(qterms & set(_TOKEN.findall(subject.lower())))
        if overlap:
            global_overlap[subject] = overlap
            support.setdefault(subject, 0)
    if qa_subject:
        qa_subject = qa_subject.strip().lower()
        if qa_subject in ledger.by_subject:


            support.setdefault(qa_subject, 0)
            global_overlap[qa_subject] = global_overlap.get(qa_subject, 0) + 10_000
    structured = [
        subject for subject in support
        if subject.startswith(("decision:", "policy:")) and global_overlap.get(subject, 0)
    ]
    subject_pool = structured or list(support)
    ranked_subjects = sorted(
        subject_pool,
        key=lambda subject: (
            -global_overlap.get(subject, 0),
            -int(subject.startswith(("decision:", "policy:"))),
            -support[subject],
            subject,
        ),
    )
    subjects = ranked_subjects[: max(0, int(max_subjects))] if discover_subjects else []

    changed = bool(_WHAT_CHANGED.search(question))
    selected: list[Claim] = []
    mandatory: set[str] = set()


    use_as_of = as_of is not None or as_of_period is not None
    for subject in subjects:
        organizational = subject.startswith(("decision:", "policy:"))
        if organizational:
            claims = (
                ledger.as_of_all(subject, as_of, period=as_of_period)
                if use_as_of else ledger.current_all(subject)
            )
        else:
            claims = (
                ledger.as_of_all(subject, as_of, actor=actor, period=as_of_period)
                if use_as_of else ledger.current_all(subject, actor=actor)
            )


        if not claims and actor is not None and not organizational:
            claims = (
                ledger.as_of_all(subject, as_of, period=as_of_period)
                if use_as_of else ledger.current_all(subject)
            )
        if not claims:
            continue
        for claim in claims:
            selected.append(claim)
            mandatory.update(claim.value_ref)
            if changed and claim.supersedes:
                previous = ledger.claims[claim.supersedes]
                selected.append(previous)
                mandatory.update(previous.value_ref)


    inapplicable = norm_evidence.inapplicable_msg_ids if norm_evidence is not None else frozenset()
    mandatory.difference_update(inapplicable)
    if norm_evidence is not None:
        mandatory.update(norm_evidence.mandatory_msg_ids)
    selected = [c for c in selected if not set(c.value_ref) & inapplicable]
    adjusted: list[Span] = []
    demoted = 0
    for span in spans:
        if span.msg_id in inapplicable or ledger.is_superseded(span.msg_id, as_of=as_of, period=as_of_period):
            adjusted.append(Span(
                msg_id=span.msg_id,
                score=span.score - abs(float(penalty)),
                view=span.view,
                headers=span.headers,
            ))
            demoted += 1
        else:
            adjusted.append(span)

    ceiling = max((span.score for span in adjusted), default=0.0) + 1.0
    injected = 0
    for msg_id in sorted(mandatory):
        if msg_id in by_id:
            continue
        adjusted.append(Span(msg_id=msg_id, score=ceiling, view="validity"))
        injected += 1

    unique: dict[str, Claim] = {}
    for claim in selected:
        unique.setdefault(claim.claim_id, claim)
    block = ValidityBlock(
        tuple(unique.values()), as_of=as_of, as_of_period=as_of_period, ledger=ledger
    ) if unique else None
    return ValidityOverlay(
        spans=tuple(adjusted),
        block=block,
        mandatory_msg_ids=frozenset(mandatory),
        stats={
            "enabled": True,
            "candidate_subjects": len(support),
            "subjects": len(subjects),
            "selected_claims": len(unique),
            "demoted": demoted,
            "injected": injected,
            "what_changed": changed,
            "as_of": as_of,
            "as_of_period": as_of_period,
        },
    )
