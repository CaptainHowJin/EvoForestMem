from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class StructureUpdateConfig:
    structure_update_mode: str = 'off'
    seed: int = 1729
    max_candidates: int = 2
    verify_per_point: int = 8
    feedback_fraction: float = .05
    replay_workers: int = 1

    def __post_init__(self):
        if self.structure_update_mode not in {'off', 'shadow', 'verified_v1', 'verified_v2', 'stream_verified_v1'}:
            raise ValueError('unknown structure_update_mode')
        for name, maximum in [('max_candidates', 2), ('verify_per_point', 8), ('replay_workers', 8)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(name + ' exceeds the verified_v1 limit')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if not 0 <= self.feedback_fraction <= .05:
            raise ValueError('feedback is limited to 5% of business tokens')


@dataclass
class StructureDelta:
    operation: str
    tree_id: str
    before_version: int
    after_version: int
    modified: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()
    ancestors: tuple[str, ...] = ()
    rollback: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CompletedQuery:

    question: str
    ask_time: float
    asker: str | None
    final_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]

    @property
    def identity(self):
        return fingerprint([self.question.strip(), self.ask_time, self.asker])


@dataclass
class Witness:
    question: str
    ask_time: float
    asker: str | None
    sources: dict[str, str]
    groups: list[list[str]]
    quotes: list[dict]
    requirements: list[dict]
    request_keys: list[str]

    @property
    def identity(self):
        return fingerprint([self.question.strip(), self.ask_time, self.asker])

    @property
    def evidence(self):
        return set().union(*(set(group) for group in self.groups)) if self.groups else set()

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    identity: str
    final_ids: tuple[str, ...]
    source_versions: dict[str, str]
    context_sha256: str
    complete: bool
    recalled: int
    support_size: int
    context_tokens: int = 0


def acceptance(generation_keep, generation_candidate, check_keep, check_candidate):

    groups = ((generation_keep, generation_candidate), (check_keep, check_candidate))
    for before, after in groups:
        if (len(before) != 4 or len(after) != 4 or
                [r.identity for r in before] != [r.identity for r in after]):
            return False, 'probe_identity_mismatch'
        for old, new in zip(before, after):
            if old.support_size != new.support_size or new.support_size <= 0:
                return False, 'support_version_mismatch'
            if old.complete and not new.complete:
                return False, 'complete_probe_regressed'
            for mid in set(old.source_versions) & set(new.source_versions):
                if old.source_versions[mid] != new.source_versions[mid]:
                    return False, 'source_version_mismatch'
    if sum(r.complete for r in generation_candidate) <= sum(r.complete for r in generation_keep):
        return False, 'no_generation_coverage_gain'
    if sum(r.complete for r in check_candidate) < sum(r.complete for r in check_keep):
        return False, 'check_coverage_regressed'

    if sum(r.recalled / r.support_size for r in check_candidate) + 1e-12 < sum(
            r.recalled / r.support_size for r in check_keep):
        return False, 'check_recall_regressed'
    if all(old.context_sha256 == new.context_sha256 for before, after in groups
           for old, new in zip(before, after)):
        return False, 'context_unchanged'
    return True, 'verified_coverage_gain'
