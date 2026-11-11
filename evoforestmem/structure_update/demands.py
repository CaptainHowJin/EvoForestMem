from dataclasses import asdict, dataclass, field

from .contracts import ProbeResult, fingerprint


@dataclass
class DemandWitness:
    question: str
    ask_time: float
    asker: str | None
    sources: dict
    requirements: list
    quotes: list
    request_keys: list
    final_ids: list
    verified_at: float

    @property
    def identity(self):
        return fingerprint([self.question.strip(), self.ask_time, self.asker])

    @property
    def groups(self):
        return [g for r in self.requirements if r['status'] == 'supported' for g in r['groups']]

    @property
    def evidence(self):
        return set().union(*map(set, self.groups)) if self.groups else set()

    def to_dict(self):
        return asdict(self)


def coverage(witness, selected):
    selected = set(selected)
    return {r['key']: {'complete': any(set(g) <= selected for g in r['groups']),
                       'recall': max(len(set(g) & selected) / len(g) for g in r['groups']),
                       'signature': fingerprint(r)}
            for r in witness.requirements if r['status'] == 'supported' and r['groups']}


@dataclass(frozen=True)
class DemandProbe(ProbeResult):
    demands: dict = field(default_factory=dict)
    unknown_demands: int = 0
    trace: dict = field(default_factory=dict)

    @property
    def coverage(self):
        return sum(r['complete'] for r in self.demands.values()) / max(1, len(self.demands))

    @property
    def recall(self):
        return sum(r['recall'] for r in self.demands.values()) / max(1, len(self.demands))


def demand_acceptance(before, after, checks_before, checks_after):
    pairs = ((before, after), (checks_before, checks_after))
    for old, new in pairs:
        if len(old) != 4 or len(new) != 4 or [r.identity for r in old] != [r.identity for r in new]:
            return False, 'probe_identity_mismatch'
        for a, b in zip(old, new):
            if not a.demands or a.demands.keys() != b.demands.keys():
                return False, 'demand_identity_mismatch'
            for key in a.demands:
                x, y = a.demands[key], b.demands[key]
                if x['signature'] != y['signature']:
                    return False, 'support_version_mismatch'
                if x['complete'] and not y['complete']:
                    return False, 'complete_demand_regressed'
            if any(a.source_versions[mid] != b.source_versions[mid]
                   for mid in a.source_versions.keys() & b.source_versions.keys()):
                return False, 'source_version_mismatch'
    if sum(r.coverage for r in after) <= sum(r.coverage for r in before) + 1e-12:
        return False, 'no_generation_demand_gain'
    if sum(r.coverage for r in checks_after) + 1e-12 < sum(r.coverage for r in checks_before):
        return False, 'check_coverage_regressed'
    if sum(r.recall for r in checks_after) + 1e-12 < sum(r.recall for r in checks_before):
        return False, 'check_recall_regressed'
    if all(a.context_sha256 == b.context_sha256 for old, new in pairs for a, b in zip(old, new)):
        return False, 'context_unchanged'
    return True, 'verified_demand_coverage_gain'
