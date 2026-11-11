from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .contracts import fingerprint


@dataclass(frozen=True)
class OnlineUtilityConfig:
    mode: str = 'off'
    token_reference: float = .05
    maintenance_seconds_per_message: float = .05
    feedback_seconds_per_query: float = 5.
    initial_price: float = .01
    price_step: float = .1
    exploration: float = .5

    def __post_init__(self):
        if self.mode not in {'off', 'shadow', 'online'}:
            raise ValueError('unknown online utility mode')
        for name in ('token_reference', 'maintenance_seconds_per_message',
                     'feedback_seconds_per_query', 'initial_price', 'price_step', 'exploration'):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError('invalid utility resource parameter: ' + name)


def evidence_utility(witness, ids):
    ids = set(ids)
    return sum(r['status'] == 'supported' and any(set(g) <= ids for g in r['groups'])
               for r in witness.requirements) / max(1, len(witness.requirements))


def probe_utility(probe):
    return sum(r['complete'] for r in probe.demands.values()) / max(
        1, len(probe.demands) + probe.unknown_demands)


def paired_gain(before, after):
    if not before or len(before) != len(after):
        raise ValueError('paired probes must be nonempty and aligned')
    for a, b in zip(before, after):
        if (a.identity != b.identity or a.unknown_demands != b.unknown_demands
                or a.demands.keys() != b.demands.keys()
                or any(a.demands[k]['signature'] != b.demands[k]['signature'] for k in a.demands)
                or any(a.source_versions[m] != b.source_versions[m]
                       for m in a.source_versions.keys() & b.source_versions.keys())):
            raise ValueError('paired snapshot/requirement mismatch')
    return sum(probe_utility(b) - probe_utility(a) for a, b in zip(before, after)) / len(before)


def paired_acceptance(before, after, checks_before, checks_after):
    gain, check = paired_gain(before, after), paired_gain(checks_before, checks_after)
    for old, new in ((before, after), (checks_before, checks_after)):
        if any(a.demands[k]['complete'] and not b.demands[k]['complete']
               for a, b in zip(old, new) for k in a.demands):
            return False, 'verified_requirement_regressed', gain, check
    if gain <= 0 or check < 0:
        return False, 'no_independently_checked_utility_gain', gain, check
    return True, 'paired_evidence_utility_gain', gain, check


class UtilityLedger:

    def __init__(self, config=None):
        self.config = config or OnlineUtilityConfig()
        self.statistics, self.observations = {}, {}
        self.seconds = 0.

    def score(self, action, dependency='', *, optimistic=True, known_at=None):
        row = self.statistics.get(fingerprint([action, dependency]), {})
        if known_at is not None and row.get('available_at', float('-inf')) > known_at:
            row = {}
        n = row.get('n', 0)
        mean = row.get('mean', 0.)

        bonus = self.config.exploration * math.sqrt(
            2 * math.log(2 + len(self.observations)) / max(1, n)) if optimistic else 0.
        cost = row.get('mean_seconds', 0.) / self.config.maintenance_seconds_per_message
        return mean + bonus - self.config.initial_price * cost

    def observe(self, *, action, dependency, query_ids, prefix, known_at, available_at,
                before_ids, after_ids, gain, seconds):
        if (not all(math.isfinite(x) for x in (known_at, available_at, gain, seconds))
                or available_at < known_at or not -1 <= gain <= 1 or seconds < 0):
            raise ValueError('invalid or noncausal utility observation')
        key = fingerprint([action, dependency, query_ids, prefix, known_at, before_ids, after_ids])
        row = dict(action=action, dependency=dependency, query_ids=list(query_ids), prefix=prefix,
                   known_at=known_at, available_at=available_at, before_ids=before_ids,
                   after_ids=after_ids, gain=gain, seconds=seconds)
        if key in self.observations:
            old = self.observations[key]
            if old['gain'] != gain:
                raise ValueError('conflicting duplicate utility observation')
            return False
        self.observations[key] = row
        stat = self.statistics.setdefault(fingerprint([action, dependency]),
            dict(n=0, mean=0., m2=0., mean_seconds=0., available_at=available_at))
        stat['n'] += 1
        delta = gain - stat['mean']
        stat['mean'] += delta / stat['n']
        stat['m2'] += delta * (gain - stat['mean'])
        stat['mean_seconds'] += (seconds - stat['mean_seconds']) / stat['n']
        stat['available_at'] = max(stat['available_at'], available_at)
        self.seconds += seconds
        return True

    def state(self):
        return deepcopy(dict(schema='online-utility-v1', config=asdict(self.config),
                             statistics=self.statistics, observations=self.observations, seconds=self.seconds))

    def restore(self, state):
        if state.get('schema') != 'online-utility-v1' or state['config'] != asdict(self.config):
            raise ValueError('online utility checkpoint configuration mismatch')
        self.statistics, self.observations = deepcopy(state['statistics']), deepcopy(state['observations'])
        self.seconds = state['seconds']


def graph_dependency(record):
    return fingerprint([record['versions'], record['bindings']])


def resource_price(config, spent, reference):


    pressure = spent / max(1., reference) - 1.
    return config.initial_price * math.exp(max(-20., min(20., config.price_step * pressure)))


def recent_legal_witnesses(history, log, ledger, known_at, limit=8):

    from .feedback import applicable, source_version
    selected = []
    for original in reversed(history):
        if original.ask_time > known_at or original.verified_at > known_at:
            continue
        w = deepcopy(original)
        for r in w.requirements:
            r['groups'] = [g for g in r['groups'] if all(m in log and
                source_version(log.get(m)) == w.sources.get(m) and
                applicable(log, ledger, m, w.ask_time) and applicable(log, ledger, m, known_at) for m in g)]
            if r['status'] == 'supported' and not r['groups']:
                r['status'] = 'uncertain'
        if w.groups:
            selected.append(w)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda w: (w.verified_at, w.identity))
