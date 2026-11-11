from dataclasses import dataclass
import math
import re
import numpy as np


@dataclass(frozen=True)
class AssociationConfig:
    mode: str = 'off'
    navigation: str = 'off'
    seed: int = 1729
    family_similarity: float = .95
    query_similarity: float = .80
    max_outgoing: int = 4
    max_candidates: int = 20000
    max_active: int = 10000
    witness_limit: int = 8
    trigger_limit: int = 4
    trial_limit: int = 4
    weak_weight: float = .25
    group_weight: float = .5
    learning_policy: str = 'utility_v1'
    cooccurrence_window: int = 128
    shortcut_work_price: float = .1
    shortcut_token_price: float = .01

    def __post_init__(self):
        if self.learning_policy not in {'utility_v1', 'shortcut_v2', 'evidence_v3', 'associative_recall_v4'}:
            raise ValueError('unknown association learning policy')
        if type(self.cooccurrence_window) is not int or self.cooccurrence_window < 2:
            raise ValueError('cooccurrence window must contain at least two families')
        if any(not math.isfinite(x) or x < 0 for x in
               (self.shortcut_work_price, self.shortcut_token_price)):
            raise ValueError('invalid shortcut resource price')
        if self.navigation not in {'off', 'one_hop_v1', 'recall_v4'}:
            raise ValueError('unknown association navigation')
        if (self.learning_policy == 'associative_recall_v4') != (self.navigation == 'recall_v4'):
            raise ValueError('associative_recall_v4 requires recall_v4 navigation')
        if self.navigation != 'off' and self.mode == 'off':
            raise ValueError('association navigation requires an enabled graph')
        if self.mode not in {'off', 'shadow', 'verified_v1'}:
            raise ValueError('unknown association mode')
        for name in ('max_outgoing', 'max_candidates', 'max_active', 'witness_limit', 'trigger_limit', 'trial_limit'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('association limits must be positive integers')
        if self.trigger_limit > 4 or self.trial_limit > 4 or self.max_outgoing > 4:
            raise ValueError('association v1 limits exceeded')
        if not 0 < self.query_similarity <= self.family_similarity <= 1:
            raise ValueError('invalid similarity thresholds')
        if any(not math.isfinite(x) or x < 0 for x in (self.weak_weight, self.group_weight)):
            raise ValueError('invalid association reward')


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    value = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return max(-1., min(1., value)) if math.isfinite(value) else 0.


def normalized(question):
    return ' '.join(re.findall(r'\w+', question.casefold()))


def bindings(spec):
    return {k: sorted(v) if isinstance(v, (list, tuple)) else v
            for k in ('asker', 'channels', 'people', 'artifacts', 'as_of_date')
            if (v := getattr(spec, k, None))}


def compatible(a, b):
    for key in a.keys() & b.keys():
        x, y = a[key], b[key]
        if isinstance(x, list) and isinstance(y, list):
            if not set(x) & set(y):
                return False
        elif x != y:
            return False
    return True


def query_bindings(spec, render_names=None):

    result = bindings(spec)
    text = ' ' + normalized(spec.question) + ' '
    if render_names:
        for kind, names in zip(('channels', 'people'), render_names):
            found = sorted(key for key, label in names.items() if len(normalized(label)) >= 4
                           and ' ' + normalized(label) + ' ' in text)
            if found:
                result.setdefault(kind, found)
    dates = sorted(set(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', spec.question)))
    if dates:
        result['explicit_dates'] = dates
    return result
