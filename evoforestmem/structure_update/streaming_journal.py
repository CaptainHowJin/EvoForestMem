from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .contracts import fingerprint
from .io import load_object, save_object, write_json


SCHEMA = 'stream-verified-v1'


class StreamJournal:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / 'records.wal').exists():
            raise ValueError('cannot mix append and file stream journals')
        self.put('identity', {'schema': SCHEMA, **identity})

    def get(self, name):
        path = self.directory / (name + '.json')
        if not path.exists():
            return None
        envelope = json.loads(path.read_text())
        if envelope['sha256'] != fingerprint(envelope['value']):
            raise ValueError('stream journal checksum mismatch: ' + name)
        return envelope['value']

    def put(self, name, value):
        prior = self.get(name)
        if prior is not None:
            if fingerprint(prior) != fingerprint(value):
                raise ValueError('cannot overwrite a durable stream record: ' + name)
            return
        write_json(self.directory / (name + '.json'), {'sha256': fingerprint(value), 'value': value})

    def object(self, value):
        return save_object(self.directory / 'objects', value)

    def names(self, directory, prefix=''):
        return [str(p.relative_to(self.directory).with_suffix(''))
                for p in sorted((self.directory / directory).glob(prefix + '*.json'))]

    def close(self):
        pass

    def step(self, seq, phase):
        return f'messages/{seq:012d}/{phase}'

    def read_step(self, seq, phase):
        return self.get(self.step(seq, phase))

    def write_step(self, seq, phase, value):
        return self.put(self.step(seq, phase), value)


def summary_patch(store):

    result = {}
    for name in ('cards', '_vectors', '_chunks', '_chunk_texts', 'requests'):
        mapping = getattr(store, name)
        rows = []
        for key, value in mapping.changed.items():
            if name == 'cards':
                value = asdict(value)
            elif name == '_vectors':
                value = np.asarray(value).tolist()
            rows.append([key, value])
        result[name] = {'set': rows, 'delete': list(mapping.deleted)}
    result['forced_add'] = sorted(store._forced.added)
    result['forced_delete'] = sorted(store._forced.deleted)
    result['window'] = store._window
    result['stats'] = store.stats
    result['cost_by_view_level'] = dict(store.cost_by_view_level)
    return result


def apply_summary_patch(store, patch, trees):
    from collections import Counter, OrderedDict, defaultdict
    from ..forest.hierarchical_summaries import HierarchicalSummary
    key_of = lambda key: tuple(key) if isinstance(key, list) else key
    for name in ('cards', '_vectors', '_chunks', '_chunk_texts', 'requests'):
        mapping = getattr(store, name)
        for key in patch[name]['delete']:
            mapping.pop(key_of(key), None)
        for key, value in patch[name]['set']:
            key = key_of(key)
            if name == 'cards':
                value = dict(value)
                for field in ('input_ids', 'dependencies', 'citations', 'structural_children', 'pending_input_ids'):
                    value[field] = tuple(value[field])
                value = HierarchicalSummary(**value)
            elif name == '_vectors':
                value = np.asarray(value, dtype=np.float32)
            elif name == '_chunks':
                value = [tuple(row) for row in value]
            elif name == '_chunk_texts':
                value = tuple(value)
            mapping[key] = value
    store._forced.update(map(tuple, patch['forced_add']))
    store._forced.difference_update(map(tuple, patch['forced_delete']))
    store._window = patch['window']
    store.stats = dict(patch['stats'])
    store.cost_by_view_level = defaultdict(Counter, {key: Counter(v) for key, v in patch['cost_by_view_level'].items()})

    for key, _ in patch['cards']['set']:
        store._source_invalid.add(tuple(key))
    for key in patch['cards']['delete']:
        store._source_cache.pop(tuple(key), None)
        store._source_children.pop(tuple(key), None)
    store._recall_cache = OrderedDict()
    store._checked_boundary = False
    store._trees = trees


def guard(before, after):

    if len(before) != len(after) or [p.identity for p in before] != [p.identity for p in after]:
        return False, 'witness_identity_mismatch'
    for a, b in zip(before, after):
        if a.demands.keys() != b.demands.keys():
            return False, 'demand_identity_mismatch'
        for key, x in a.demands.items():
            y = b.demands[key]
            if x['signature'] != y['signature']:
                return False, 'demand_version_mismatch'
            if x['complete'] and not y['complete']:
                return False, 'satisfied_demand_regressed'
        if any(a.source_versions[mid] != b.source_versions[mid]
               for mid in a.source_versions.keys() & b.source_versions.keys()):
            return False, 'source_version_mismatch'
    if sum(p.recall for p in after) + 1e-12 < sum(p.recall for p in before):
        return False, 'support_recall_regressed'
    return True, 'local_only_no_history' if not before else 'local_and_history_verified'
