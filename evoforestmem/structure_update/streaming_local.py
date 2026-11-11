from collections import OrderedDict
from collections.abc import MutableMapping, MutableSet
from copy import copy, deepcopy
from dataclasses import asdict

from ..forest.tree_types import MemTreeNode


class OverlayMap(MutableMapping):
    def __init__(self, base, copier=None):
        self.base, self.copier = base, copier
        self.changed, self.deleted = {}, set()

    def __getitem__(self, key):
        if key in self.deleted:
            raise KeyError(key)
        if key not in self.changed:
            value = self.base[key]
            if self.copier is None:
                return value
            self.changed[key] = self.copier(value)
        return self.changed[key]

    def __setitem__(self, key, value):
        self.deleted.discard(key)
        self.changed[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self.changed.pop(key, None)
        self.deleted.add(key)

    def __contains__(self, key):
        return key not in self.deleted and (key in self.changed or key in self.base)

    def __iter__(self):
        yield from (key for key in self.base if key not in self.deleted)
        yield from (key for key in self.changed if key not in self.base and key not in self.deleted)

    def __len__(self):
        return len(self.base) - sum(k in self.base for k in self.deleted) + sum(k not in self.base for k in self.changed)

    def publish(self):
        for key in self.deleted:
            self.base.pop(key, None)
        self.base.update(self.changed)
        return self.base


class OverlaySet(MutableSet):
    def __init__(self, base):
        self.base, self.added, self.deleted = base, set(), set()

    def __contains__(self, key):
        return key not in self.deleted and (key in self.base or key in self.added)

    def __iter__(self):
        yield from (key for key in self.base if key not in self.deleted)
        yield from (key for key in self.added if key not in self.base)

    def __len__(self):
        return len(self.base) - sum(k in self.base for k in self.deleted) + sum(k not in self.base for k in self.added)

    def add(self, key):
        self.deleted.discard(key)
        self.added.add(key)

    def discard(self, key):
        self.added.discard(key)
        self.deleted.add(key)

    def update(self, keys):
        for key in keys:
            self.add(key)

    def clear(self):
        self.deleted.update(self.base)
        self.added.clear()


def clone_tree(tree):
    result = copy(tree)
    result.nodes = OverlayMap(tree.nodes, deepcopy)
    result.next_index_by_level = dict(tree.next_index_by_level)
    result.dirty_l0_node_ids = OverlaySet(tree.dirty_l0_node_ids)
    result.dirty_internal_node_ids_by_level = OverlayMap(tree.dirty_internal_node_ids_by_level, OverlaySet)

    return result


def validate_local(original, candidate):
    if (candidate.payload_ids_ordered is not original.payload_ids_ordered or
            candidate.leaf_by_payload is not original.leaf_by_payload):
        raise ValueError('local trial replaced raw membership')
    nodes = candidate.nodes
    for nid in list(nodes.changed):
        n = nodes[nid]
        prior = original.nodes.get(nid)
        if prior is not None and n == prior:
            continue
        if not n.level:
            if prior is None or any(getattr(n, k) != getattr(prior, k) for k in
                    ('child_ids', 'prev_leaf_id', 'next_leaf_id', 'time_start', 'time_end', 'item_count')):
                raise ValueError('local operation altered raw leaf identity/order')
            continue
        if not 1 <= len(n.child_ids) <= candidate.k or len(set(n.child_ids)) != len(n.child_ids):
            raise ValueError('invalid local fan-out')
        children = [nodes[cid] for cid in n.child_ids]
        if any(c.parent_id != nid or c.level != n.level-1 for c in children):
            raise ValueError('invalid local parent/level link')
        if any(a.time_start > b.time_start for a, b in zip(children, children[1:])):
            raise ValueError('local operation reordered children')
        if n.item_count != sum(c.item_count for c in children):
            raise ValueError('local coverage count mismatch')
    root = nodes[candidate.root_node_id]
    if root.parent_id is not None or root.item_count != len(original.payload_ids_ordered):
        raise ValueError('local root coverage mismatch')
    for nid in nodes.deleted:
        parent_id = original.nodes[nid].parent_id
        if parent_id in nodes and nid in nodes[parent_id].child_ids:
            raise ValueError('retired node remains linked')


def tree_patch(tree):

    if not isinstance(tree.nodes, OverlayMap):
        raise TypeError('expected an isolated tree')
    changed = {nid: asdict(n) for nid, n in tree.nodes.changed.items()
               if nid not in tree.nodes.base or asdict(n) != asdict(tree.nodes.base[nid])}
    return {'tree_id': tree.tree_id, 'version': tree.version, 'root': tree.root_node_id,
            'next_index': tree.next_index_by_level, 'nodes': changed, 'retired': sorted(tree.nodes.deleted)}


def apply_tree_patch(tree, patch):
    if tree.tree_id != patch['tree_id']:
        raise ValueError('tree patch identity mismatch')
    for nid in patch['retired']:
        prior = tree.nodes.pop(nid, None)
        if prior is not None:
            tree.dirty_internal_node_ids_by_level.get(prior.level, set()).discard(nid)
    for nid, value in patch['nodes'].items():
        node = MemTreeNode(**deepcopy(value))
        tree.nodes[nid] = node
        if node.level:
            dirty = tree.dirty_internal_node_ids_by_level.setdefault(node.level, set())
            (dirty.add if node.summary_dirty else dirty.discard)(nid)
    tree.version, tree.root_node_id = patch['version'], patch['root']
    tree.next_index_by_level = {int(k): v for k, v in patch['next_index'].items()}
    tree.centroid = None


SUMMARY_MAPS = ('cards', 'requests', '_vectors', '_chunks', '_chunk_texts', '_source_cache',
                '_source_tree_versions', '_source_children', '_journal_versions', '_cache', '_length_failures')


def clone_summary(original, trees):
    from ..forest.hierarchical_summaries import HierarchicalSummaryStore
    result = HierarchicalSummaryStore(client=original.client, count_tokens=original.count_tokens,
        embedder=original.embedder, config=original.config, cache_dir=None,
        vector_lookup=original.vector_lookup)
    for name in SUMMARY_MAPS:
        setattr(result, name, OverlayMap(getattr(original, name)))
    result._revision_inputs = original._revision_inputs
    result._navigation_dependency_bounds = OverlayMap(getattr(original, '_navigation_dependency_bounds', {}))
    result._source_invalid = OverlaySet(original._source_invalid)
    result._forced = OverlaySet(original._forced)
    result._window = original._window
    result.stats = dict(original.stats)
    result.cost_by_view_level = deepcopy(original.cost_by_view_level)
    result._trees = trees
    result._stream_scoped = True
    result.parallelism_file = original.parallelism_file
    return result


def stage_summary(original, trees, delta):
    result = clone_summary(original, trees)
    tid, tree = delta.tree_id, trees[delta.tree_id]
    for nid in delta.retired:
        for name in ('cards', '_source_cache', '_source_children', '_vectors', '_chunks', '_chunk_texts'):
            getattr(result, name).pop((tid, nid), None)
    for nid in (*delta.modified, *delta.ancestors):
        node = tree.nodes[nid]
        result._source_invalid.add((tid, nid))
        if node.level:
            result._forced.add((tid, nid))
            node.summary_dirty = True
            tree.dirty_internal_node_ids_by_level.setdefault(node.level, set()).add(nid)
    return result


def publish_summary(original, trial, trees):
    for name in SUMMARY_MAPS:
        getattr(trial, name).publish()
    original._forced = set(trial._forced)
    original._source_invalid = set(trial._source_invalid)
    original._window = trial._window
    original.stats = trial.stats
    original.cost_by_view_level = trial.cost_by_view_level
    original._recall_cache = OrderedDict()
    original._checked_boundary = False
    original._trees = trees
