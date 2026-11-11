from dataclasses import asdict, dataclass
import math

import numpy as np

from .operations import LocalOperation


@dataclass(frozen=True)
class StreamingConfig:
    partition_penalty: float = .02
    minimum_gain: float = .001
    checkpoint_stride: int = 256
    proposal_policy: str = 'eager_v1'
    evidence_stride: int = 32
    urgent_stride: int = 8
    urgent_gain: float = .10
    citation_repair: str = 'off'
    journal_backend: str = 'files_v1'

    def __post_init__(self):
        if not math.isfinite(self.partition_penalty) or self.partition_penalty < 0:
            raise ValueError('invalid partition penalty')
        if not math.isfinite(self.minimum_gain) or self.minimum_gain < 0:
            raise ValueError('invalid minimum gain')
        if type(self.checkpoint_stride) is not int or self.checkpoint_stride < 1:
            raise ValueError('invalid stream checkpoint stride')
        if self.proposal_policy not in {'eager_v1', 'amortized_v2', 'low_cost_v3', 'utility_v1'}:
            raise ValueError('unknown streaming proposal policy')
        if self.proposal_policy == 'low_cost_v3' and (self.evidence_stride, self.urgent_stride) != (128, 32):
            raise ValueError('low_cost_v3 requires explicit 128/32 evidence strides')
        if self.citation_repair not in {'off', 'bounded_v1'}:
            raise ValueError('unknown streaming citation repair')
        if self.journal_backend not in {'files_v1', 'append_v1', 'append_v2'}:
            raise ValueError('unknown streaming journal backend')
        if self.citation_repair != 'off' and self.proposal_policy == 'eager_v1':
            raise ValueError('citation repair requires amortized_v2')
        if (type(self.evidence_stride) is not int or type(self.urgent_stride) is not int
                or not 1 <= self.urgent_stride <= self.evidence_stride):
            raise ValueError('invalid streaming evidence strides')
        if not math.isfinite(self.urgent_gain) or self.urgent_gain <= self.minimum_gain:
            raise ValueError('urgent gain must exceed minimum gain')

    def identity(self):

        value = asdict(self)
        if self.proposal_policy == 'eager_v1':
            value = {k: value[k] for k in ('partition_penalty', 'minimum_gain', 'checkpoint_stride')}
            if self.journal_backend != 'files_v1':
                value['journal_backend'] = self.journal_backend
            return value
        if self.citation_repair == 'off':
            value.pop('citation_repair')
        if self.journal_backend == 'files_v1':
            value.pop('journal_backend')
        return value


def admission(config, seq, last_trial_seq, candidates):

    evidence = seq - last_trial_seq
    gain = candidates[0][1]['gain'] if candidates else 0.
    urgent = gain >= config.urgent_gain
    required = config.urgent_stride if urgent else config.evidence_stride
    admitted = bool(candidates) and evidence >= required
    return {'policy': config.proposal_policy, 'new_messages': evidence,
            'required_messages': required, 'best_gain': gain, 'urgent': urgent,
            'admitted': admitted, 'consumed': admitted,
            'reason': ('no_local_gain' if not candidates else
                       'significant_change' if admitted and urgent else
                       'evidence_accumulated' if admitted else 'insufficient_new_evidence')}


class LocalStatistics:

    def __init__(self, vector):
        self.vector, self.values = vector, {}
        self.leaf_reads = 0

    def invalidate(self, trees, insertions):
        anchors = set()
        for tid, leaf in insertions:
            tree = trees[tid]
            for nid in tree.path_to_root(leaf):
                self.values.pop((tid, nid), None)
                node = tree.nodes[nid]
                if node.level in (1, 2):
                    anchors.add((tid, nid))
                parent = tree.nodes.get(node.parent_id)
                if parent is not None:
                    index = parent.child_ids.index(nid)
                    for adjacent in parent.child_ids[max(0, index-1):index+2]:

                        self.values.pop((tid, adjacent), None)
                        if node.level in (1, 2):
                            anchors.add((tid, adjacent))
        return anchors

    def changed(self, delta):
        for nid in (*delta.modified, *delta.retired, *delta.ancestors):
            self.values.pop((delta.tree_id, nid), None)

    def get(self, tree, nid):
        key = tree.tree_id, nid
        node = tree.nodes[nid]
        children = tuple(node.child_ids)
        prior = self.values.get(key)
        if prior is not None and prior[0] == children:
            return prior[1], prior[2]
        if node.level == 0:
            value = np.asarray(self.vector(node.child_ids[0]), dtype=np.float64)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError('stream policy requires finite document vectors')
            norm = float(np.linalg.norm(value))
            total, count = value / max(norm, 1e-12), 1
            self.leaf_reads += 1
        else:
            parts = [self.get(tree, cid) for cid in children]
            count = sum(n for n, _ in parts)
            total = sum((v for _, v in parts), np.zeros_like(parts[0][1]))
        self.values[key] = children, count, total
        return count, total


def score(parts, penalty):

    count = sum(n for n, _ in parts)
    return sum(float(np.linalg.norm(v)) for _, v in parts) / count - penalty * len(parts)


def propose_local(trees, anchors, statistics, config, limit=2):
    found = {}

    def consider(tree, op, groups_before, groups_after):
        def aggregate(ids):
            rows = [statistics.get(tree, cid) for cid in ids]
            return sum(n for n, _ in rows), sum((v for _, v in rows), np.zeros_like(rows[0][1]))
        before = score([aggregate(g) for g in groups_before], config.partition_penalty)
        after = score([aggregate(g) for g in groups_after], config.partition_penalty)
        gain = after - before
        if gain > config.minimum_gain:
            found[op.identity] = op, {'before': before, 'after': after, 'gain': gain}

    for tid, nid in sorted(anchors):
        tree = trees[tid]
        node = tree.nodes.get(nid)
        if node is None or node.level not in (1, 2):
            continue
        children = node.child_ids
        for cut in range(2, len(children)-1):
            consider(tree, LocalOperation('split', tid, nid, cut=cut), [children],
                     [children[:cut], children[cut:]])
        parent = tree.nodes.get(node.parent_id)
        if parent is None:
            continue
        index = parent.child_ids.index(nid)
        if index + 1 == len(parent.child_ids):
            continue
        right = tree.nodes[parent.child_ids[index+1]]
        if right.level != node.level:
            continue
        both = children + right.child_ids
        before = [children, right.child_ids]
        if len(both) <= tree.k:
            consider(tree, LocalOperation('merge', tid, nid, right.node_id), before, [both])
        for cut in range(max(2, len(both)-tree.k), min(tree.k, len(both)-2)+1):
            if cut != len(children):
                consider(tree, LocalOperation('boundary', tid, nid, right.node_id, cut),
                         before, [both[:cut], both[cut:]])
    return sorted(found.values(), key=lambda row: (-row[1]['gain'], row[0].identity))[:limit]


def propose_prefix(trees, anchors, statistics, config, limit=2):

    found = {}
    tolerance = 1e-10

    def scan(tree, children):
        parts = [statistics.get(tree, cid) for cid in children]
        counts = np.cumsum([0, *[n for n, _ in parts]])
        sums = np.vstack((np.zeros_like(parts[0][1]), np.cumsum([v for _, v in parts], axis=0)))
        def part(start, stop):
            return int(counts[stop] - counts[start]), sums[stop] - sums[start]
        return part

    def consider(tree, op, children, part, before_ranges, after_ranges):
        before = score([part(a, b) for a, b in before_ranges], config.partition_penalty)
        after = score([part(a, b) for a, b in after_ranges], config.partition_penalty)
        gain = after - before
        if config.proposal_policy == 'utility_v1' or gain > config.minimum_gain - tolerance:
            found[op.identity] = (op, gain, tree, children, before_ranges, after_ranges)

    for tid, nid in sorted(anchors):
        tree = trees[tid]
        node = tree.nodes.get(nid)
        if node is None or node.level not in (1, 2):
            continue
        children = node.child_ids
        part = scan(tree, children)
        end = len(children)
        for cut in range(2, end-1):
            consider(tree, LocalOperation('split', tid, nid, cut=cut), children, part,
                     [(0, end)], [(0, cut), (cut, end)])
        parent = tree.nodes.get(node.parent_id)
        if parent is None:
            continue
        index = parent.child_ids.index(nid)
        if index + 1 == len(parent.child_ids):
            continue
        right = tree.nodes[parent.child_ids[index+1]]
        if right.level != node.level:
            continue
        both = children + right.child_ids
        part, end = scan(tree, both), len(both)
        before = [(0, len(children)), (len(children), end)]
        if end <= tree.k:
            consider(tree, LocalOperation('merge', tid, nid, right.node_id), both, part, before, [(0, end)])
        for cut in range(max(2, end-tree.k), min(tree.k, end-2)+1):
            if cut != len(children):
                consider(tree, LocalOperation('boundary', tid, nid, right.node_id, cut),
                         both, part, before, [(0, cut), (cut, end)])
    ranked = sorted(found.values(), key=lambda row: (-row[1], row[0].identity))
    if not ranked:
        return []
    cutoff = ranked[min(limit, len(ranked))-1][1] - 2*tolerance
    exact = []
    for op, gain, tree, children, before_ranges, after_ranges in ranked:
        if gain < cutoff:
            break
        def aggregate(bounds):
            start, stop = bounds
            parts = [statistics.get(tree, cid) for cid in children[start:stop]]
            return sum(n for n, _ in parts), sum((v for _, v in parts), np.zeros_like(parts[0][1]))
        before = score([aggregate(bounds) for bounds in before_ranges], config.partition_penalty)
        after = score([aggregate(bounds) for bounds in after_ranges], config.partition_penalty)
        if config.proposal_policy == 'utility_v1' or after - before > config.minimum_gain:
            exact.append((op, {'before': before, 'after': after, 'gain': after-before}))
    return sorted(exact, key=lambda row: (-row[1]['gain'], row[0].identity))[:limit]
