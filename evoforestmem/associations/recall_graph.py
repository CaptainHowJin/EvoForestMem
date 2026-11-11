from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import math
import time

import numpy as np

from .contracts import normalized
from ..routers.incremental_artifact import IDENTIFIER, URL
from ..structure_update.contracts import fingerprint
from ..structure_update.feedback import applicable, source_version

POLICY = 'associative_recall_v4'
SCHEMA = 'associative-recall-graph-v4'


def unit_vector(value):
    v = np.asarray(value, dtype=np.float32)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError('finite one-dimensional source embedding required')
    return v / max(float(np.linalg.norm(v)), 1e-12)


def reference_keys(message):


    keys = {'url:' + s.rstrip('.,;') for s in URL.findall(message.text)}
    keys.update(message.channel_key + ':' + s.lower() for s in IDENTIFIER.findall(message.text)
                if any(c.isdigit() for c in s) or '_' in s or s.isupper())
    return keys


class RecallGraph:
    def __init__(self, config, vector_lookup):
        self.config, self.vector_lookup = config, vector_lookup
        self.records, self.statistics, self.type_statistics = {}, {}, {}
        self.events, self.seen_feedback = [], set()
        self.version, self.seconds, self.boundary = 0, 0., None
        self._units, self._by_source, self._adjacency = {}, {}, {}
        self._vectors, self._versions, self._signatures = {}, {}, {}

    def vector(self, mid):
        if mid not in self._vectors:
            self._vectors[mid] = unit_vector(self.vector_lookup(mid))
        return self._vectors[mid]

    def probability(self, row):
        prior = self.type_statistics.get(row['kind'], {'success': 0., 'exposures': 0})
        p = (1. + prior['success']) / (2. + prior['exposures'])
        own = self.statistics.get(row['id'], {'success': 0., 'exposures': 0})
        return (own['success'] + 2. * p) / (own['exposures'] + 2.)

    def strength(self, row):
        return row['prior'] * self.probability(row)

    def __deepcopy__(self, memo):


        other = type(self)(self.config, self.vector_lookup)
        memo[id(self)] = other
        other.restore(self.state())
        return other

    def rebind(self, trees, placement, log, ledger, known_at):
        tick = time.perf_counter()
        if self.boundary is not None and known_at < self.boundary:
            raise ValueError('association writer cannot move backwards')
        legal = {m.msg_id: m for m in log if applicable(log, ledger, m.msg_id, known_at)}
        versions = {m: source_version(v) for m, v in legal.items()}
        for mid in self._versions:
            if versions.get(mid) != self._versions[mid]:
                self._vectors.pop(mid, None)
        self._versions = versions
        units, by_source = {}, {}
        for tid, tree in sorted(trees.items()):
            if tree.view != 'channel' or tree.deactivated:
                continue
            for nid, node in sorted(tree.nodes.items()):
                singleton = node.level == 0 and nid == tree.root_node_id
                if node.level != 1 and not singleton:
                    continue
                members = node.child_ids if singleton else [m for leaf in node.child_ids for m in tree.nodes[leaf].child_ids]
                ids = [m for m in members if m in legal]
                if not ids:
                    continue
                key = fingerprint([tid, nid])
                vector = unit_vector(np.sum([self.vector(m) for m in ids], axis=0))
                units[key] = dict(tree=tid, node=nid, members=ids, vector=vector,
                    signature=fingerprint([(m, versions[m]) for m in ids]))
                by_source.update({m: key for m in ids})
        self._units, self._by_source = units, by_source
        changed = {k for k, u in units.items() if self._signatures.get(k) != u['signature']}
        formed, invalidated = 0, 0

        def add(a, b, kind, prior):
            nonlocal formed
            if a == b or a not in by_source or b not in by_source or by_source[a] == by_source[b]:
                return
            key = fingerprint([POLICY, a, b, kind, versions[a], versions[b]])
            if key not in self.records:
                self.records[key] = dict(id=key, source=a, target=b, kind=kind,
                    versions={a: versions[a], b: versions[b]}, prior=float(prior),
                    created_at=known_at, valid=True)
                formed += 1


        keys = sorted(units)
        matrix = np.asarray([units[k]['vector'] for k in keys])
        for key in sorted(changed):
            scores = matrix @ units[key]['vector']
            order = sorted(range(len(keys)), key=lambda i: (-float(scores[i]), keys[i]))
            count = 0
            for i in order:
                other = keys[i]
                if other == key or scores[i] <= 0:
                    continue
                a, b = units[key]['members'][0], units[other]['members'][0]
                add(a, b, 'semantic', scores[i]); add(b, a, 'semantic', scores[i])
                count += 1
                if count == self.config.max_outgoing:
                    break


        references = defaultdict(list)
        for mid, message in legal.items():
            if mid not in by_source:
                continue
            if message.reply_to in by_source:
                add(mid, message.reply_to, 'reply', 1.)
                add(message.reply_to, mid, 'reply', 1.)
            for reference in reference_keys(message):
                references[reference].append(mid)
        for ids in references.values():
            ids.sort(key=lambda m: (legal[m].ts, legal[m].seq, m))
            previous = None
            for mid in ids:
                if previous is not None and by_source[previous] != by_source[mid]:
                    add(previous, mid, 'reference', 1.); add(mid, previous, 'reference', 1.)
                previous = mid
        for row in self.records.values():
            valid = all(versions.get(m) == v and m in by_source for m, v in row['versions'].items())
            if row['valid'] and not valid:
                invalidated += 1
            row['valid'] = valid


        if len(self.records) > self.config.max_candidates:
            ordered = sorted(self.records, key=lambda k: (not self.records[k]['valid'],
                -self.strength(self.records[k]), -self.records[k]['created_at'], k))
            dropped = ordered[self.config.max_candidates:]
            self.events.append(dict(op='capacity_eviction', known_at=known_at, ids=dropped))
            self.records = {k: self.records[k] for k in ordered[:self.config.max_candidates]}
        self._bind_adjacency()
        self._signatures = {k: u['signature'] for k, u in units.items()}
        self.boundary, self.version = known_at, self.version + 1
        elapsed = time.perf_counter() - tick
        self.seconds += elapsed
        self.events.append(dict(op='ingest', known_at=known_at, formed=formed, invalidated=invalidated,
            units=len(units), changed_units=len(changed), routable=sum(map(len, self._adjacency.values())), seconds=elapsed))
        return self.events[-1]

    def _bind_adjacency(self):
        choices = defaultdict(dict)
        for row in self.records.values():
            if not row['valid']:
                continue
            a, b = self._by_source.get(row['source']), self._by_source.get(row['target'])
            if a is None or b is None or a == b:
                continue
            previous = choices[a].get(b)
            if previous is None or (self.strength(row), row['id']) > (self.strength(previous), previous['id']):
                choices[a][b] = row
        self._adjacency = {a: sorted(rows.values(), key=lambda r: (-self.strength(r), r['id']))
                           [:self.config.max_outgoing] for a, rows in choices.items()}

    def candidates(self, reader, nav, base_ids, known_at, effective_at):

        def legal(mid):
            return (nav.eligible(mid) and applicable(reader.log, reader.validity, mid, known_at)
                    and applicable(reader.log, reader.validity, mid, effective_at))
        seeds = {}
        for rank, mid in enumerate(dict.fromkeys([*base_ids, *nav.direct]), 1):
            key = self._by_source.get(mid)
            if key and legal(mid):
                seeds[key] = max(seeds.get(key, 0.), 1. / (60 + rank))
        visited = {(tid, nid) for tid, nid, level, _ in nav.paths if level == 1}
        for key, unit in self._units.items():
            if (unit['tree'], unit['node']) in visited and any(legal(m) for m in unit['members']):
                seeds.setdefault(key, 1. / (60 + len(seeds) + 1))
        seeds = dict(sorted(seeds.items(), key=lambda p: (-p[1], p[0]))
                     [:min(32, reader.config.navigation_nodes)])
        nodes, frontier, transitions, paths = set(seeds), set(seeds), {}, {k: [] for k in seeds}
        considered, traversed = set(), set()
        for _ in range(2):
            following = set()
            for a in sorted(frontier, key=lambda k: (-seeds.get(k, 0.), k)):
                for row in self._adjacency.get(a, ()):
                    considered.add(row['id'])
                    if row['created_at'] > known_at or not all(legal(m) and
                            source_version(reader.log.get(m)) == v for m, v in row['versions'].items()):
                        continue
                    b = self._by_source[row['target']]
                    if b not in nodes and len(nodes) >= reader.config.navigation_nodes:
                        continue
                    transitions.setdefault(a, {})[b] = row
                    traversed.add(row['id'])
                    if b not in nodes:
                        paths[b] = [*paths[a], row['id']]
                        following.add(b); nodes.add(b)
                    elif not paths.get(b):
                        paths[b] = [*paths[a], row['id']][:2]
            frontier = following
        detail = dict(formed=len(self.records), seed_units=len(seeds), reachable_units=len(nodes),
            considered_edges=len(considered), traversed_edges=len(traversed),
            triggered=sorted(traversed), restart=.5, hops=2, iterations=20)
        if not seeds or not traversed:
            return [], {}, detail
        reset = {k: v / sum(seeds.values()) for k, v in seeds.items()}
        rank = {k: reset.get(k, 0.) for k in sorted(nodes)}
        for _ in range(20):
            nxt = {k: .5 * reset.get(k, 0.) for k in sorted(nodes)}
            dangling = 0.
            for a, value in rank.items():
                targets = transitions.get(a, {})
                if not targets:
                    dangling += .5 * value
                    continue
                total = sum(self.strength(r) for r in targets.values())
                for b, row in targets.items():
                    nxt[b] += .5 * value * self.strength(row) / total
            for k, value in reset.items():
                nxt[k] += dangling * value
            rank = nxt
        old, scores, provenance = set(base_ids), {}, {}


        for key in sorted(nodes, key=lambda k: (-rank[k]/max(1,len(self._units[k]['members'])), k)):
            if not paths.get(key):
                continue
            for mid in self._units[key]['members']:
                if mid in old or not legal(mid):
                    continue
                scores[mid] = rank[key] / max(1, len(self._units[key]['members']))
                provenance[mid] = list(dict.fromkeys(paths[key]))
                if len(scores) >= reader.config.navigation_sources:
                    break
            if len(scores) >= reader.config.navigation_sources:
                break

        dense = reader.dense.score_subset(nav.qvec, scores) if nav.qvec is not None and reader.dense else {}
        rankings = [sorted(scores, key=lambda m: (-scores[m], m))]
        if dense:
            rankings.append(sorted(dense, key=lambda m: (-dense[m], m)))
        fused = defaultdict(float)
        for order in rankings:
            for i, mid in enumerate(order, 1):
                fused[mid] += 1. / (60 + i)
        selected = sorted(fused, key=lambda m: (-fused[m], m))[:8]
        detail.update(novel_pool=len(scores), novel_sources=selected)
        return selected, {m: provenance[m] for m in selected}, detail

    def feedback(self, question, bindings, source_versions, outcomes, known_at):
        family = fingerprint([normalized(question), bindings, source_versions])
        if family in self.seen_feedback:
            return False
        if not outcomes:
            return False
        self.seen_feedback.add(family)
        by_kind = defaultdict(list)
        for key, value in outcomes.items():
            if key not in self.records or value is None:
                continue
            value = max(0., min(1., float(value)))
            kind = self.records[key]['kind']
            by_kind[kind].append(value)
            stat = self.statistics.setdefault(key, dict(success=0., exposures=0, available_at=known_at))
            stat['success'] += value; stat['exposures'] += 1; stat['available_at'] = known_at
        for kind, values in by_kind.items():
            stat = self.type_statistics.setdefault(kind, dict(success=0., exposures=0, available_at=known_at))
            stat['success'] += sum(values)/len(values); stat['exposures'] += 1; stat['available_at'] = known_at
        self._bind_adjacency()
        self.version += 1
        return True

    def state(self):
        return dict(schema=SCHEMA, config=asdict(self.config), records=deepcopy(self.records),
            statistics=deepcopy(self.statistics), type_statistics=deepcopy(self.type_statistics),
            events=deepcopy(self.events), seen_feedback=sorted(self.seen_feedback), version=self.version,
            seconds=self.seconds, boundary=self.boundary,
            signatures=deepcopy(self._signatures))

    def restore(self, value):
        if value.get('schema') != SCHEMA or value['config'] != asdict(self.config):
            raise ValueError('fresh associative recall v4 state required')
        for name in ('records', 'statistics', 'type_statistics', 'events', 'version', 'seconds', 'boundary'):
            setattr(self, name, deepcopy(value[name]))
        self.seen_feedback = set(value['seen_feedback'])
        self._signatures = deepcopy(value['signatures'])
        self._units, self._by_source, self._adjacency, self._vectors, self._versions = {}, {}, {}, {}, {}
