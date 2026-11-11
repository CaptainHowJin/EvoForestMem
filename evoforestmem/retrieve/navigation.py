from collections import Counter
from copy import copy
from dataclasses import replace
import heapq
import time
import numpy as np
from ..index.sparse import SparseIndex
from .fusion import Fused, Signal, fuse
from .s0_analyse import QuerySpec
from .s4_assemble import Span

class Navigator:

    def __init__(self, reader, question, qvec, spec, cutoff, period):
        self.reader, self.question, self.qvec = (reader, question, qvec)
        self.trees, self.store = (reader.trees, reader.routing_summaries)
        self.views = tuple((v for v in reader.views if spec.in_scope(v)))
        self.cutoff, self.period = (cutoff, period)
        self.node_limit, self.source_limit = (reader.config.navigation_nodes, reader.config.navigation_sources)
        self.sources, self.origins, self.visited, self.node_scores = ({}, {}, set(), {})
        self.lexical, self.dense, self.support = ({}, {}, {})
        self.expanded_by_view, self.levels = (Counter(), Counter())
        self.frontiers = {v: [] for v in self.views}
        self.work = Counter()
        self.direct, self.association_ids = ([], [])
        self.paths = []
        self.association_done = False
        self.sparse = reader.sparse
        self.historical = cutoff is not None or period is not None
        if self.historical:
            self.sparse = SparseIndex(k1=reader.sparse.k1, b=reader.sparse.b, idf_scheme=reader.sparse.idf_scheme, epsilon=reader.sparse.epsilon)
            self.sparse.add_many(((m.msg_id, m.text) for m in reader.log if self.visible(m.msg_id)))
            self.work['historical_raw_index_rows'] = len(self.sparse)
        from ..structure_update.streaming_policy import LocalStatistics
        self.centroids = reader.navigation_centroids if not self.historical else None
        if self.centroids is None and reader.dense is not None:
            self.centroids = LocalStatistics(reader.dense.vector)
        self.centroid_start = self.centroids.leaf_reads if self.centroids else 0
        self._catalog()
        roots = [(tid, t.root_node_id) for tid, t in sorted(self.trees.items()) if t.view in self.views and (not t.deactivated)]
        self._score_many(roots)
        self.work['roots_scored'] = len(roots)
        for key in roots:
            self._push(key)
        self.work['direct_postings'] = self.sparse.postings_work(question)
        count = min(reader.config.navigation_direct, self.source_limit)
        for mid, _ in self.sparse.search(question, k=count):
            if self.eligible(mid):
                self._add(mid, 'direct')
                self.direct.append(mid)

    def visible(self, mid):
        self.work['visibility_checks'] += 1
        log = self.reader.log
        return mid in log and (self.cutoff is None or log.get(mid).ts <= self.cutoff) and (self.period is None or log.get(mid).seq <= self.period)

    def locations(self, mid):
        return [(p.tree_id, self.trees[p.tree_id].leaf_by_payload[mid]) for p in self.reader.placement.placements_of(mid) if p.tree_id in self.trees and (not self.trees[p.tree_id].deactivated) and (self.trees[p.tree_id].view in self.views) and (mid in self.trees[p.tree_id].leaf_by_payload)]

    def eligible(self, mid):
        return self.visible(mid) and bool(self.locations(mid))

    def _catalog(self):
        self.cards, self.chunks = ({}, {})
        if self.store is None:
            self.summary_index = SparseIndex()
            return
        if not hasattr(self.store, '_navigation_dependency_bounds'):
            self.store._navigation_dependency_bounds = {}
        bounds = self.store._navigation_dependency_bounds
        for key, card in self.store.cards.items():
            self.work['catalog_rows'] += 1
            tree = self.trees.get(key[0])
            node = tree.nodes.get(key[1]) if tree else None
            checked_stale = card.stale and getattr(self.store, '_checked_boundary', False) and (getattr(self.store, '_checked_tree_versions', {}).get(key[0]) == tree.version) if tree else False
            if tree is None or node is None or tree.deactivated or (tree.view not in self.views) or (node.summary_dirty and (not checked_stale)) or (key in self.store._forced) or (card.status not in {'complete', 'inherited'}):
                continue
            cached = bounds.get(key)
            if cached is None or cached[0] != card.version:
                if not all((mid in self.reader.log for mid in card.dependencies)):
                    continue
                self.work['dependency_metadata_reads'] += len(card.dependencies)
                cached = (card.version, max((self.reader.log.get(m).seq for m in card.dependencies), default=-1), max((self.reader.log.get(m).ts for m in card.dependencies), default=float('-inf')))
                bounds[key] = cached
            if self.period is not None and cached[1] > self.period or (self.cutoff is not None and cached[2] > self.cutoff):
                continue
            self.cards[key] = card
            self.chunks.update(self.store._chunks.get(key, ()))
        signature = tuple(((k, c.version, tuple(self.store._chunks.get(k, ()))) for k, c in self.cards.items()))
        cached = getattr(self.store, '_navigation_catalog', None)
        if cached is not None and cached[0] == signature:
            self.summary_index = cached[1]
        else:
            self.summary_index = SparseIndex()
            self.summary_index.add_many(self.chunks.items())
            self.store._navigation_catalog = (signature, self.summary_index)
            self.work['summary_index_rows_built'] += len(self.chunks)

    def _score_many(self, keys):
        keys = list(dict.fromkeys((k for k in keys if k not in self.node_scores)))
        chunk_ids = list(dict.fromkeys((c for k in keys if k in self.cards for c, _ in self.store._chunks.get(k, ()))))
        lexical = self.summary_index.score_subset(self.question, chunk_ids)
        dense = {}
        if chunk_ids and self.qvec is not None:
            known = [c for c in chunk_ids if c in self.store._vectors]
            if known:
                q = np.asarray(self.qvec)
                q = q / max(float(np.linalg.norm(q)), 1e-12)
                values = np.asarray([self.store._vectors[c] for c in known]) @ q
                dense = dict(zip(known, values))
        self.work['summary_chunks_scored'] += len(chunk_ids)
        for key in keys:
            tree, node = (self.trees[key[0]], self.trees[key[0]].nodes[key[1]])
            if key in self.cards:
                score = max((0.5 * (lexical.get(c, 0.0) / (1.0 + lexical.get(c, 0.0)) + (float(np.clip(dense[c], -1, 1)) + 1.0) / 2.0) for c, _ in self.store._chunks.get(key, ()) if c in dense), default=0.0)
            else:
                score = 0.0
                past = (self.cutoff is None or node.time_end <= self.cutoff) and self.period is None
                if past and self.centroids is not None and (self.qvec is not None):
                    _, vector = self.centroids.get(tree, key[1])
                    score = 0.5 * ((float(np.clip(vector @ self.qvec / max(float(np.linalg.norm(vector) * np.linalg.norm(self.qvec)), 1e-12), -1, 1)) + 1.0) / 2.0)
                    self.work['centroid_nodes_scored'] += 1
            self.node_scores[key] = score

    def _push(self, key, queue=None):
        if key not in self.visited:
            target = queue if queue is not None else self.frontiers[self.trees[key[0]].view]
            heapq.heappush(target, (-self.node_scores[key], *key))

    def _add(self, mid, origin):
        if mid in self.sources or len(self.sources) >= self.source_limit or (not self.eligible(mid)):
            return
        self.sources[mid] = None
        self.origins[mid] = origin

    def _step(self, queue):
        if not queue:
            return False
        _, tid, nid = heapq.heappop(queue)
        key = (tid, nid)
        if key in self.visited:
            return True
        tree, node = (self.trees[tid], self.trees[tid].nodes[nid])
        if node.level and self.work['internal_expansions'] >= self.node_limit:
            self._push(key, queue)
            return False
        self.visited.add(key)
        self.levels[f'{tree.view}/L{node.level}'] += 1
        self.paths.append([tid, nid, node.level, node.parent_id])
        if not node.level:
            for mid in node.child_ids:
                self._add(mid, 'hierarchy')
        else:
            self.work['internal_expansions'] += 1
            self.expanded_by_view[tree.view] += 1
            children = [(tid, cid) for cid in node.child_ids]
            self._score_many(children)
            for child in children:
                self._push(child, queue)
        return True

    def expand(self):
        cfg = self.reader.config
        while len(self.sources) < self.source_limit:
            progressed = False
            for view in self.views:
                queue = self.frontiers[view]
                if queue and len(self.sources) < self.source_limit and (self.expanded_by_view[view] < cfg.navigation_view_floor):
                    progressed = self._step(queue) or progressed
            if not progressed:
                break
        while len(self.sources) < self.source_limit:
            live = [q for q in self.frontiers.values() if q]
            if not live:
                break
            queue = min(live, key=lambda q: q[0])
            if not self._step(queue):
                leaves = [(entry, q) for q in live for entry in q if self.trees[entry[1]].nodes[entry[2]].level == 0]
                for entry, q in sorted(leaves, key=lambda x: x[0]):
                    if len(self.sources) >= self.source_limit:
                        break
                    q.remove(entry)
                    heapq.heapify(q)
                    self._step([entry])
                break

    def associate(self, records):
        before = set(self.sources)
        self.source_limit = len(self.sources) + 128
        self.node_limit = self.work['internal_expansions'] + 32
        queue, targets = ([], [])
        for record in records:
            for mid in record['members']:
                self._add(mid, 'association')
                for tid, leaf in self.locations(mid):
                    parent = self.trees[tid].nodes[leaf].parent_id
                    targets.append((tid, parent or leaf))
        self._score_many(targets)
        for key in targets:
            self._push(key, queue)
        while queue and len(self.sources) < self.source_limit:
            if not self._step(queue):
                break
        self.association_ids = sorted(set(self.sources) - before)
        self.association_done = True

    def associate_frontier(self, records):

        before = set(self.sources)
        self.source_limit = max(self.source_limit, len(self.sources) + 128)
        self.node_limit = max(self.node_limit, self.work['internal_expansions'] + 32)
        priorities = {}
        for record in records:
            for mid in record['members']:
                self._add(mid, 'association')
                for tid, leaf in self.locations(mid):
                    parent = self.trees[tid].nodes[leaf].parent_id
                    key = (tid, parent or leaf)
                    priorities[key] = max(priorities.get(key, 0.0), max(0.0, record.get('utility_score', 0.0)))
        self._score_many(priorities)
        for key, priority in priorities.items():
            self.node_scores[key] += priority
            self._push(key)
        self.expand()
        self.association_ids = sorted(set(self.sources) - before)
        self.association_done = True

    def enqueue_shortcuts(self, records):

        before = set(self.sources)
        priorities = {}
        for record in records:
            for mid in record['members']:
                self._add(mid, 'association')
                for tid, leaf in self.locations(mid):
                    parent = self.trees[tid].nodes[leaf].parent_id
                    key = (tid, parent or leaf)
                    priorities[key] = max(priorities.get(key, 0.0), record.get('navigation_value', 0.0))
        self._score_many(priorities)
        for key, priority in priorities.items():
            if key not in self.visited:
                heapq.heappush(self.frontiers[self.trees[key[0]].view], (-(self.node_scores[key] + priority), *key))
        self.association_ids = sorted(set(self.sources) - before)

    def recall(self, question, qvec, cutoff, period, spec):
        from .forest_first import TreeChoice
        if not self.association_done:
            self.expand()
        cfg = self.reader.config.s1
        ids = list(self.sources)
        new = [m for m in ids if m not in self.lexical]
        self.lexical.update(self.sparse.score_subset(question, new))
        self.work['raw_sparse_scored'] += len(new)
        if self.reader.dense is not None and qvec is not None and cfg.dense_weight:
            self.dense.update(self.reader.dense.score_subset(qvec, new))
            self.work['raw_dense_scored'] += len(new)
        for mid in new:
            keys = [(tid, self.trees[tid].nodes[nid].parent_id) for tid, nid in self.locations(mid)]
            keys = [k for k in keys if k in self.cards and self.trees[k[0]].nodes[k[1]].level == 1]
            self._score_many(keys)
            self.support[mid] = max((self.node_scores[k] for k in keys), default=0.0)
        lexical_order = sorted((m for m in ids if self.lexical[m] > 0), key=lambda m: (-self.lexical[m], m))
        signals = [Signal.of('bm25#0', lexical_order, weight=cfg.sparse_weight, rrf_k=cfg.msg_rrf_k)]
        if self.dense:
            signals.append(Signal.of('dense#0', sorted(ids, key=lambda m: (-self.dense.get(m, 0.0), m)), weight=cfg.dense_weight, rrf_k=cfg.msg_rrf_k))
        pool = {f.item: f for f in fuse(signals)}
        for mid in ids:
            pool.setdefault(mid, Fused(item=mid, score=0.0, best_rank=len(ids) + 1))
        raw = sorted(pool, key=lambda m: (-pool[m].score, m))
        ranks = {m: r for r, m in enumerate(raw)}
        for rank, mid in enumerate(sorted((m for m in ids if self.support[m] > 0), key=lambda m: (-self.support[m], ranks[m])), 1):
            f = pool[mid]
            bump = 0.35 / (cfg.msg_rrf_k + rank)
            pool[mid] = replace(f, score=f.score + bump, signals=(*f.signals, 'summary_support'))
        fused = sorted(pool.values(), key=lambda f: (-f.score, ranks[f.item]))
        tids = sorted({tid for mid in ids for tid, _ in self.locations(mid)})
        spans = [Span(msg_id=f.item, score=1.0 / (cfg.span_rrf_k + rank), view=min((tid for tid, _ in self.locations(f.item)), default=None), headers=()) for rank, f in enumerate(fused, 1)]
        choices = [TreeChoice(tid, self.trees[tid].view, 0.0, ('hierarchical_v1',), False) for tid in tids]
        trace = self.trace()
        trace.update(mode='hierarchical_v1', candidate_ids=ids, outside_seed_ids=sorted(set(ids) - set(self.direct)), trees_visited=tids, source_evaluations=len(ids), query_llm_calls=0)
        return (fused, choices, spans, trace, [(mid, self.lexical[mid]) for mid in lexical_order])

    def trace(self):
        return {**self.work, 'nodes_visited': len(self.visited), 'visits_by_view_level': dict(self.levels), 'paths': list(self.paths), 'direct_ids': self.direct, 'candidate_count': len(self.sources), 'association_added_ids': self.association_ids, 'node_budget': self.node_limit, 'source_budget': self.source_limit, 'centroid_vector_reads': self.centroids.leaf_reads - self.centroid_start if self.centroids else 0}

def _insufficient(result):
    verdict = result.stats.get('sufficiency')
    return result.stats.get('fallback', False) or not result.provenance or bool(verdict and verdict['verdict'] != 'supported')

def search_hierarchical(reader, question, **kwargs):
    graph = reader.association_graph
    if graph is not None and graph.config.learning_policy == 'associative_recall_v4':
        from .recall_navigation import search
        return search(reader, question, **kwargs)
    from ..associations.contracts import query_bindings
    from ..associations.selection import select
    started = time.perf_counter()
    cfg = reader.config
    active = copy(reader)
    active.full_raw_search = True
    active.association_graph = None
    active._navigation_spec = reader.analyser.analyse(question) if reader.analyser is not None else QuerySpec(question=question, variants=(question,))
    active._navigation_qvec = reader.embedder.encode([question], kind='query')[0] if reader.embedder else None
    cutoff = active._navigation_spec.as_of_date if active._navigation_spec.as_of_date is not None else kwargs.get('as_of')
    nav = Navigator(active, question, active._navigation_qvec, active._navigation_spec, cutoff, kwargs.get('as_of_period'))
    active._navigation = nav
    active._assembly_capture = {}
    if nav.historical and reader.sufficiency is not None:
        from .sufficiency import SufficiencyGate
        active.sufficiency = SufficiencyGate([m for m in reader.log if nav.visible(m.msg_id)], config=reader.sufficiency.config)
        active.sufficiency.log = reader.log
    fallback, attempts, full_work = (False, 1, {})
    with nav.sparse.query_cache():
        result = active._search(question, **kwargs)
        if _insufficient(result) and cfg.navigation_expand_once:
            nav.node_limit, nav.source_limit = (2 * cfg.navigation_nodes, 2 * cfg.navigation_sources)
            result = active._search(question, **kwargs)
            attempts += 1
        if _insufficient(result) and cfg.navigation_fallback:
            active._navigation = None
            result = active._search(question, **kwargs)
            fallback, attempts = (True, attempts + 1)
            full_work = result.stats.get('routing_summaries', {})
        base_result, base = (result, dict(active._assembly_capture))
        graph = reader.association_graph
        if graph is not None:
            records = None
            if graph.config.navigation == 'one_hop_v1' and (not base['out'].fallback):

                class Eligible:

                    def __contains__(self, mid):
                        return nav.eligible(mid)
                records = graph.triggered(base['out'].provenance, active._navigation_qvec.tolist() if active._navigation_qvec is not None else [], query_bindings(base['spec'], reader.render_names), reader.log, reader.validity, base['known_at'], cutoff if cutoff is not None else base['known_at'], Eligible())
                if records and (not fallback) and (graph.config.mode != 'shadow' or getattr(reader, 'association_trial', False)):
                    nav.associate(records)
                    result = active._search(question, **kwargs)
                    attempts += 1
            expanded = active._assembly_capture
            chosen, detail = select(reader, base['out'], expanded['ranked'], base['spec'], base['vector'], base['known_at'], base['cutoff'], expanded['mandatory'], expanded['render'], triggered_records=records, observation_ranked=base['ranked'])
            verdict = active.sufficiency.assess(question, chosen.provenance) if active.sufficiency and (not chosen.fallback) else None
            result = replace(result, context=chosen.context, provenance=chosen.provenance, stats={**result.stats, 'association': detail, 'n_spans': chosen.n_spans, 'tokens': chosen.tokens, 'truncated': chosen.truncated, 'fallback': chosen.fallback, 'sufficiency': verdict.to_dict() if verdict else None})
            if 'feedback_ranked_ids' in base_result.stats:
                result.stats['feedback_ranked_ids'] = base_result.stats['feedback_ranked_ids']
    result.stats['navigation'] = {**nav.trace(), 'strategy': 'hierarchical_v1', 'attempts': attempts, 'full_fallback': fallback, 'fallback_reason': 'evidence_insufficient' if fallback else None, 'full_source_evaluations': full_work.get('source_evaluations', 0), 'full_dense_evaluations': full_work.get('dense_evaluations', 0), 'full_summary_chunks_scored': full_work.get('scored_summary_chunks', 0), 'full_nodes_visited': full_work.get('nodes_visited', 0), 'direct_in_context': sorted(set(nav.direct) & set(result.provenance)), 'association_added_in_context': sorted(set(nav.association_ids) & set(result.provenance)), 'unlinked_source_order': list(base_result.provenance), 'elapsed_ms': (time.perf_counter() - started) * 1000.0}
    result.stats['elapsed_ms'] = round((time.perf_counter() - started) * 1000.0, 3)
    if hasattr(reader, '_baseline_capture'):
        reader._baseline_capture.update(assembly=base, navigation=nav, reader=active)
    return result
