from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence
from ..forest.placement import PlacementMap
from ..log.store import MessageLog
from .fusion import Fused, Signal, aggregate_views, fuse
from .pipeline import PipelineConfig, PipelineResult
from .s0_analyse import QueryAnalyser, QuerySpec
from .s4_assemble import Span, assemble
log_ = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class TreeChoice:

    tree_id: str
    view: str
    score: float
    signals: tuple[str, ...]
    pinned: bool

class ForestFirstRetriever:


    def __init__(self, *, log: MessageLog, placement: PlacementMap, sparse, dense=None, embedder=None, trees: Mapping[str, Any] | None=None, roots=None, analyser: QueryAnalyser | None=None, config: PipelineConfig | None=None, views: Sequence[str] | None=None, artifact_keys: Mapping[str, Sequence[str]] | None=None, artifact_soft_keys: set[str] | None=None, validity=None, norms=None, episodes=None, render_names: tuple[Mapping[str, str], Mapping[str, str]] | None=None, routing_summaries=None, full_raw_search: bool=False, association_graph=None, sufficiency_statistics=None, navigation_centroids=None) -> None:
        self.log = log
        self.placement = placement
        self.sparse = sparse
        self.dense = dense
        self.embedder = embedder
        self.trees = dict(trees or {})
        self.roots = roots
        self.analyser = analyser
        self.config = config or PipelineConfig()
        self.views = tuple(views) if views else ('channel', 'actor', 'artifact')
        self.artifact_keys = artifact_keys
        from ..forest.types import tree_id_for
        self.artifact_soft_tree_ids = {tree_id_for('artifact', k) for k in artifact_soft_keys or ()}
        self.validity = validity
        self.norms = norms
        self.episodes = episodes
        self.render_names = render_names
        self.routing_summaries = routing_summaries
        self.full_raw_search = full_raw_search or getattr(routing_summaries, 'mode', None) == 'hierarchical_full'
        self.association_graph = association_graph
        self.navigation_centroids = navigation_centroids
        cfg = self.config
        self.sufficiency = None
        if cfg.sufficiency_header:
            from .sufficiency import SufficiencyGate
            self.sufficiency = SufficiencyGate(log, statistics=sufficiency_statistics)
        self.duration = None
        if cfg.duration_headers:
            from ..forest.episodes import EpisodeTable
            from .duration import DurationResolver
            table = episodes if episodes is not None else EpisodeTable(log)
            self.duration = DurationResolver(log=log, table=table, scores=sparse, selector=None)
        if self.norms is None and cfg.norm_spans > 0:
            from ..forest.norms import NormConfig, NormIndex
            self.norms = NormIndex.from_stream(log, config=NormConfig(max_results=cfg.norm_spans))

    def _message_fusion(self, question: str, qvec=None) -> list:

        cfg = self.config.s1
        signals: list[Signal] = [Signal.of('bm25#0', (m for m, _ in self.sparse.search(question, k=cfg.msg_pool)), weight=cfg.sparse_weight, rrf_k=cfg.msg_rrf_k)]
        if self.dense is not None and self.embedder is not None and (cfg.dense_weight != 0.0):
            if qvec is None:
                qvec = self.embedder.encode([question], kind='query')[0]
            signals.append(Signal.of('dense#0', (m for m, _ in self.dense.search(qvec, k=cfg.msg_pool)), weight=cfg.dense_weight, rrf_k=cfg.msg_rrf_k))
        return fuse(signals, limit=cfg.msg_pool)

    def _binds(self, spec: QuerySpec) -> dict[str, list[str]]:

        from ..forest.types import tree_id_for
        by_view: dict[str, list[str]] = {v: [] for v in self.views}
        known = set(self.trees) if self.trees else None

        def push(view: str, key: str) -> None:
            tid = tree_id_for(view, key)
            if known is not None and tid not in known:
                return
            if view in by_view and tid not in by_view[view]:
                by_view[view].append(tid)
        if spec.asker:
            push('actor', spec.asker)
        for ch in spec.channels:
            push('channel', ch)
        for person in spec.people:
            push('actor', person)
        if self.artifact_keys and 'artifact' in by_view:
            from .s0_analyse import norm_artifact_tokens, _contains_run
            q_tokens = norm_artifact_tokens(spec.question)
            for key, runs in self.artifact_keys.items():
                if any((_contains_run(q_tokens, tokens) for tokens in runs)):
                    push('artifact', key)
        return by_view

    def _root_enabled(self, view: str) -> bool:

        if self.roots is None:
            return False
        if not self.config.s1.require_summaries_for_root:
            return True
        return self.roots.summarised(view) > 0

    def _select_trees(self, spec: QuerySpec, ranked_ids: Sequence[str], qvec, summary_trees=None) -> list[TreeChoice]:

        cfg = self.config.s1
        binds = self._binds(spec)
        choices: list[TreeChoice] = []
        for view in self.views:
            if not spec.in_scope(view):
                continue
            signals: list[Signal] = []
            tree_scores = self.placement.trees_of(ranked_ids, view, rrf_k=cfg.leaf2tree_rrf_k, max_hits_per_tree=cfg.hits_cap_for(view))
            if tree_scores:
                signals.append(Signal.of('leaf2tree', tree_scores, view=view, weight=cfg.w_leaf2tree, rrf_k=cfg.tree_rrf_k))
            if summary_trees and summary_trees.get(view):
                signals.append(Signal.of('summary', summary_trees[view], view=view, weight=self.routing_summaries.config.tree_weight, rrf_k=cfg.tree_rrf_k))
            if self._root_enabled(view):
                ranked = self.roots.rank(spec.question, view=view, qvec=qvec)
                if ranked:
                    signals.append(Signal.of('root#0', ranked, view=view, weight=cfg.w_root, rrf_k=cfg.tree_rrf_k))
            view_binds = binds.get(view, [])
            if view_binds:
                signals.append(Signal.of('bind', view_binds, view=view, weight=cfg.w_bind, rrf_k=cfg.tree_rrf_k))
            if not signals:
                continue
            pool = fuse(signals)
            cap = int(cfg.trees_per_view.get(view, cfg.default_trees_per_view))
            pinned = set(view_binds) if cfg.bind_mode == 'pin' else set()
            if view == 'artifact':
                pinned.difference_update(self.artifact_soft_tree_ids)
            taken = 0
            for f in pool:
                is_pin = f.item in pinned
                if taken >= cap and (not is_pin):
                    continue
                choices.append(TreeChoice(tree_id=f.item, view=view, score=f.score, signals=f.signals, pinned=is_pin))
                if not is_pin:
                    taken += 1
        return choices

    def _summary_recall(self, question, fused, qvec, cutoff, period, spec):

        store = self.routing_summaries
        if store is None:
            return (fused, {}, {})

        def visible(mid):
            return mid in self.log and (cutoff is None or self.log.get(mid).ts <= cutoff) and (period is None or self.log.get(mid).seq <= period)
        ids, trees, opened = store.recall(question, trees=self.trees, visible=visible, views=[v for v in self.views if spec.in_scope(v)])
        lexical = self.sparse.score_subset(question, ids) if ids else {}
        dense = self.dense.score_subset(qvec, ids) if ids and qvec is not None and (self.dense is not None) and (self.config.s1.dense_weight != 0) else {}
        scores = {mid: (lexical.get(mid, 0) / (1 + lexical.get(mid, 0)) + ((max(-1.0, min(1.0, dense[mid])) + 1) / 2 if mid in dense else 0)) / (2 if dense else 1) for mid in ids}
        ranked = sorted((mid for mid in ids if scores[mid] > 0), key=lambda mid: (-scores[mid], mid))
        pool = {f.item: f for f in fused}
        original = set(pool)
        for rank, mid in enumerate(ranked, 1):
            bump = store.config.source_weight / (self.config.s1.msg_rrf_k + rank)
            old = pool.get(mid, Fused(item=mid, score=0, best_rank=rank))
            pool[mid] = replace(old, score=old.score + bump, signals=(*old.signals, 'summary_raw'), contributions={**old.contributions, 'summary_raw': bump})
        augmented = sorted(pool.values(), key=lambda f: -f.score)
        return (augmented, trees, {'mode': 'budgeted_l1', 'opened': opened, 'source_evaluations': len(ids), 'sparse_evaluations': len(lexical), 'dense_evaluations': len(dense), 'outside_seed_ids': sorted(set(pool) - original), 'candidate_ids': ids, 'query_llm_calls': 0})

    def _full_recall(self, question, qvec, cutoff, period, spec):

        from ..forest.hierarchical_summaries import traverse_forest
        from ..index.sparse import SparseIndex

        def visible(mid):
            return mid in self.log and (cutoff is None or self.log.get(mid).ts <= cutoff) and (period is None or self.log.get(mid).seq <= period)
        views = [view for view in self.views if spec.in_scope(view)]
        support = {}
        if self.routing_summaries is not None:
            support, trace = self.routing_summaries.recall(question, trees=self.trees, visible=visible, views=views, qvec=qvec, explain_ids=getattr(self, 'diagnostic_source_ids', ()))
        else:
            trace = traverse_forest(self.trees, views=views, visible=visible)
        ids = trace['candidate_ids']
        sparse = self.sparse
        if cutoff is not None or period is not None:
            sparse = SparseIndex(k1=sparse.k1, b=sparse.b, idf_scheme=sparse.idf_scheme, epsilon=sparse.epsilon)
            sparse.add_many(((m.msg_id, m.text) for m in self.log if visible(m.msg_id)))
        all_lexical = sparse.scores(question)
        lexical = {mid: all_lexical.get(mid, 0.0) for mid in ids}
        dense = self.dense.score_subset(qvec, ids) if self.dense is not None and qvec is not None and (self.config.s1.dense_weight != 0) else {}
        cfg = self.config.s1
        lexical_order = sorted((mid for mid in ids if lexical[mid] > 0), key=lambda mid: (-lexical[mid], mid))
        signals = [Signal.of('bm25#0', lexical_order, weight=cfg.sparse_weight, rrf_k=cfg.msg_rrf_k)]
        if dense:
            signals.append(Signal.of('dense#0', sorted(dense, key=lambda mid: (-dense[mid], mid)), weight=cfg.dense_weight, rrf_k=cfg.msg_rrf_k))
        pool = {row.item: row for row in fuse(signals)}
        for mid in ids:
            pool.setdefault(mid, Fused(item=mid, score=0.0, best_rank=len(ids) + 1))
        raw_order = sorted(pool, key=lambda mid: (-pool[mid].score, mid))
        raw_rank = {mid: rank for rank, mid in enumerate(raw_order)}
        seeds = set(raw_order[:cfg.msg_pool])
        for rank, mid in enumerate(sorted((mid for mid in ids if support.get(mid, 0) > 0), key=lambda mid: (-support[mid], raw_rank[mid])), 1):
            row = pool[mid]
            bump = self.routing_summaries.config.source_weight / (cfg.msg_rrf_k + rank)
            pool[mid] = replace(row, score=row.score + bump, signals=(*row.signals, 'summary_support'), contributions={**row.contributions, 'summary_support': bump})
        fused = sorted(pool.values(), key=lambda row: (-row.score, raw_rank[row.item]))
        watched = set(getattr(self, 'diagnostic_source_ids', ()))
        if watched:
            trace['source_rank_trace'] = {row.item: {'raw_rank': raw_rank[row.item] + 1, 'summary_support': support.get(row.item, 0.0), 'fused_score': row.score, 'fused_rank': rank} for rank, row in enumerate(fused, 1) if row.item in watched}
        choices = [TreeChoice(tid, self.trees[tid].view, 0.0, ('full_search',), False) for tid in trace['trees_visited']]
        spans = [Span(msg_id=row.item, score=1.0 / (cfg.span_rrf_k + rank), view=next((p.tree_id for p in self.placement.placements_of(row.item) if p.tree_id in trace['trees_visited']), None), headers=()) for rank, row in enumerate(fused, 1)]
        trace.update(mode='hierarchical_full' if self.routing_summaries else 'full_raw_control', source_evaluations=len(ids), sparse_evaluations=len(ids), dense_evaluations=len(dense), supported_sources=len(support), outside_seed_ids=sorted(set(ids) - seeds), query_llm_calls=0, summary_support_weight=0.35 if self.routing_summaries else 0.0)
        return (fused, choices, spans, trace, [(mid, lexical[mid]) for mid in lexical_order])

    def _drill(self, fused_msgs, choices: Sequence[TreeChoice]) -> list[Span]:

        cfg = self.config.s1
        want: dict[str, dict[str, int]] = {}
        for rank, c in enumerate(choices):
            want.setdefault(c.view, {})[c.tree_id] = rank
        per_view: dict[str, list[tuple[str, float]]] = {}
        attribution: dict[tuple[str, str], str] = {}
        for view, tree_rank in want.items():
            out: list[tuple[str, float]] = []
            rank = 0
            for f in fused_msgs:
                eligible = [p.tree_id for p in self.placement.in_view(f.item, view) if p.tree_id in tree_rank]
                if not eligible:
                    continue
                rank += 1
                out.append((f.item, 1.0 / (cfg.span_rrf_k + rank)))
                attribution[view, f.item] = min(eligible, key=lambda t: tree_rank[t])
            per_view[view] = out
        crossed = aggregate_views(per_view, mode=cfg.aggregation, payloads=attribution, charge_shared=False)
        return [Span(msg_id=c.item, score=c.score, view=c.payload.get(c.owner, c.owner), headers=()) for c in crossed]

    def _leaf_up_headers(self, spans: Sequence[Span]) -> dict[tuple[str, str], tuple[str, ...]]:

        if not self.config.leaf_up_headers or not self.trees:
            return {}
        out: dict[tuple[str, str], tuple[str, ...]] = {}
        for span in spans:
            tree = self.trees.get(span.view or '')
            if tree is None:
                continue
            placements = [p for p in self.placement.placements_of(span.msg_id) if p.tree_id == tree.tree_id]
            if not placements:
                continue
            node = tree.nodes.get(placements[0].leaf_node_id)
            headers: list[str] = []
            while node is not None and node.parent_id:
                node = tree.nodes.get(node.parent_id)
                if node is not None and node.level >= 1:
                    header = str(getattr(node, 'summary', '') or '').strip()
                    if header:
                        headers.append(header)
            if headers:
                out[tree.tree_id, span.msg_id] = tuple(reversed(headers))
        return out

    def search(self, question: str, *, qa_item: Any=None, as_of: float | None=None, as_of_period: int | None=None, version_time: float | None=None) -> PipelineResult:
        if self.config.retrieval_strategy == 'hierarchical_v1':
            from .navigation import search_hierarchical
            return search_hierarchical(self, question, qa_item=qa_item, as_of=as_of, as_of_period=as_of_period, version_time=version_time)
        if self.full_raw_search:
            with self.sparse.query_cache():
                return self._search(question, qa_item=qa_item, as_of=as_of, as_of_period=as_of_period, version_time=version_time)
        return self._search(question, qa_item=qa_item, as_of=as_of, as_of_period=as_of_period, version_time=version_time)

    def _search(self, question: str, *, qa_item=None, as_of=None, as_of_period=None, version_time=None) -> PipelineResult:
        cfg = self.config
        t0 = time.perf_counter()
        spec = getattr(self, '_navigation_spec', None) or (self.analyser.analyse(question) if self.analyser is not None else QuerySpec(question=question, variants=(question,)))
        needs_qvec = self.embedder is not None and (self.dense is not None and cfg.s1.dense_weight != 0.0 or any((self._root_enabled(view) for view in self.views)) or (self.full_raw_search and self.routing_summaries is not None))
        qvec = self._navigation_qvec if hasattr(self, '_navigation_qvec') else self.embedder.encode([question], kind='query')[0] if needs_qvec else None
        fused_msgs = [] if self.full_raw_search else self._message_fusion(question, qvec)
        ranked_ids = tuple((f.item for f in fused_msgs))
        cutoff = spec.as_of_date if spec.as_of_date is not None else as_of
        version_time = cutoff if cutoff is not None else self.log.latest_ts if version_time is None else version_time
        if cutoff is not None:
            fused_msgs = [f for f in fused_msgs if self.log.get(f.item).ts <= cutoff]
            ranked_ids = tuple((f.item for f in fused_msgs))
        if getattr(self, '_navigation', None) is not None:
            fused_msgs, choices, spans, summary_stats, full_lexical = self._navigation.recall(question, qvec, cutoff, as_of_period, spec)
        elif self.full_raw_search:
            fused_msgs, choices, spans, summary_stats, full_lexical = self._full_recall(question, qvec, cutoff, as_of_period, spec)
        else:
            fused_msgs, summary_trees, summary_stats = self._summary_recall(question, fused_msgs, qvec, cutoff, as_of_period, spec)
            ranked_ids = tuple((f.item for f in fused_msgs))
            choices = self._select_trees(spec, ranked_ids, qvec, summary_trees)
            spans = self._drill(fused_msgs, choices)
        derived: list[Any] = []
        mandatory_ids: set[str] = set()
        stats_extra: dict[str, Any] = {}
        if cutoff is not None:
            stats_extra['as_of_cutoff'] = cutoff
        norm_hits = False
        evidence = None
        if self.norms is not None:
            from .validity import select_norm_evidence, has_update_intent
            evidence = select_norm_evidence(self.norms, question=question, ledger=self.validity, log=self.log, known_at=version_time, period=as_of_period, include_history=spec.temporal_op == 'audit' or has_update_intent(question))
            derived.extend(evidence.blocks)
            mandatory_ids.update(evidence.mandatory_msg_ids)
            stats_extra['norm_versions'] = evidence.stats
            norm_hits = bool(evidence.stats['hits'])
        if self.validity is not None and cfg.validity_overlay:
            from .validity import apply_validity, has_update_intent
            version_query = spec.temporal_op in ('as_of', 'audit') or has_update_intent(question)
            if norm_hits or version_query:
                overlay = apply_validity(spans, question=question, ledger=self.validity, as_of=version_time, as_of_period=as_of_period, actor=spec.asker, penalty=0.0 if spec.temporal_op == 'audit' else cfg.validity_penalty, norm_evidence=evidence, discover_subjects=version_query)
                spans = list(overlay.spans)
                if cfg.validity_headers and overlay.block is not None:
                    derived.append(overlay.block)
                mandatory_ids.update(overlay.mandatory_msg_ids)
                stats_extra['validity'] = {**overlay.stats, 'demotion': 'soft' if cfg.validity_penalty == 0.0 else 'hard'}
        if self.duration is not None:
            fact = self.duration.resolve(spec)
            if fact is not None:
                derived.append(fact)
                mandatory_ids.update(fact.msg_ids)
        present = {s.msg_id for s in spans}
        ceiling = max((s.score for s in spans), default=0.0) + 1.0
        spans = list(spans)
        for mid in sorted(mandatory_ids - present):
            if mid in self.log and (cutoff is None or self.log.get(mid).ts <= cutoff) and (not self.full_raw_search or as_of_period is None or self.log.get(mid).seq <= as_of_period):
                spans.append(Span(msg_id=mid, score=ceiling, view=None, headers=()))
        ranked = sorted(spans, key=lambda s: (-s.score, s.msg_id))
        if getattr(self, 'capture_ranked_ids', False):
            stats_extra['feedback_ranked_ids'] = [s.msg_id for s in ranked[:32]]
        must = [s for s in ranked if s.msg_id in mandatory_ids]
        rest = [s for s in ranked if s.msg_id not in mandatory_ids]
        capped = [*must, *rest][:cfg.max_spans]
        reason = None
        if not choices:
            reason = 'no_trees'
        elif len(capped) < cfg.min_spans:
            reason = 'few_spans'
        fallback_spans = None
        if reason and cfg.fallback_enabled:
            raw = full_lexical if self.full_raw_search else self.sparse.search(question, k=cfg.s1.msg_pool)
            if cutoff is not None:
                raw = [(mid, sc) for mid, sc in raw if self.log.get(mid).ts <= cutoff]
            fallback_spans = [Span(msg_id=mid, score=sc, view=None, headers=()) for mid, sc in raw[:cfg.fallback_k]]
        depth = min(cfg.max_depth, int(spec.target_depth)) if cfg.depth_policy == 'spec' else cfg.max_depth
        headers_map = {} if self.full_raw_search else self._leaf_up_headers(capped)
        if headers_map:
            capped = [Span(msg_id=s.msg_id, score=s.score, view=s.view, headers=headers_map.get((s.view or '', s.msg_id), s.headers)) for s in capped]
        out = assemble(self.log, capped, budget_tokens=cfg.budget_tokens, style=cfg.style, max_depth=depth, min_spans=max(1, cfg.min_spans) if reason else cfg.min_spans, header_budget_frac=cfg.header_budget_frac, section_headers=cfg.section_headers, lifecycle_marks=cfg.lifecycle_marks, marker_marks=cfg.marker_marks, rank_numbers=cfg.rank_numbers, derived=derived, mandatory_msg_ids=mandatory_ids, fallback_spans=fallback_spans, render_names=self.render_names if cfg.render_display_names else None, render_seconds=cfg.render_seconds)
        verdict = None
        if self.sufficiency is not None and (not out.fallback) and (cfg.style == 'structured'):
            verdict = self.sufficiency.assess(question, out.provenance)
            if verdict.announce:
                from .sufficiency import _SufficiencyBlock
                out = assemble(self.log, capped, budget_tokens=cfg.budget_tokens, style=cfg.style, max_depth=depth, min_spans=max(1, cfg.min_spans) if reason else cfg.min_spans, header_budget_frac=cfg.header_budget_frac, section_headers=cfg.section_headers, lifecycle_marks=cfg.lifecycle_marks, marker_marks=cfg.marker_marks, rank_numbers=cfg.rank_numbers, derived=[*derived, _SufficiencyBlock(verdict)], mandatory_msg_ids=mandatory_ids, fallback_spans=fallback_spans, render_names=self.render_names if cfg.render_display_names else None, render_seconds=cfg.render_seconds)
        if self.association_graph is not None or hasattr(self, '_assembly_capture'):
            from ..associations.selection import select
            associated_verdicts = {out.context: verdict}

            def render_associated(selected):
                kwargs = dict(budget_tokens=cfg.budget_tokens, style=cfg.style, max_depth=depth, min_spans=cfg.min_spans, header_budget_frac=cfg.header_budget_frac, section_headers=cfg.section_headers, lifecycle_marks=cfg.lifecycle_marks, marker_marks=cfg.marker_marks, rank_numbers=cfg.rank_numbers, mandatory_msg_ids=mandatory_ids, render_names=self.render_names if cfg.render_display_names else None, render_seconds=cfg.render_seconds)
                candidate = assemble(self.log, selected, derived=derived, **kwargs)
                candidate_verdict = None
                if self.sufficiency is not None and (not candidate.fallback) and (cfg.style == 'structured'):
                    candidate_verdict = self.sufficiency.assess(question, candidate.provenance)
                    if candidate_verdict.announce:
                        from .sufficiency import _SufficiencyBlock
                        candidate = assemble(self.log, selected, derived=[*derived, _SufficiencyBlock(candidate_verdict)], **kwargs)
                associated_verdicts[candidate.context] = candidate_verdict
                return candidate
            if hasattr(self, '_assembly_capture'):
                self._assembly_capture.update(out=out, ranked=ranked, spec=spec, vector=qvec, known_at=version_time, cutoff=cutoff, mandatory=mandatory_ids, render=render_associated, verdicts=associated_verdicts)
            if self.association_graph is not None:
                out, stats_extra['association'] = select(self, out, ranked, spec, qvec, version_time, cutoff, mandatory_ids, render_associated)
                verdict = associated_verdicts[out.context]
        stats = {'retriever': 'forest_first', 'version_authority': 'validity_ledger' if self.validity is not None else 'unavailable', 'version_time': version_time, 'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 3), 'question_type': spec.question_type, 'temporal_op': spec.temporal_op, 'trees_selected': [{'tree_id': c.tree_id, 'view': c.view, 'score': round(c.score, 6), 'signals': list(c.signals), 'pinned': c.pinned} for c in choices], 'n_spans': out.n_spans, 'tokens': out.tokens, 'truncated': out.truncated, 'fallback': out.fallback, 'fallback_reason': reason, 'mandatory': len(mandatory_ids & set(out.provenance)), 'sufficiency': None if verdict is None else verdict.to_dict(), **stats_extra}
        if self.routing_summaries is not None or self.full_raw_search:
            stats['routing_summaries'] = {**summary_stats, 'recovered_in_context': sorted(set(out.provenance) & set(summary_stats['outside_seed_ids']))}
        return PipelineResult(context=out.context, provenance=out.provenance, stats=stats, spec=spec)
