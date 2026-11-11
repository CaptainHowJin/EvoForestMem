from itertools import combinations
import time

from .contracts import query_bindings
from ..structure_update.feedback import source_version


def select(retriever, keep, ranked, spec, vector, known_at, cutoff, mandatory, render,
           *, triggered_records=None, observation_ranked=None, observation_keep=None):
    started = time.perf_counter()
    graph, cfg = retriever.association_graph, retriever.config
    vector = vector.tolist() if hasattr(vector, 'tolist') else list(vector or [])
    condition = query_bindings(spec, retriever.render_names)

    observed = ranked if observation_ranked is None else observation_ranked
    unlinked = keep if observation_keep is None else observation_keep
    trace = {'base_ids': list(unlinked.provenance), 'ranked_ids': [s.msg_id for s in observed[:32]],
             'scores': {s.msg_id: s.score for s in observed[:32]}, 'vector': vector,
             'bindings': condition, 'versions': {m: source_version(retriever.log.get(m))
                 for m in dict.fromkeys([*unlinked.provenance, *(s.msg_id for s in observed[:32])])},
             'locations': {m: [[p.tree_id, p.leaf_node_id] for p in retriever.placement.placements_of(m)]
                           for m in unlinked.provenance}, 'triggered': [], 'selected_groups': [],
             'evidence_changed': False, 'context_changed': False, 'reason': 'no_trigger'}
    trace['trigger_funnel'] = {}
    original = {s.msg_id: s for s in ranked}
    if keep.fallback or keep.tokens > cfg.budget_tokens or keep.n_spans > cfg.max_spans:
        trace['reason'] = 'baseline_fallback_or_infeasible'
        trace['seconds'] = time.perf_counter() - started
        return keep, trace
    try:
        triggered = triggered_records if triggered_records is not None else graph.triggered(keep.provenance, vector, condition, retriever.log, retriever.validity,
                                    known_at, cutoff if cutoff is not None else known_at, set(original),
                                    diagnostics=trace['trigger_funnel'])
        trace['triggered'] = [r['id'] for r in triggered]
        if not triggered or (graph.config.mode == 'shadow' or graph.utility.config.mode == 'shadow') and not getattr(retriever, 'association_trial', False):
            trace['seconds'] = time.perf_counter() - started
            return keep, trace
        maximum = max((max(0., s.score) for s in ranked if s.msg_id not in mandatory), default=0.)
        weak = {}
        for r in triggered:
            if r['kind'] == 'cooccurrence' and graph.utility.config.mode == 'off':
                for mid in r['members']:
                    if mid not in keep.provenance:
                        weak[mid] = max(weak.get(mid, 0.), graph.config.weak_weight * maximum *
                                        r['similarity'] * r['confidence'])
        groups = [r for r in triggered if r['kind'] == 'complement']


        pool = {s.msg_id for s in ranked[:32]} | set(keep.provenance) | mandatory
        pool.update(m for r in triggered for m in r['members'])
        order = sorted((m for m in pool if m in original),
                       key=lambda m: (-(original[m].score + weak.get(m, 0.)), m))
        def utility(ids):
            ids = set(ids)
            if graph.utility.config.mode != 'off':


                field = 'utility_score' if getattr(retriever, 'association_trial', False) else 'utility_mean'
                return sum(original[m].score for m in sorted(ids)) + maximum * sum(
                    max(0., g.get(field, 0.)) * max(0., g['similarity'])
                    for g in triggered if set(g['members']) <= ids)
            return sum(original[m].score + weak.get(m, 0.) for m in sorted(ids)) + sum(
                graph.config.group_weight * maximum * g['similarity'] * (len(g['members']) - 1)
                for g in groups if set(g['members']) <= ids)
        best, best_score, selected = keep, utility(keep.provenance), []
        trace['combinations'] = 0
        for size in range(len(groups) + 1):
            for chosen in combinations(groups, size):
                reserved = mandatory | {m for r in chosen for m in r['members']}
                if len(reserved) > cfg.max_spans or not reserved <= set(original):
                    continue
                ids = [m for m in order if m in reserved]
                ids += [m for m in order if m not in reserved][:max(0, cfg.max_spans - len(ids))]
                trace['combinations'] += 1
                while ids:


                    spans = sorted((original[m] for m in ids), key=lambda s: (-s.score, s.msg_id))
                    candidate = render(spans)
                    if (reserved <= set(candidate.provenance) and candidate.tokens <= cfg.budget_tokens
                            and candidate.n_spans <= cfg.max_spans and not candidate.fallback):
                        value = utility(candidate.provenance)
                        if value > best_score + 1e-12:
                            best, best_score = candidate, value
                            selected = [r['id'] for r in chosen]
                        break
                    removable = [m for m in ids if m not in reserved]
                    if not removable:
                        break
                    worst = min(removable, key=lambda m: (original[m].score + weak.get(m, 0.), m))
                    ids.remove(worst)
        trace.update(selected_groups=selected, evidence_changed=set(best.provenance) != set(keep.provenance),
                     context_changed=best.context != keep.context, reason='selected' if best is not keep else 'KEEP',
                     keep_utility=utility(keep.provenance), selected_utility=best_score)
        trace['seconds'] = time.perf_counter() - started
        return best, trace
    except Exception as exc:
        trace.update(reason='association_failed', error=type(exc).__name__ + ': ' + str(exc),
                     seconds=time.perf_counter() - started)
        return keep, trace
