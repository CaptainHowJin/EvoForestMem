from collections import defaultdict
from copy import copy
from dataclasses import replace
import time

from ..associations.contracts import query_bindings
from ..associations.recall_audit import complete_keys, select_current
from ..structure_update.contracts import CompletedQuery
from ..structure_update.feedback import source_version
from .s4_assemble import Span


def search(reader, question, **kwargs):
    from .navigation import search_hierarchical
    started = time.perf_counter()
    graph, active = reader.association_graph, copy(reader)
    active.association_graph, active._baseline_capture = None, {}
    active.capture_ranked_ids = True
    base = search_hierarchical(active, question, **kwargs)
    capture = active._baseline_capture
    assembly, nav = capture['assembly'], capture['navigation']
    known_at = assembly['known_at']
    effective_at = assembly['cutoff'] if assembly['cutoff'] is not None else known_at
    ranked = assembly['ranked']
    pool = list(dict.fromkeys([*base.provenance, *[s.msg_id for s in ranked]]))[:32]
    detail = dict(policy='associative_recall_v4', observation_kind='unlinked_final',
        base_ids=list(base.provenance), ranked_ids=pool, triggered=[], selected_groups=[],
        context_changed=False, evidence_changed=False, counterfactual_measured=True,
        shortcut_selected_ids=[], verified_support_ids=[], verified_coverage_before=0.,
        masked_coverage=0., verified_coverage_after=0., reason='no_associative_candidates',
        bindings=query_bindings(assembly['spec'], reader.render_names),
        versions={m: source_version(reader.log.get(m)) for m in pool},
        baseline_context_sha256=__import__('hashlib').sha256(base.context.encode()).hexdigest())
    controller = getattr(graph, 'controller', None)
    recognizer = getattr(controller, 'recognizer', None)
    query = CompletedQuery(question, known_at, getattr(assembly['spec'], 'asker', None),
                           tuple(base.provenance), tuple(pool))
    if recognizer is not None:
        recognizer.queries.add(query.identity)


    if graph.boundary is not None and min(effective_at, known_at) < graph.boundary:
        new, provenance, funnel = [], {}, {'reason': 'historical_graph_unavailable'}
    else:
        new, provenance, funnel = graph.candidates(reader, nav, pool, known_at, effective_at)
    detail.update(trigger_funnel=funnel, triggered=funnel.get('triggered', []),
                  novel_sources=new, association_paths=provenance)
    chosen, witness, receipt, audited = base, None, {}, False
    audit_started = time.perf_counter()
    if new and recognizer is not None:
        witness, receipt = recognizer.recognize(query, [*pool, *new], reader.log)
        detail['reason'] = receipt['status']
        audited = receipt['status'] in {'verified', 'unverified'}
    elif new:
        detail['reason'] = 'recognizer_unavailable'
    audit_seconds = time.perf_counter() - audit_started
    outcomes = {}
    if witness:
        spans = {s.msg_id: s for s in ranked if s.msg_id in pool}


        for i, mid in enumerate(pool):
            spans.setdefault(mid, Span(msg_id=mid, score=1. / (60 + i + 1)))
        for i, mid in enumerate(new):
            spans[mid] = Span(msg_id=mid, score=1. / (60 + len(pool) + i + 1))
        ordered = list(spans.values())
        keep, render, mandatory = assembly['out'], assembly['render'], assembly['mandatory']
        masked, count_a = select_current(keep, ordered, witness, mandatory, render, reader.config, set(pool))
        linked, count_b = select_current(keep, ordered, witness, mandatory, render, reader.config, set(spans))
        keys_base, keys_masked, keys_linked = [complete_keys(witness, c) for c in (keep, masked, linked)]
        denominator = max(1, len(witness.requirements))
        gain = keys_linked - keys_masked
        beneficial = bool(gain) and keys_base <= keys_linked and len(keys_linked) > len(keys_masked)
        accepted = beneficial
        if graph.config.mode == 'shadow' or controller.utility_config.mode == 'shadow':
            accepted = False
        if accepted:
            verdict = assembly['verdicts'][linked.context]
            chosen = replace(base, context=linked.context, provenance=linked.provenance,
                stats={**base.stats, 'tokens': linked.tokens, 'n_spans': linked.n_spans,
                    'truncated': linked.truncated, 'fallback': linked.fallback,
                    'sufficiency': verdict.to_dict() if verdict else None})
            detail['reason'] = 'verified_incremental_evidence'
        else:
            detail['reason'] = 'no_increment_over_masked_selector'


        all_known = all(r['status'] == 'supported' for r in witness.requirements)
        contributions = defaultdict(set)
        for mid, path in provenance.items():
            if beneficial and mid in linked.provenance:
                removed = replace(linked, provenance=[m for m in linked.provenance if m != mid])
                lost = gain - complete_keys(witness, removed)
                for edge in path:
                    contributions[edge].update(lost)
        for edge in {e for path in provenance.values() for e in path}:
            if contributions[edge]:
                outcomes[edge] = 1.
            elif all_known:
                outcomes[edge] = 0.
        detail.update(requirements=witness.requirements, audit_requests=witness.request_keys,
            masked_source_order=list(masked.provenance), trial_source_order=list(linked.provenance),
            verified_coverage_before=len(keys_base)/denominator, masked_coverage=len(keys_masked)/denominator,
            verified_coverage_after=len(keys_linked if accepted else keys_base)/denominator,
            trial_coverage=len(keys_linked)/denominator, packing_candidates=count_a+count_b,
            verified_support_ids=sorted(witness.evidence & set(chosen.provenance)))
    changed = chosen.context != base.context
    detail.update(context_changed=changed, evidence_changed=list(chosen.provenance) != list(base.provenance),
        shortcut_selected_ids=sorted(set(new) & set(chosen.provenance)),
        feedback_status=receipt.get('status', 'not_requested'), feedback_seconds=audit_seconds,
        edge_feedback=outcomes, seconds=time.perf_counter()-started)
    if controller is not None:
        controller.pending.append(dict(question=question, ask_time=known_at, bindings=detail['bindings'],
            versions=receipt.get('versions', {}), outcomes=outcomes, changed=changed, audited=audited))
    chosen.stats['association'] = detail
    chosen.stats['navigation'] = {**base.stats['navigation'], 'policy':'associative_recall_v4',
        'early_stop':False, 'association_added_in_context':detail['shortcut_selected_ids']}
    chosen.stats['elapsed_ms'] = round((time.perf_counter()-started)*1000, 3)
    return chosen
