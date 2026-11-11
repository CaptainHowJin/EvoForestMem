from copy import deepcopy
from dataclasses import asdict
import time

from .contracts import fingerprint
from .controller import valid_summaries
from .online_utility import paired_acceptance, paired_gain, recent_legal_witnesses
from .operations import preview
from .streaming_journal import summary_patch
from .streaming_local import stage_summary, tree_patch, validate_local
from .streaming_policy import propose_prefix


def decide(ctl, message, seq, anchors, value):
    row, cfg = value['decision'], ctl.utility.config
    row['utility_mode'] = cfg.mode

    if len(ctl.history) < 2:
        row['reason'] = 'utility_no_independent_history'
        return value
    graph = getattr(ctl.session, 'association_graph', None)
    graph_seconds = (graph.seconds if graph.config.learning_policy == 'associative_recall_v4'
                     else graph.utility.seconds) if graph is not None else 0.
    credit = (seq + 1) * cfg.maintenance_seconds_per_message - ctl.utility.seconds - graph_seconds
    estimated = max((r['mean_seconds'] for r in ctl.utility.statistics.values()), default=1.)
    row['admission'] = dict(policy='utility_v1', seconds_credit=credit,
                            expected_seconds=estimated, admitted=credit >= estimated)
    if credit < estimated:
        row['reason'] = 'utility_time_budget_deferred'
        return value
    legal = recent_legal_witnesses(ctl.history, ctl.session.log, ctl.session.updater.ledger,
                                   message.ts, max(2, ctl.config.verify_per_point))
    legal = [w for w in legal if ctl.witness_prefixes.get(w.identity, seq + 2) <= seq]
    if len(legal) < 2:
        row['reason'] = 'utility_no_independent_history'
        return value
    candidates = propose_prefix(ctl.trees, anchors, ctl.statistics, ctl.streaming_config, 16)
    def action(op):
        return f'forest:{op.tree_id}:{op.kind}:{ctl.trees[op.tree_id].nodes[op.node_id].level}'

    candidates.sort(key=lambda x: (-ctl.utility.score(action(x[0])), -x[1]['gain'], x[0].identity))
    row['candidates'] = [{'operation': asdict(op), **d} for op, d in candidates]
    if not candidates:
        row['reason'] = 'utility_no_local_candidate'
        return value
    witnesses = legal[-max(2, ctl.config.verify_per_point):]
    split = max(1, len(witnesses) // 2)
    row['history'] = [{'identity': w.identity, 'verified_at': w.verified_at,
                       'verified_prefix': ctl.witness_prefixes[w.identity]} for w in witnesses]
    op, diagnostics = candidates[0]
    trial = dict(operation=asdict(op), **diagnostics, accepted=False)
    tick, staged, before, after = time.perf_counter(), None, [], []
    ledger = deepcopy(ctl.utility)
    gain = 0.
    try:
        before = ctl.replay(witnesses, ctl.trees, ctl.session.routing_summaries, message.ts)
        original = ctl.trees[op.tree_id]
        ctl._fault('candidate')
        candidate, delta = preview(original, op, local=True)
        staged = stage_summary(ctl.session.routing_summaries, {op.tree_id: candidate}, delta)
        ctl._fault('before_summary')
        trial['summary_cost'] = staged.refresh(trees={op.tree_id: candidate}, log=ctl.session.log)
        ctl._fault('after_summary')
        if not valid_summaries(staged, {op.tree_id: candidate}, delta):
            trial['reason'] = 'incomplete_candidate_summary'
        else:
            after = ctl.replay(witnesses, {**ctl.trees, op.tree_id: candidate}, staged, message.ts)
            ok, reason, generation_gain, check_gain = paired_acceptance(
                before[:split], after[:split], before[split:], after[split:])
            gain = paired_gain(before, after)
            trial.update(accepted=ok, reason=reason, utility_gain=gain,
                         generation_gain=generation_gain, check_gain=check_gain)
            if ok:
                validate_local(original, candidate)
                if ctl.trees[op.tree_id].version != delta.before_version:
                    raise ValueError('stale utility trial')
                if cfg.mode == 'online':
                    value.update(tree=tree_patch(candidate), summary=summary_patch(staged), delta=asdict(delta))
                    row.update(action=op.kind.upper(), operation=asdict(op),
                               before_version=delta.before_version, after_version=delta.after_version)
                    ctl._accepted_trial = staged
                else:
                    trial['shadow_only'] = True
    except OSError:
        raise
    except Exception as exc:
        trial.update(accepted=False, reason='utility_trial_failed', error=repr(exc))
    finally:
        elapsed = time.perf_counter() - tick
        trial['seconds'] = elapsed
        trial['request_keys'] = sorted(staged.requests.changed) if staged else []
        row['request_keys'].extend(trial['request_keys'])
        row['trials'] = [trial]
    if before and after:
        ledger.observe(action=action(op), dependency='', query_ids=[w.identity for w in witnesses],
            prefix=fingerprint([ctl.prefix, seq, ctl.trees[op.tree_id].version]),
            known_at=message.ts, available_at=message.ts,
            before_ids=[list(r.final_ids) for r in before], after_ids=[list(r.final_ids) for r in after],
            gain=gain, seconds=elapsed)
    else:
        ledger.seconds += elapsed
    row['reason'] = trial.get('reason', 'utility_keep')
    value['utility'] = ledger.state()  
    return value
