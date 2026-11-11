from __future__ import annotations
from dataclasses import asdict
import hashlib
import resource
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
REPO = Path(__file__).resolve().parents[1]
import numpy as np
from evoforestmem.forest.builder import ForestBuilder
from evoforestmem.forest.placement import PlacementMap
from evoforestmem.forest.store import TreeStore
from evoforestmem.forest.types import DEFAULT_FOREST_CONFIG
from evoforestmem.forest.update import PeriodUpdater
from evoforestmem.index.dense import DenseIndex
from evoforestmem.log.store import MessageLog
from evoforestmem.log.validity import ValidityLedger
from evoforestmem.stream.session import StreamingSession
MERGED_ENV = REPO

def _to_ts(time_str: str) -> float:
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    import tempfile as _tmp
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = _tmp.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write('\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

def recall(cited: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(cited & gold) / len(gold)

def run(batch: str, *, limit_qa: int | None=None, summary_key_env: str='ABLATION_LLM_API_KEY', point_stride: int=1, render_display_names: bool | None=None, render_seconds: bool | None=None, overrides: dict[str, dict[str, Any]] | None=None, progress_out: Path | None=None, qwen_tokenizer: str | None=None, qwen_base_url: str | None=None, qwen_model: str | None=None, closure_config_path: Path | None=None, card_policy: str='fixed', card_snapshot_dir: Path | None=None, reuse_card_snapshots: bool=False, embedding_device: str | None=None, revision_events_path: Path | None=None, artifact_routing: str='auto', artifact_min_messages: int=2, routing_summary_config_path: Path | None=None, summary_cache_dir: Path | None=None, routing_summary_out: Path | None=None, summary_concurrency: int | None=None, summary_refresh_policy: str='force', summary_refresh_interval: int | None=None, summary_profile: str='legacy', resume_from_progress: bool=False, point_observer: Any=None) -> dict[str, Any]:
    with_summaries = True
    summary_mode = 'hierarchical_full'
    retriever_mode = 'forest_first'
    analyser_mode = 'off'
    planner_mode = 'off'
    dense_mode = 'on'
    lsm_max_delta = 0
    seal_on_append = False
    per_view_k = None
    per_tree_k = None
    full_raw_search = False
    embedding_base_url = None
    hierarchical_summaries = True
    budgeted_summaries = False
    routing_summaries = True
    t_start = time.perf_counter()
    closure_config = evidence_planner = token_counter = card_client = None
    routing_summary_client = routing_summary_config = None
    hierarchical_summaries = with_summaries and summary_mode == 'hierarchical_full'
    budgeted_summaries = with_summaries and retriever_mode == 'forest_first' and (summary_mode in {'auto', 'budgeted'})
    routing_summaries = budgeted_summaries or hierarchical_summaries
    import yaml
    from evoforestmem.retrieve.evidence_planner import LocalQwenClient, QwenTokenCounter
    if not qwen_tokenizer or not qwen_base_url or (not qwen_model):
        raise ValueError('budgeted summaries require --qwen-tokenizer, --qwen-base-url and --qwen-model')
    token_counter = QwenTokenCounter(qwen_tokenizer)
    routing_summary_client = LocalQwenClient(base_url=qwen_base_url, model=qwen_model, timeout=600)
    from evoforestmem.forest.hierarchical_summaries import HierarchicalSummaryConfig
    values = yaml.safe_load(routing_summary_config_path.read_text()) if routing_summary_config_path else {}
    values = values or {}
    if summary_concurrency is not None:
        values['concurrency'] = summary_concurrency
    values['refresh_policy'] = summary_refresh_policy
    values['summary_profile'] = summary_profile
    if summary_refresh_interval is not None:
        values['checkpoint_stride'] = summary_refresh_interval
    routing_summary_config = HierarchicalSummaryConfig(**values)
    revision_events = defaultdict(list)
    if revision_events_path is not None:
        for line in revision_events_path.read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                if not all((event.get(k) for k in ('trigger_msg_id', 'target_msg_id', 'subject'))):
                    raise ValueError('revision events require trigger_msg_id, target_msg_id and scoped subject')
                revision_events[event['trigger_msg_id']].append(event)
    log_dir = REPO / 'data' / 'log' / batch
    index_dir = REPO / 'data' / 'index' / batch
    log = MessageLog.load(log_dir / 'messages.jsonl')
    messages = sorted(log, key=lambda m: (m.ts, m.seq))
    print(f'batch {batch}: {len(messages)} messages')
    emb_dir = index_dir / 'embeddings'
    emb_ids = json.loads((emb_dir / 'ids.json').read_text())
    emb_vecs = np.load(emb_dir / 'vectors.npy')
    dense_vectors = {mid: emb_vecs[i] for i, mid in enumerate(emb_ids)}
    print(f'precomputed dense vectors: {len(dense_vectors)}')
    id_map = json.loads((index_dir / 'dataset_id_map.json').read_text())
    qa_path = index_dir / 'qa_with_time.jsonl'
    if not qa_path.is_file():
        raise FileNotFoundError('Prepared QA missing; run prepare first')
    qa_rows = [json.loads(l) for l in qa_path.read_text().splitlines() if l.strip()]
    if limit_qa:
        qa_rows = qa_rows[:limit_qa]
    gold_by_qa: dict[str, set[str]] = {}
    unresolved = []
    for row in qa_rows:
        poly_gold: set[str] = set()
        for mid in row.get('evidence_msg_ids') or []:
            poly = id_map.get(mid)
            if poly is not None:
                poly_gold.add(poly)
        if poly_gold:
            gold_by_qa[row['qa_id']] = poly_gold
        else:
            unresolved.append(row['qa_id'])
    print(f'QA: {len(qa_rows)}; gold resolvable: {len(gold_by_qa)}; no-evidence/unresolved: {len(unresolved)}')
    points: dict[str, list[dict]] = defaultdict(list)
    for row in qa_rows:
        points[row['ask_time']].append(row)
    ordered_points = sorted(points, key=_to_ts)
    print(f'ask_time points: {len(ordered_points)}')
    from evoforestmem.index.embedder import LocalEmbedder
    from evoforestmem.retrieve.pipeline import PipelineConfig, apply_corpus_profile, load_yaml
    raw_cfg = load_yaml(None) or {}
    if globals().get('EMBEDDING_MODEL'):
        raw_cfg.setdefault('embeddings', {})['model'] = EMBEDDING_MODEL
    raw_cfg = apply_corpus_profile(raw_cfg, batch.replace('group_', 'group-').replace('ever_', 'ever'))
    if overrides:
        for section, kv in overrides.items():
            sec = dict(raw_cfg.get(section) or {})
            sec.update(kv)
            raw_cfg[section] = sec
    s4_over: dict[str, bool] = {}
    if render_display_names is not None:
        s4_over['render_display_names'] = bool(render_display_names)
    if render_seconds is not None:
        s4_over['render_seconds'] = bool(render_seconds)
    session_config = None
    if s4_over:
        s4 = dict(raw_cfg.get('s4') or {})
        s4.update(s4_over)
        raw_cfg['s4'] = s4
    if overrides or s4_over:
        session_config = PipelineConfig.from_dict(raw_cfg)
    eff_render_names = s4_over.get('render_display_names') if s4_over else None
    if eff_render_names is None:
        eff_render_names = PipelineConfig.from_dict(raw_cfg).render_display_names
    render_names = None
    if eff_render_names:
        from evoforestmem.log.registry import Registry
        channel_reg = Registry.load(log_dir / 'channels.json')
        speaker_reg = Registry.load(log_dir / 'speakers.json')
        render_names = ({k: channel_reg.display(k) for k in channel_reg.keys}, {k: speaker_reg.display(k) for k in speaker_reg.keys})
    emb_cfg = raw_cfg.get('embeddings') or {}
    node_device = embedding_device or str(emb_cfg.get('node_device', 'mps'))
    if node_device == 'auto':
        node_device = None
    embedder = None
    embed_type = LocalEmbedder
    embedder = embed_type(str(emb_cfg.get('model', 'Qwen/Qwen3-Embedding-0.6B')), device=node_device, batch_size=int(emb_cfg.get('node_batch', 16)), cache_dir=index_dir / 'embeddings_cache', **{})
    print(f'embedder device: {(embedder.device if embedder is not None else 'disabled')}')
    out_dir = REPO / 'state'
    out_dir.mkdir(parents=True, exist_ok=True)
    up_log = MessageLog()
    up_placement = PlacementMap()
    up_store = TreeStore(f'{batch}-stream', root=out_dir / 'forest')
    up_ledger = ValidityLedger()
    forest_config = None
    builder = ForestBuilder(store=up_store, placement=up_placement, config=forest_config or DEFAULT_FOREST_CONFIG, per_tree_k=per_tree_k)
    updater = PeriodUpdater(log=up_log, store=up_store, placement=up_placement, ledger=up_ledger, builder=builder)
    summariser = None
    analyser = None
    analyser_source = 'off'
    session = StreamingSession(updater, embedder=embedder, dense=DenseIndex(embedder.dim, model=embedder.name) if embedder is not None else None, dense_vectors=dense_vectors, summariser=summariser, config=session_config, render_names=render_names, retriever_mode=retriever_mode, artifact_routing=artifact_routing, artifact_min_messages=artifact_min_messages, closure_config=closure_config, evidence_planner=evidence_planner, count_tokens=token_counter, card_client=card_client, card_policy=card_policy, card_snapshot_dir=card_snapshot_dir, reuse_card_snapshots=reuse_card_snapshots, routing_summary_client=routing_summary_client, routing_summary_config=routing_summary_config, routing_summary_cache=summary_cache_dir, summary_mode=summary_mode, full_raw_search=full_raw_search, analyser=analyser, corpus=batch)
    if point_observer is not None:
        point_observer.setup(session, token_counter)
    rows_out: list[dict[str, Any]] = []
    b_leaks = 0
    next_msg = 0
    refresh_calls = 0
    compaction_reports: list[dict[str, Any]] = []
    search_latencies: list[float] = []
    refresh_ledger: list[dict[str, Any]] = []
    point_timing: dict[str, dict[str, Any]] = {}
    completed_points: list[str] = []

    def _summary_stats() -> dict[str, Any]:
        if session.routing_summaries is not None:
            return dict(session.routing_summaries.stats)
        if summariser is None:
            return {}
        return summariser.stats.to_dict()
    resume_points = 0
    if resume_from_progress and hierarchical_summaries and (progress_out is not None):
        partial_path = progress_out.resolve().with_name('retrieval.partial.json')
        if partial_path.exists() and progress_out.exists():
            partial = json.loads(partial_path.read_text())
            progress = partial.get('resume') or json.loads(progress_out.read_text())
            prior_config = progress.get('routing_summaries', {}).get('config', {})
            if prior_config.get('summary_profile', 'legacy') != summary_profile:
                raise ValueError('resume summary profile mismatch')
            if summary_profile == 'compact_v1':
                for name in ('checkpoint_stride', 'refresh_policy', 'scheduler', 'source_weight'):
                    if prior_config.get(name) != getattr(routing_summary_config, name):
                        raise ValueError('resume summary configuration mismatch: ' + name)
            if progress['batch'] != batch:
                raise ValueError('resume batch mismatch')
            previous_rows = partial['rows']
            ids = [r['qa_id'] for r in previous_rows]
            if len(set(ids)) != len(ids):
                raise ValueError('duplicate saved retrieval questions')
            previous_by_id = {r['qa_id']: r for r in previous_rows}
            for point in progress['completed_points']:
                if point != ordered_points[resume_points]:
                    raise ValueError('resume points are not a complete prefix')
                if not all((r['qa_id'] in previous_by_id for r in points[point])):
                    break
                resume_points += 1
            completed_points = ordered_points[:resume_points]
            for point in completed_points:
                for qa in points[point]:
                    row = previous_by_id[qa['qa_id']]
                    if row['ask_time'] != point or row['question'] != qa['question'] or row['b_leak']:
                        raise ValueError('saved retrieval identity/visibility mismatch')
                    if any((mid not in log or log.get(mid).ts > _to_ts(point) or log.text(mid) not in row['b_context'] for mid in row['b_cited'])):
                        raise ValueError('saved retrieval is not causal original evidence')
                    row.setdefault('summary_refresh_policy', 'force')
                    rows_out.append(row)
            point_timing = {p: progress.get('per_point', {}).get(p, {}) for p in completed_points}
            search_latencies = [r['search_ms'] for r in rows_out]
            if completed_points:
                last_ts = _to_ts(completed_points[-1])
                while next_msg < len(messages) and messages[next_msg].ts <= last_ts:
                    m = messages[next_msg]
                    session.ingest(m, explicit_events=revision_events.get(m.msg_id, ()))
                    next_msg += 1
                checkpoint = progress.get('summary_checkpoint') if next_msg == progress['next_msg'] else None
                if point_observer is not None:
                    point_observer.restore(progress.get('structural_state'))
                store = session.routing_summaries
                try:
                    if session.stream_controller is not None:
                        restored = session.stream_controller.restore_summary_boundary(checkpoint)
                    else:
                        restored = store.restore(trees=up_store.trees, log=up_log, checkpoint=checkpoint, prior_report=progress.get('routing_summaries'))
                    print('restored summary prefix: ' + json.dumps(restored), flush=True)
                except (ValueError, OSError, KeyError) as exc:
                    print('summary restore rejected: ' + repr(exc), flush=True)
                    if point_observer is not None and point_observer.enabled:
                        raise ValueError('verified structure requires its matching summary checkpoint') from exc
                    _atomic_json(progress_out.with_name('resume_failure.json'), {'error': repr(exc), 'messages': next_msg, 'completed_questions_preserved': len(rows_out)})
                    if checkpoint is not None:
                        store.restore(trees=up_store.trees, log=up_log, prior_report=progress.get('routing_summaries'))
                    else:
                        from dataclasses import replace
                        store.cards.clear()
                        for tree in up_store.trees.values():
                            for node in tree.nodes.values():
                                if node.level:
                                    node.summary_dirty = True
                                    tree.dirty_internal_node_ids_by_level.setdefault(node.level, set()).add(node.node_id)
                        original = store.config
                        store.config = replace(original, refresh_policy='force')
                        store.refresh(trees=up_store.trees, log=up_log)
                        store.config = original
                print(f'resume: {len(rows_out)} existing contexts preserved; prefix={next_msg}', flush=True)
                refresh_calls = store.stats['calls']
    for point_idx, point in enumerate(ordered_points):
        if point_idx < resume_points:
            continue
        point_started = time.perf_counter()
        summary_seconds_before = session.routing_summaries.stats['elapsed_sec']
        previous_next_msg = next_msg
        ask_ts = _to_ts(point)
        while next_msg < len(messages) and messages[next_msg].ts <= ask_ts:
            current_message = messages[next_msg]
            events = revision_events.get(current_message.msg_id, ())
            for event in events:
                if event['target_msg_id'] not in up_log:
                    raise ValueError('revision event references an unavailable target')
            session.ingest(current_message, explicit_events=events)
            next_msg += 1
            if routing_summaries and next_msg % routing_summary_config.checkpoint_stride == 0:
                delta = session.refresh_routing_summaries()
                refresh_calls += delta.get('calls', 0)
                refresh_ledger.append({'messages_ingested': next_msg, **delta})
        if point_idx % point_stride != 0:
            continue
        delta = session.refresh_routing_summaries()
        refresh_calls += delta.get('calls', 0)
        refresh_ledger.append({'ask_time': point, 'messages_ingested': next_msg, **delta, 'coverage': session.routing_summaries.report()['coverage']})
        ingest_s = time.perf_counter() - point_started
        ingested_ids = None
        retriever = session.retriever()
        if point_observer is not None:
            retriever.capture_ranked_ids = point_observer.enabled
        for row in points[point]:
            qa_id = row['qa_id']
            t0 = time.perf_counter()
            res_b = retriever.search(row['question'], **{'version_time': ask_ts})
            search_ms = (time.perf_counter() - t0) * 1000.0
            if point_observer is not None:
                point_observer.answered(row, res_b)
            import hashlib
            trace = res_b.stats['routing_summaries']
            for name in ('candidate_ids', 'outside_seed_ids'):
                values = trace.pop(name)
                trace[name + '_count'] = len(values)
                trace[name + '_sha256'] = hashlib.sha256(json.dumps(values).encode()).hexdigest()
            search_latencies.append(search_ms)
            cited_b = set(res_b.provenance)
            if ingested_ids is None:
                ingested_ids = {m.msg_id for m in up_log}
            leak = [m for m in cited_b if m not in ingested_ids]
            if leak:
                b_leaks += 1
            g = gold_by_qa.get(qa_id, set())
            rows_out.append({'qa_id': qa_id, 'unified_category': row.get('unified_category'), 'category': row.get('category'), 'subcategory': row.get('subcategory'), 'question_type': row.get('question_type'), 'ask_time': point, 'question': row.get('question'), 'gold_answer': row.get('answer'), 'b_cited': sorted(cited_b), 'b_context': res_b.context, 'b_leak': bool(leak), 'b_recall': recall(cited_b, g), 'complete_evidence': g <= cited_b if g else None, 'evidence_diagnostic_status': 'dataset references; not adjudicated audit', 'gold_evidence_ids': sorted(g), 'future_gold_ids': sorted((mid for mid in g if mid in log and log.get(mid).ts > ask_ts)), 'retrieval_stats': res_b.stats, 'search_ms': round(search_ms, 2), 'summary_refresh_policy': summary_refresh_policy, 'summary_profile': summary_profile, 'summary_refresh_interval': routing_summary_config.checkpoint_stride})
        if point_observer is not None:
            point_observer.end_batch(ask_ts)
        completed_points.append(point)
        point_timing[point] = {'ingest_s': ingest_s, 'n_msgs': next_msg - previous_next_msg, 'summary_s': session.routing_summaries.stats['elapsed_sec'] - summary_seconds_before, 'ingest_includes_summary': routing_summaries, 'n_qas': len(points[point]), 'wall_s': time.perf_counter() - point_started}
        if progress_out is not None:
            if point_observer is not None and hasattr(point_observer, 'before_checkpoint'):
                point_observer.before_checkpoint()
            progress = {'schema_version': 'evoforestmem.retrieval-progress.v1', 'batch': batch, 'completed_points': completed_points, 'next_msg': next_msg, 'per_point': point_timing, 'routing_summaries': session.routing_summaries.report(), 'note': 'CPU forest replay with prefix-validated summary restore; partial journal is the commit record.'}
            if hierarchical_summaries and resume_from_progress:
                import uuid
                state_path = progress_out.with_name(f'summary-state-{next_msg}-{uuid.uuid4().hex}.json')
                progress['summary_checkpoint'] = session.routing_summaries.checkpoint(state_path, trees=up_store.trees, log=up_log)
            if point_observer is not None:
                progress['structural_state'] = point_observer.checkpoint()
            _atomic_json(progress_out.resolve().with_name('retrieval.partial.json'), {'rows': rows_out, 'resume': progress})
            _atomic_json(progress_out.resolve(), progress)
            if point_observer is not None and hasattr(point_observer, 'after_checkpoint'):
                point_observer.after_checkpoint()
        print(f'point {point_idx + 1}/{len(ordered_points)}: {next_msg} messages, {len(rows_out)} retrieval rows', flush=True)
    print(f'streaming done: {next_msg} messages ingested, {len(rows_out)} answers, causal violations: {b_leaks}')
    internal_nodes_by_view: dict[str, int] = defaultdict(int)
    for tree in up_store.trees.values():
        for node in tree.nodes.values():
            if node.level > 0:
                internal_nodes_by_view[tree.view] += 1
    internal_nodes_by_view = dict(sorted(internal_nodes_by_view.items()))
    lat = np.asarray(search_latencies, dtype=np.float64) if search_latencies else np.zeros(0)
    by_cat: dict[str, dict[str, float]] = defaultdict(lambda: {'n': 0, 'recall': 0.0})
    by_ucat: dict[str, dict[str, float]] = defaultdict(lambda: {'n': 0, 'recall': 0.0})
    by_sub: dict[str, dict[str, float]] = defaultdict(lambda: {'n': 0, 'recall': 0.0})
    for row in rows_out:
        for table, key in ((by_cat, 'category'), (by_ucat, 'unified_category'), (by_sub, 'subcategory')):
            k = row.get(key) or '?'
            table[k]['n'] += 1
            table[k]['recall'] += row['b_recall']
    sum_recall = sum((r['b_recall'] for r in rows_out if gold_by_qa.get(r['qa_id'])))
    n_recall = sum((1 for r in rows_out if gold_by_qa.get(r['qa_id'])))
    implementation_hash = hashlib.sha256()
    for path in sorted((REPO / 'evoforestmem').rglob('*.py')):
        implementation_hash.update(str(path.relative_to(REPO)).encode())
        implementation_hash.update(path.read_bytes())
    summary = {'implementation_sha256': implementation_hash.hexdigest(), 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == 'darwin' else 1), 'schema_version': 'evoforestmem.update-dataset-asktime.v1', 'batch': batch, 'analyser': analyser_source, 'zero_gateway_calls': not with_summaries and (not (retriever_mode == 'evidence_closure' and planner_mode == 'on')), 'retriever_mode': retriever_mode, 'full_raw_search': full_raw_search or hierarchical_summaries, 'summary_refresh_policy': summary_refresh_policy, 'summary_profile': summary_profile, 'summary_policy_counts': dict(__import__('collections').Counter((row.get('summary_refresh_policy') for row in rows_out))), 'dense_mode': dense_mode, 'revision_event_source': str(revision_events_path) if revision_events_path else None, 'revision_links': {'claims': len(up_ledger), 'resolved': sum((c.superseded_by is not None and c.relation_source in (None, 'explicit', 'benchmark_given') for c in up_ledger.claims.values())), 'note': 'Strict mode does not invent replacements from marker or lexical similarity; absent supported links means revision coverage is unestablished.'}, 'embedding_device': node_device, 'embedding_base_url': embedding_base_url, 'embedding_cost': embedder.stats().to_dict() if embedder is not None else None, 'effective_config': asdict(closure_config) if closure_config is not None else session.config.to_dict(), 'session': session.stats(), 'evidence_card_cost': dict(session.card_store.stats) if session.card_store is not None else None, 'routing_summary_cost': dict(session.routing_summaries.stats), 'planner_cost': {'calls': sum((r['retrieval_stats'].get('trace', {}).get('planner', {}).get('calls', 0) for r in rows_out)), 'prompt_tokens': sum((r['retrieval_stats'].get('trace', {}).get('planner', {}).get('usage', {}).get('prompt_tokens', 0) for r in rows_out)), 'completion_tokens': sum((r['retrieval_stats'].get('trace', {}).get('planner', {}).get('usage', {}).get('completion_tokens', 0) for r in rows_out)), 'cached_tokens': sum((r['retrieval_stats'].get('trace', {}).get('planner', {}).get('usage', {}).get('cached_tokens', 0) for r in rows_out))}, 'answer_cost': None, 'answer_cost_note': 'Retrieval-only replay; answer/judge costs belong to their separate runs.', 'corpus': {'messages': len(messages), 'dense_vectors': len(dense_vectors)}, 'qa': {'total': len(qa_rows), 'gold_resolvable': len(gold_by_qa), 'no_evidence_or_unresolved': len(unresolved), 'ask_time_points': len(ordered_points), 'points_answered': sum((1 for i in range(len(ordered_points)) if i % point_stride == 0))}, 'model_b_streaming': {'answers': len(rows_out), 'causal_violations': b_leaks, 'mean_gold_recall': round(sum_recall / n_recall, 4) if n_recall else None, 'with_summaries': with_summaries, 'summary_llm_calls': refresh_calls, 'point_stride': point_stride, 'seal_on_append': seal_on_append, 'lsm_max_delta_messages': lsm_max_delta, 'per_view_k': per_view_k or {v.view: v.k for v in DEFAULT_FOREST_CONFIG.views}, 'per_tree_k_count': 0, 'internal_nodes_by_view': internal_nodes_by_view, 'latency_by_embedding_cache': {label: {'n': len(values), 'p50_ms': float(np.percentile(values, 50)) if values else None, 'p95_ms': float(np.percentile(values, 95)) if values else None} for label, values in {'cache_hit': [r['search_ms'] for r in rows_out if r['retrieval_stats'].get('trace', {}).get('embedding_cache', {}).get('hits', 0) > 0 and r['retrieval_stats'].get('trace', {}).get('embedding_cache', {}).get('misses', 0) == 0], 'cache_miss': [r['search_ms'] for r in rows_out if r['retrieval_stats'].get('trace', {}).get('embedding_cache', {}).get('misses', 0) > 0]}.items()}, 'search_latency_ms': {'p50': round(float(np.percentile(lat, 50)), 2) if len(lat) else None, 'p95': round(float(np.percentile(lat, 95)), 2) if len(lat) else None, 'mean': round(float(lat.mean()), 2) if len(lat) else None}, 'summary_cost': {**session.routing_summaries.stats, 'cost_usd': None, 'per_refresh': refresh_ledger, 'mode': 'hierarchical_full'}}, 'by_category': {cat: {'n': int(v['n']), 'recall': round(v['recall'] / v['n'], 4)} for cat, v in sorted(by_cat.items())}, 'by_unified_category': {cat: {'n': int(v['n']), 'recall': round(v['recall'] / v['n'], 4)} for cat, v in sorted(by_ucat.items())}, 'by_subcategory': {cat: {'n': int(v['n']), 'recall': round(v['recall'] / v['n'], 4)} for cat, v in sorted(by_sub.items())}, 'elapsed_sec': round(time.perf_counter() - t_start, 1), 'timing': {'per_point': point_timing}}
    if routing_summaries and routing_summary_out is not None:
        session.routing_summaries.save(routing_summary_out)
    return (summary, rows_out)
