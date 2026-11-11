from __future__ import annotations
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import time
from .answer_protocol import answer_messages, judge_messages, parse_mc, question_type
from .contracts import CompletedQuery
from .feedback import source_version
from .io import load_object, save_object, write_json
from .requests import RequestJournal, SummaryJournalClient, response_usage

class ExperimentHooks:

    def __init__(self, directory, config, *, answer_base_url, answer_model, judge_base_url, judge_model, request_sources=(), cache_only=False, qa_rows=(), run_identity='', association_config=None, streaming_config=None, utility_config=None):
        self.directory, self.config = (Path(directory), config)
        self.answer_model, self.judge_model = (answer_model, judge_model)
        self.journal = RequestJournal(self.directory / 'requests', answer_base_url, sources=request_sources, cache_only=cache_only)
        self.judge_journal = RequestJournal(self.directory / 'judging_requests', judge_base_url, sources=request_sources, cache_only=cache_only)
        self.qas = {r['qa_id']: r for r in qa_rows}
        self.run_identity = run_identity
        self.streaming_config = streaming_config
        from .online_utility import OnlineUtilityConfig
        self.utility_config = utility_config or OnlineUtilityConfig()
        self.answers, self.pending, self.timings = ({}, [], [])
        self.session = self.controller = self.feedback = None
        from ..associations.contracts import AssociationConfig
        self.association_config = association_config or AssociationConfig()
        self.associations, self.pending_traces = (None, [])

    @property
    def enabled(self):
        return self.config.structure_update_mode != 'off'

    def setup(self, session, token_counter):
        self.session = session
        if self.association_config.navigation != 'off' and session.config.retrieval_strategy != 'hierarchical_v1':
            raise ValueError('association navigation requires hierarchical_v1 retrieval')
        self.counter = token_counter
        self.client = session.routing_summaries.client
        session.routing_summaries.client = SummaryJournalClient(self.client, self.journal)
        from ..associations.recall_audit import RecallAudit
        feedback_cls, extra = RecallAudit, {}
        feedback_utility = replace(self.utility_config, token_reference=0.1)
        self.feedback = feedback_cls(self.config, self.client, self.journal, token_counter, utility_config=feedback_utility, **extra)
        if self.enabled:
            from .streaming import StreamingController
            self.controller = StreamingController(session, self.config, self.directory / 'structure', streaming_config=self.streaming_config, run_identity=self.run_identity, association_config=self.association_config, utility_config=self.utility_config)
        from ..associations.recall_controller import RecallController as AssociationController
        self.associations = AssociationController(session, self.association_config, self.controller, utility_config=self.utility_config)
        self.associations.recognizer = self.feedback

    def summary_refreshed(self):
        for key, row in self.session.routing_summaries.requests.items():
            if row.get('response', {}).get('usage'):
                self.feedback.credit('summary:' + key, row['response']['usage'])

    def answered(self, row, result):
        if row['qa_id'] not in self.qas:
            raise ValueError('Answer requested outside the selected QA scope')
        started = time.perf_counter()
        qa = {**row, **self.qas.get(row['qa_id'], {})}
        payload = {'model': self.answer_model, 'messages': answer_messages(qa, result.context), 'temperature': 0, 'max_tokens': 1000, 'chat_template_kwargs': {'enable_thinking': False}}
        response, key, hit = self.journal.complete(payload, role='answer')
        if response['choices'][0]['finish_reason'] != 'stop':
            raise ValueError('answer response exhausted its historical output limit: ' + key)
        text = response['choices'][0]['message']['content'].strip()
        kind = question_type(qa)
        if kind == 'multiple_choice':
            text = parse_mc(text)
        self.feedback.credit('answer:' + key, response['usage'])
        ids = tuple(result.provenance)
        from datetime import datetime, timezone
        parsed = datetime.fromisoformat(row['ask_time'].replace('Z', '+00:00'))
        stamp = (parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)).timestamp()
        record = {'qa_id': row['qa_id'], 'question': row['question'], 'ask_time': row['ask_time'], 'question_type': kind, 'answer': qa['answer'], 'generated_answer': text, 'context': result.context, 'source_order': ids, 'source_versions': {mid: source_version(self.session.log.get(mid)) for mid in ids}, 'request_key': key, 'request_payload': payload, 'answer_cache_hit': hit, 'answer_seconds': time.perf_counter() - started, 'is_correct': text == str(qa['answer']).strip().upper() if kind == 'multiple_choice' else None}
        self.answers[row['qa_id']] = record
        if 'navigation' in result.stats:
            record['navigation'] = result.stats['navigation']
            record['retrieval_seconds'] = result.stats['elapsed_ms'] / 1000.0
        record['stream_snapshot'] = {'messages': len(self.session.log), 'prefix': self.controller.prefix}
        if self.associations is not None:
            query = CompletedQuery(row['question'], stamp, row.get('asker'), ids, ())
            trace = result.stats.get('association')
            if trace is None:
                raise ValueError('association experiment requires a captured forest_first trace')
            self.pending_traces.append({**trace, 'identity': query.identity, 'question': query.question, 'ask_time': stamp, 'asker': query.asker})
            record['association'] = {k: trace.get(k) for k in ('triggered', 'selected_groups', 'evidence_changed', 'context_changed', 'reason', 'seconds', 'error', 'trigger_funnel')}
            if trace.get('policy') in {'shortcut_v2', 'evidence_v3'}:
                record['association'].update({k: trace.get(k) for k in ('policy', 'observation_kind', 'shortcut_selected_ids', 'counterfactual_measured', 'shortcut_features', 'packing_changed', 'verified_coverage_before', 'verified_coverage_after', 'verified_support_ids')})
            if trace.get('policy') == 'associative_recall_v4':
                record['association'] = dict(trace)
        if self.enabled:
            self.pending.append(CompletedQuery(row['question'], stamp, row.get('asker'), ids, tuple(result.stats['feedback_ranked_ids'])))

    def end_batch(self, known_at):
        self.summary_refreshed()
        row = {'known_at': known_at, 'feedback_seconds': 0.0, 'structure_seconds': 0.0, 'save_seconds': 0.0}
        row['save_timing_complete'] = False
        if self.enabled:
            tick = time.perf_counter()
            witnesses = self.feedback.after_batch(self.pending, self.session.log, self.session.updater.ledger)
            row['feedback_seconds'] = time.perf_counter() - tick
            self.controller.add_witnesses(witnesses)
            tick = time.perf_counter()
            row['structure_seconds'] = time.perf_counter() - tick
        if self.associations is not None:
            decision = self.associations.maintain(self.pending_traces, known_at)
            row['association_seconds'] = decision['seconds']
            row['association_selection_seconds'] = sum((t.get('seconds', 0.0) for t in self.pending_traces))
            self.pending_traces.clear()
        self.pending.clear()
        row['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.timings.append(row)

    def before_checkpoint(self):
        self._save_started = time.perf_counter()

    def checkpoint(self):
        state = {'run_identity': self.run_identity, 'config': asdict(self.config), 'structure': self.controller.checkpoint() if self.controller else None, 'feedback': self.feedback.state(), 'answers': self.answers, 'timings': self.timings, 'embedding_usage': dict(getattr(self.session.embedder, 'service_usage', {}))}
        state['online_utility'] = self.controller.utility.state()
        if self.associations is not None:
            state['associations'] = self.associations.state()
        reference = save_object(self.directory / 'point_objects', state)
        self._point_reference = reference
        return reference

    def after_checkpoint(self):
        row = self.timings[-1]
        row.update(save_seconds=time.perf_counter() - self._save_started, save_timing_complete=True, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write_json(self.directory / 'save_timings' / (self._point_reference['sha256'] + '.json'), {k: row[k] for k in ('save_seconds', 'save_timing_complete', 'peak_rss_bytes')})

    def restore(self, reference):
        if reference is None:
            raise ValueError('experiment resume requires its atomic point state')
        state = load_object(reference)
        if not set(state['answers']) <= set(self.qas):
            raise ValueError('Checkpoint contains out-of-scope QA')
        if state['run_identity'] != self.run_identity or state['config'] != asdict(self.config):
            raise ValueError('experiment code/data/configuration changed on resume')
        if self.controller:
            self.controller.restore(state['structure'])
            self.controller.utility.restore(state['online_utility'])
        elif state['structure'] is not None:
            raise ValueError('off run cannot restore adaptive trees')
        self.feedback.restore(state['feedback'])
        if self.associations is not None:
            self.associations.restore(state['associations'])
        elif state.get('associations') is not None:
            raise ValueError('association off cannot restore an active association run')
        self.answers, self.timings = (state['answers'], state['timings'])
        timing = self.directory / 'save_timings' / (reference['sha256'] + '.json')
        if self.timings and timing.exists():
            self.timings[-1].update(json.loads(timing.read_text()))
        if hasattr(self.session.embedder, 'service_usage'):
            for key, count in state.get('embedding_usage', {}).items():
                self.session.embedder.service_usage[key] += count

    def finish(self):
        if not set(self.answers) <= set(self.qas):
            raise ValueError('Final results contain out-of-scope QA')
        requests = self.journal.records()
        progress_path = self.directory / 'summary.progress.json'
        driver_timings = json.loads(progress_path.read_text()).get('per_point', {}) if progress_path.exists() else {}
        value = {'schema': 'structural-compact-v2-results', 'run_identity': self.run_identity, 'config': asdict(self.config), 'rows': list(self.answers.values()), 'feedback': self.feedback.state(), 'timings': self.timings, 'stream_timings': driver_timings, 'decisions': self.controller.decisions if self.controller else [], 'trial_costs': self.controller.costs if self.controller else [], 'summary': self.session.routing_summaries.report(), 'embedding_cache': self.session.embedder.stats().to_dict(), 'embedding_service_usage': dict(getattr(self.session.embedder, 'service_usage', {})), 'actual_requests': [{k: v for k, v in r.items() if k not in ('payload', 'response')} | {'usage': response_usage(r.get('response'))} for r in requests]}
        value['online_utility'] = self.controller.utility.state()
        if self.associations is not None:
            value['associations'] = self.associations.state()
        value['association_config'] = asdict(self.association_config)
        from collections import Counter
        value.update(schema='stream-verified-v1-results', streaming_config=self.controller.streaming_config.identity(), streaming={'messages': len(self.session.log), 'prefix': self.controller.prefix, 'action_counts': dict(Counter((r['action'] for r in self.controller.decisions))), 'journal': str(self.controller.journal.directory), 'leaf_vector_reads': self.controller.statistics.leaf_reads})
        write_json(self.directory / 'results.json', value)
        return value

def judge_results(path, journal, judge_model):

    path = Path(path)
    value = json.loads(path.read_text())
    from .answer_protocol import JUDGE_SCHEMA

    def request_for(row):
        return {'model': judge_model, 'messages': judge_messages(row, row['generated_answer']), 'temperature': 0, 'max_tokens': 4096, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'judge_verdict', 'strict': True, 'schema': JUDGE_SCHEMA}}}
    missing = any((journal.cached(request_for(r), vote) is None for r in value['rows'] if r['question_type'] != 'multiple_choice' for vote in range(3)))
    if missing:
        if journal.cache_only:
            raise LookupError('judge cache incomplete; no judging request sent')
        from urllib.request import Request, urlopen
        import os
        with urlopen(Request(journal.base_url + '/models', headers={'Authorization': 'Bearer ' + os.environ.get('EVOFORESTMEM_JUDGE_API_KEY', 'EMPTY')}), timeout=5) as response:
            models = json.load(response)['data']
        if judge_model not in {m['id'] for m in models}:
            raise ValueError('judge service model mismatch')
    for row in value['rows']:
        if row['question_type'] == 'multiple_choice':
            continue
        votes = []
        payload = request_for(row)
        keys = []
        for vote in range(3):
            response, key, _ = journal.complete(payload, role='judge', replicate=vote)
            if response['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('judge response incomplete: ' + key)
            import re
            content = response['choices'][0]['message']['content']
            label = None
            for match in reversed(list(re.finditer('\\{[^{}]*\\}', content))):
                try:
                    value_ = json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
                if isinstance(value_, dict) and 'label' in value_:
                    label = value_['label']
                    if isinstance(label, dict):
                        label = label.get('label')
                    label = label.strip().upper() if isinstance(label, str) else None
                    break
            if label not in {'CORRECT', 'WRONG'}:
                raise ValueError('judge returned an invalid verdict: ' + key)
            votes.append(label == 'CORRECT')
            keys.append(key)
        row.update(is_correct=sum(votes) >= 2, judge_votes=votes, judge_request_keys=keys)
        write_json(path, value)
    value['scoring_complete'] = all((type(r.get('is_correct')) is bool for r in value['rows']))
    value['correct'] = sum((r['is_correct'] for r in value['rows'])) if value['scoring_complete'] else None
    value['actual_judge_requests'] = [{k: v for k, v in r.items() if k not in ('payload', 'response')} | {'usage': response_usage(r.get('response'))} for r in journal.records()]
    write_json(path, value)
    return value
