from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
import threading
import time

import numpy as np

from .contracts import fingerprint
from .controller import valid_summaries
from .demand_controller import DemandController
from .demands import DemandProbe, DemandWitness, coverage
from .feedback import source_version
from .io import load_object
from .operations import preview
from .streaming_journal import SCHEMA, StreamJournal, apply_summary_patch, guard, summary_patch
from .streaming_local import (apply_tree_patch, clone_summary, clone_tree, publish_summary,
                              stage_summary, tree_patch, validate_local)
from .streaming_policy import LocalStatistics, StreamingConfig, admission, propose_local, propose_prefix


class StreamRecoveryRequired(RuntimeError):
    pass


class LockedRetriever:

    def __init__(self, controller):
        object.__setattr__(self, '_controller', controller)
        object.__setattr__(self, '_overrides', {})

    def __setattr__(self, name, value):
        self._overrides[name] = value

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        with self._controller.lock:
            self._controller.ensure_healthy()
            return getattr(self._controller.session._retriever(), name)

    def search(self, *args, **kwargs):
        with self._controller.lock:
            self._controller.ensure_healthy()
            self._controller.refresh_graph_bindings()
            reader = self._controller.session._retriever()
            for key, value in self._overrides.items():
                setattr(reader, key, value)
            return reader.search(*args, **kwargs)


class StreamingController(DemandController):
    def __init__(self, session, config, directory, *, streaming_config=None, run_identity='', association_config=None,
                 utility_config=None):
        super().__init__(session, config, directory)
        if config.structure_update_mode != 'stream_verified_v1':
            raise ValueError('stream controller requires stream_verified_v1')
        if (len(session.log) or session.dense is None or session.retriever_mode != 'forest_first'
                or getattr(session.routing_summaries, 'mode', None) != 'hierarchical_full'):
            raise ValueError('stream mode requires a fresh forest_first session with dense vectors and hierarchical summaries')
        if session.routing_summaries.config.maintenance_mode != 'incremental':
            raise ValueError('stream mode requires incremental summary maintenance')
        self.streaming_config = streaming_config or StreamingConfig()
        from .online_utility import OnlineUtilityConfig, UtilityLedger
        self.utility = UtilityLedger(utility_config or OnlineUtilityConfig())
        if (self.streaming_config.proposal_policy == 'utility_v1') != (self.utility.config.mode != 'off'):
            raise ValueError('utility_v1 streaming requires an explicit enabled utility configuration')
        if self.streaming_config.citation_repair != 'off':
            session.routing_summaries.config = replace(session.routing_summaries.config,
                citation_repair=self.streaming_config.citation_repair)
        self.last_trial_seq = -1
        self.lock = threading.RLock()
        self.failed = False
        self.fault_hook = None
        self.arrivals, self.receipts, self.witness_prefixes = {}, {}, {}
        self.prefix = fingerprint({'schema': SCHEMA, 'empty': True})
        self.pending_commits, self.previous_checkpoint = [], None
        self.statistics = LocalStatistics(session.dense.vector)
        journal_type = StreamJournal
        if self.streaming_config.journal_backend in {'append_v1', 'append_v2'}:
            from .streaming_wal import AppendStreamJournal
            journal_type = AppendStreamJournal
        self.journal = journal_type(self.directory / 'stream', {
            'config': asdict(config), 'streaming_config': self.streaming_config.identity(),
            'association_config': asdict(association_config) if association_config is not None else None,
            'run_identity': run_identity, 'summary_config': session.routing_summaries.config_fingerprint(),
            'retrieval_config': asdict(session.config), 'embedding': session.embedder.name,
            **({'utility_config': asdict(self.utility.config)} if self.utility.config.mode != 'off' else {})})
        session.updater.builder.arrival_order = self.arrivals
        session.stream_controller = self

    def ensure_healthy(self):
        if self.failed:
            raise StreamRecoveryRequired('stream persistence/state failed; replay into a fresh session before reading or writing')

    def _fault(self, stage):
        if self.fault_hook is not None:
            self.fault_hook(stage)

    def checkpoint_schema(self):
        return SCHEMA

    @staticmethod
    def _payload(message, events):
        value = message.to_dict()
        value.pop('seq', None)  
        return {'message': value, 'events': list(events)}

    def _prepared_vector(self, seq, message):
        prepared = self.journal.read_step(seq, 'prepared')
        if prepared is not None:
            return np.asarray(prepared['vector'], dtype=np.float32)
        if self.journal.read_step(seq, 'embedding_started') is not None:
            raise StreamRecoveryRequired('document embedding delivery is unknown; do not resend automatically')
        vector = (self.session.dense_vectors.get(message.msg_id)
                  if self.session.dense_vectors is not None else None)
        if vector is None:
            self.journal.write_step(seq, 'embedding_started', {'msg_id': message.msg_id})
            vector = self.session.embedder.encode([message.text], kind='document')[0]
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError('invalid document embedding')
        self.journal.write_step(seq, 'prepared', {'vector': vector.tolist()})
        return vector

    def ingest(self, message, *, explicit_events=(), message_lookup=None):
        from ..forest.update import UpdateReport
        events = tuple(explicit_events)
        payload = self._payload(message, events)
        digest = fingerprint(payload)
        with self.lock:
            self.ensure_healthy()
            if message.msg_id in self.session.log:
                if self.receipts.get(message.msg_id) != digest:
                    raise ValueError('conflicting redelivery of an existing message ID')
                return UpdateReport(self.session._epoch, 0, 0, 0, 0, False)
            if (not math.isfinite(message.ts) or (len(self.session.log) and
                    message.ts < self.session.log.at(len(self.session.log)-1).ts)):
                raise ValueError('stream_verified_v1 requires nondecreasing timestamps')
            from ..log.store import content_id
            base = content_id(message.channel_key, message.speaker_key, message.ts, message.text)
            expected = (f'{base}~{self.session.log._dup_counter.get(base, 0)+1}'
                        if base in self.session.log else base)
            if message.msg_id != expected:
                raise ValueError('message ID does not match the next immutable raw record')
            for event in events:
                for field in ('target_msg_id', 'trigger_msg_id'):
                    mid = event.get(field)
                    if mid is not None and mid != message.msg_id and mid not in self.session.log:
                        raise ValueError('revision event references an unavailable message')
            seq, started = len(self.session.log), time.perf_counter()
            begin = {'schema': SCHEMA, 'seq': seq, 'prefix_before': self.prefix,
                     'payload': payload, 'payload_sha256': digest}
            existing = self.journal.read_step(seq, 'begin')
            if existing is not None and fingerprint(existing) != fingerprint(begin):
                raise ValueError('stream replay does not match the durable arrival prefix')
            try:
                self._load_boundary(seq)
                cached = (self.session.dense_vectors.get(message.msg_id)
                          if self.session.dense_vectors is not None else None)
                grouped = self.streaming_config.journal_backend == 'append_v2'
                if grouped and existing is None and cached is not None:
                    cached = np.asarray(cached, dtype=np.float32)
                    if cached.ndim != 1 or not np.all(np.isfinite(cached)):
                        raise ValueError('invalid document embedding')
                    self.journal.put_many([(self.journal.step(seq, 'begin'), begin),
                        (self.journal.step(seq, 'prepared'), {'vector': cached.tolist()})])
                else:
                    self.journal.write_step(seq, 'begin', begin)
                self._fault('after_begin')
                vector = self._prepared_vector(seq, message)
                self.arrivals[message.msg_id] = seq
                report = self.session._ingest(message, explicit_events=events,
                    message_lookup=None, stream_vector=vector)
                self._fault('after_append')
                if report.messages != 1:
                    raise ValueError('new stream arrival did not append exactly one raw message')
                anchors = self.statistics.invalidate(self.trees, self.session.updater.builder.last_insertions)
                committed = self.journal.read_step(seq, 'commit')
                decided = self.journal.read_step(seq, 'decision')
                if committed is not None:
                    if decided != committed['decision']:
                        raise ValueError('commit/decision reference mismatch')
                    value = load_object(decided)
                elif decided is not None:
                    value = load_object(decided)
                else:
                    value = self._decide(message, seq, anchors, interrupted=existing is not None)
                    value['decision']['seconds'] = time.perf_counter() - started
                    if grouped:
                        if value['decision']['prefix_before'] != self.prefix or value['decision']['seq'] != seq:
                            raise ValueError('decision prefix mismatch')
                        decided = self.journal.decision(seq, value, commit_prefix=self.prefix,
                                                        payload_sha256=digest)
                    else:
                        decided = self.journal.object(value)
                        self.journal.write_step(seq, 'decision', decided)
                self._fault('after_decision')
                if value['decision']['prefix_before'] != self.prefix or value['decision']['seq'] != seq:
                    raise ValueError('decision prefix mismatch')
                next_prefix = fingerprint([self.prefix, digest, decided['sha256']])
                commit = {'seq': seq, 'decision': decided, 'prefix_after': next_prefix}
                self.journal.write_step(seq, 'commit', commit)
                self._fault('after_commit')
                self._publish(value)
                self.prefix = next_prefix
                self.receipts[message.msg_id] = digest
                self.decisions.append(value['decision'])
                self.costs.extend(value['decision']['trials'])
                self.pending_commits.append(commit)
                if len(self.session.log) % self.streaming_config.checkpoint_stride == 0:
                    self._checkpoint_increment()
                self._fault('before_ack')
                if self.journal.read_step(seq, 'ack') is None:
                    self.journal.write_step(seq, 'ack', {'seq': seq, 'prefix': self.prefix,
                        'seconds_through_commit': time.perf_counter() - started,
                        'replayed': existing is not None,
                        'completed_at': datetime.now(timezone.utc).isoformat()})
                return report
            except BaseException:
                self.failed = True
                raise

    def _decide(self, message, seq, anchors, *, interrupted=False):
        row = {'seq': seq, 'msg_id': message.msg_id, 'arrival_sequence': seq,
               'visible_messages': seq+1, 'known_at': message.ts, 'prefix_before': self.prefix,
               'action': 'KEEP', 'reason': 'recovered_uncommitted_keep' if interrupted else 'no_local_gain',
               'trials': [], 'history': [], 'request_keys': [],
               'started_at': datetime.now(timezone.utc).isoformat()}
        value = {'schema': SCHEMA, 'decision': row, 'tree': None, 'summary': None}
        if interrupted:
            return value
        if self.streaming_config.proposal_policy == 'utility_v1':
            from .utility_streaming import decide
            return decide(self, message, seq, anchors, value)
        fast = self.streaming_config.proposal_policy != 'eager_v1'
        proposer = propose_prefix if fast else propose_local
        candidates = proposer(self.trees, anchors, self.statistics, self.streaming_config,
                              self.config.max_candidates)
        row['candidates'] = [{'operation': asdict(op), **diagnostic} for op, diagnostic in candidates]
        if fast:
            row['admission'] = admission(self.streaming_config, seq, self.last_trial_seq, candidates)
            if not row['admission']['admitted']:
                row['reason'] = row['admission']['reason']
                return value
        if not candidates:
            return value
        legal = [w for w in self.legal_history(message.ts)
                 if self.witness_prefixes.get(w.identity, seq+2) <= seq]
        legal.sort(key=lambda w: (self.witness_prefixes[w.identity], w.verified_at, w.identity))
        witnesses = legal[-self.config.verify_per_point:]
        row['history'] = [{'identity': w.identity, 'verified_prefix': self.witness_prefixes[w.identity],
                           'verified_at': w.verified_at} for w in witnesses]
        try:
            before = self.replay(witnesses, self.trees, self.session.routing_summaries, message.ts)
        except OSError:
            raise
        except Exception as exc:
            row.update(reason='history_baseline_unavailable', error=repr(exc))
            return value
        for op, diagnostics in candidates:
            started, trial = time.perf_counter(), {'operation': asdict(op), **diagnostics, 'accepted': False}
            staged = None
            try:
                self._fault('candidate')
                original = self.trees[op.tree_id]
                candidate, delta = preview(original, op, local=True)
                staged = stage_summary(self.session.routing_summaries, {op.tree_id: candidate}, delta)
                self._fault('before_summary')
                cost = staged.refresh(trees={op.tree_id: candidate}, log=self.session.log)
                trial['summary_cost'] = cost
                self._fault('after_summary')
                if not valid_summaries(staged, {op.tree_id: candidate}, delta):
                    trial['reason'] = 'incomplete_candidate_summary'
                    trial['summary_errors'] = [{'node_id': card.node_id, 'status': card.status,
                        'stale': card.stale, 'error': card.error} for card in staged.cards.changed.values()
                        if card.status not in {'complete', 'inherited'} or card.stale]
                else:
                    trees = dict(self.trees)
                    trees[op.tree_id] = candidate
                    after = self.replay(witnesses, trees, staged, message.ts)
                    accepted, reason = guard(before, after)
                    trial.update(accepted=accepted, reason=reason,
                        changed_witness_contexts=sum(a.context_sha256 != b.context_sha256 for a, b in zip(before, after)))
                    if accepted:
                        validate_local(original, candidate)
                        if self.trees[op.tree_id].version != delta.before_version:
                            raise ValueError('structure version changed during isolated verification')
                        value.update(tree=tree_patch(candidate), summary=summary_patch(staged), delta=asdict(delta))
                        row.update(action=op.kind.upper(), reason=reason, operation=asdict(op),
                                   before_version=delta.before_version, after_version=delta.after_version)
            except OSError:
                raise
            except Exception as exc:
                trial.update(accepted=False, reason='candidate_exception', error=repr(exc))
            finally:
                trial['seconds'] = time.perf_counter() - started
                trial['request_keys'] = sorted(staged.requests.changed) if staged else []
                row['request_keys'].extend(trial['request_keys'])
                row['trials'].append(trial)
            if trial['accepted']:


                self._accepted_trial = staged
                break
            row['reason'] = 'candidates_rejected'
        return value

    def _publish(self, value):
        if value['schema'] != SCHEMA:
            raise ValueError('cannot mix stream and legacy checkpoint schemas')
        if value.get('utility') is not None:
            self.utility.restore(value['utility'])
        if value['decision'].get('admission', {}).get('consumed'):


            self.last_trial_seq = value['decision']['seq']
        if value['tree'] is not None:
            patch = value['tree']
            tree = self.trees[patch['tree_id']]
            if tree.version != value['decision']['before_version']:
                raise ValueError('committed patch precondition mismatch')
            apply_tree_patch(tree, patch)
            staged = getattr(self, '_accepted_trial', None)
            if staged is not None:
                publish_summary(self.session.routing_summaries, staged, self.trees)
                del self._accepted_trial
            else:
                apply_summary_patch(self.session.routing_summaries, value['summary'], self.trees)
            from .contracts import StructureDelta
            self.statistics.changed(StructureDelta(**value['delta']))
            self.session.updater.store.mark_dirty([tree.tree_id])
        graph = getattr(self.session, 'association_graph', None)
        if graph is not None:
            if self.utility.config.mode != 'off':


                self._graph_bindings_dirty = True
            else:
                graph.rebind(self.trees, self.session.updater.placement, self.session.log,
                             self.session.updater.ledger, value['decision']['known_at'])

    def refresh_graph_bindings(self):
        if getattr(self, '_graph_bindings_dirty', False):
            graph = getattr(self.session, 'association_graph', None)
            if graph is not None:
                at = self.session.log.latest_ts
                if graph.config.learning_policy == 'associative_recall_v4' and graph.boundary is not None:
                    at = max(at, graph.boundary)
                graph.rebind(self.trees, self.session.updater.placement, self.session.log,
                             self.session.updater.ledger, at)
            self._graph_bindings_dirty = False

    def replay(self, witnesses, trees, summaries, known_at):
        from copy import deepcopy
        graph = getattr(self.session, 'association_graph', None)
        if graph is not None:
            graph = deepcopy(graph)
            graph.rebind(trees, self.session.updater.placement, self.session.log,
                         self.session.updater.ledger, known_at)
        def one(witness):


            reader = self.session._retriever()
            reader.trees, reader.routing_summaries = trees, summaries
            reader.association_graph = graph
            reader.navigation_centroids = None
            reader.diagnostic_source_ids = witness.evidence
            result = reader.search(witness.question, version_time=known_at)
            ids = tuple(result.provenance)
            demands = coverage(witness, ids)
            return DemandProbe(witness.identity, ids,
                {mid: source_version(self.session.log.get(mid)) for mid in ids}, fingerprint(result.context),
                all(r['complete'] for r in demands.values()), len(witness.evidence & set(ids)),
                len(witness.evidence), summaries.count_tokens(result.context), demands,
                sum(not r['groups'] or r['status'] != 'supported' for r in witness.requirements))
        if not witnesses or self.config.replay_workers == 1:
            return [one(w) for w in witnesses]
        first = one(witnesses[0])
        with ThreadPoolExecutor(max_workers=self.config.replay_workers) as pool:
            return [first, *pool.map(one, witnesses[1:])]

    def add_witnesses(self, witnesses):
        with self.lock:
            self.ensure_healthy()
            fresh = [w for w in witnesses if w.identity not in self.witness_prefixes]
            if not fresh:
                return
            value = {'schema': SCHEMA, 'prefix': len(self.session.log), 'stream_prefix': self.prefix,
                     'witnesses': [asdict(w) for w in fresh]}
            try:
                self.journal.put(f'boundaries/{len(self.session.log):012d}/feedback-{len(self.history):012d}-{fingerprint(value)}', value)
            except BaseException:
                self.failed = True
                raise
            super().add_witnesses(fresh)
            self.witness_prefixes.update({w.identity: len(self.session.log) for w in fresh})

    def _load_boundary(self, prefix, *, feedback=True):
        boundary = self.journal.get(f'boundaries/{prefix:012d}/summary')
        if boundary is not None:
            if boundary['prefix'] != self.prefix:
                raise ValueError('summary boundary prefix mismatch')
            value = load_object(boundary['object'])
            for patch in value['trees']:
                apply_tree_patch(self.trees[patch['tree_id']], patch)
            apply_summary_patch(self.session.routing_summaries, value['summary'], self.trees)
            self.session.routing_summaries._checked_boundary = True
            self.session.routing_summaries._checked_tree_versions = {tid: t.version for tid, t in self.trees.items()}
        if not feedback:
            return
        for name in self.journal.names(f'boundaries/{prefix:012d}', 'feedback-'):
            row = self.journal.get(name)
            if row['stream_prefix'] != self.prefix:
                raise ValueError('feedback boundary prefix mismatch')
            fresh = [DemandWitness(**w) for w in row['witnesses'] if
                     fingerprint([w['question'].strip(), w['ask_time'], w['asker']]) not in self.witness_prefixes]
            super().add_witnesses(fresh)
            self.witness_prefixes.update({w.identity: prefix for w in fresh})
        associations = getattr(self.session, 'association_controller', None)
        if associations is not None:
            for name in sorted(self.journal.names(f'boundaries/{prefix:012d}', 'associations-')):
                row = self.journal.get(name)
                if row['prefix'] != self.prefix:
                    raise ValueError('association boundary prefix mismatch')
                saved = load_object(row['object'])
                if saved['round'] > associations.round:
                    associations.restore(saved)

    def association_boundary(self, key):
        for name in self.journal.names(f'boundaries/{len(self.session.log):012d}', 'associations-'):
            row = self.journal.get(name)
            if row['key'] == key:
                if row['prefix'] != self.prefix:
                    raise ValueError('association prefix mismatch')
                return load_object(row['object'])
        return None

    def publish_associations(self, key, state):

        try:
            reference = self.journal.object(state)
            self._fault('before_association_commit')
            self.journal.put(f'boundaries/{len(self.session.log):012d}/associations-{state["round"]:012d}-{key}',
                {'prefix': self.prefix, 'key': key, 'object': reference})
            self._fault('after_association_commit')
        except BaseException:
            self.failed = True
            raise

    def refresh_summaries(self):
        with self.lock:
            self.ensure_healthy()
            count = len(self.session.log)
            if self.journal.get(f'boundaries/{count:012d}/summary') is not None:
                self._load_boundary(count, feedback=False)
                return {'calls': 0, 'attempts': 0}
            try:
                clones = {tid: clone_tree(tree) for tid, tree in self.trees.items()}
                staged = clone_summary(self.session.routing_summaries, clones)
                cost = staged.refresh(trees=clones, log=self.session.log)
                value = {'schema': SCHEMA, 'trees': [tree_patch(tree) for tree in clones.values()],
                         'summary': summary_patch(staged)}
                reference = self.journal.object(value)
                self.journal.put(f'boundaries/{count:012d}/summary', {'prefix': self.prefix, 'object': reference})
                for patch in value['trees']:
                    apply_tree_patch(self.trees[patch['tree_id']], patch)
                publish_summary(self.session.routing_summaries, staged, self.trees)
                self.session.routing_summaries._checked_boundary = True
                self.session.routing_summaries._checked_tree_versions = {tid: t.version for tid, t in self.trees.items()}
                return cost
            except BaseException:
                self.failed = True
                raise

    def _checkpoint_increment(self):
        self._fault('checkpoint')
        value = {'schema': SCHEMA, 'previous': self.previous_checkpoint,
                 'messages': len(self.session.log), 'prefix': self.prefix, 'commits': self.pending_commits}
        reference = self.journal.object(value)
        self.journal.put(f'checkpoints/{len(self.session.log):012d}-{reference["sha256"]}', reference)
        self.previous_checkpoint, self.pending_commits = reference, []
        return reference

    def checkpoint(self):
        with self.lock:
            self.ensure_healthy()
            try:
                reference = self._checkpoint_increment()
                return self.journal.object({'schema': SCHEMA, 'config': asdict(self.config),
                    'streaming_config': self.streaming_config.identity(), 'messages': len(self.session.log),
                    'prefix': self.prefix, 'increment': reference,
                    'history': [asdict(w) for w in self.history], 'witness_prefixes': self.witness_prefixes})
            except BaseException:
                self.failed = True
                raise

    def restore(self, reference):
        with self.lock:
            self.ensure_healthy()
            state = load_object(reference)
            if (state.get('schema') != SCHEMA or state['config'] != asdict(self.config)
                    or state['streaming_config'] != self.streaming_config.identity()
                    or state['messages'] != len(self.session.log) or state['prefix'] != self.prefix):
                raise ValueError('stream checkpoint schema/config/prefix mismatch')
            self.history = [DemandWitness(**row) for row in state['history']]
            self.witness_prefixes = dict(state['witness_prefixes'])
            self.previous_checkpoint, self.pending_commits = state['increment'], []

    def restore_summary_boundary(self, reference):

        import json
        from pathlib import Path
        from ..forest.hierarchical_summaries import identity, fingerprint as text_fingerprint
        if reference is None:
            raise ValueError('stream resume requires its summary point snapshot')
        text = Path(reference['path']).read_text()
        if text_fingerprint(text) != reference['sha256']:
            raise ValueError('summary point checksum mismatch')
        state = json.loads(text)
        store = self.session.routing_summaries
        if (state['messages'] != len(self.session.log)
                or state['forest'] != store.forest_identity(self.trees)
                or state['prefix'] != identity([(m.msg_id, m.text, m.ts) for m in self.session.log])
                or state['config_fingerprint'] != store.config_fingerprint()):
            raise ValueError('stream summary point identity mismatch')
        self._load_boundary(len(self.session.log), feedback=False)
        current = [asdict(c) for c in store.cards.values()]
        key = lambda c: (c['tree_id'], c['node_id'])
        if fingerprint(sorted(current, key=key)) != fingerprint(sorted(state['cards'], key=key)):
            raise ValueError('stream summary dependencies do not match the saved point')
        store.restored = {'messages': len(self.session.log), 'cards': len(current), 'stream_replay': True}
        return store.restored

    def maintain(self, known_at):
        raise RuntimeError('stream mode makes decisions on ingest, never again at end_batch')

    def recover(self):

        from ..log.message import Message
        if len(self.session.log):
            raise ValueError('recover requires an empty session')
        while True:
            row = self.journal.read_step(len(self.session.log), 'begin')
            if row is None:
                break
            message = Message.from_dict({**row['payload']['message'], 'seq': row['seq']})
            self.ingest(message, explicit_events=row['payload']['events'])
        self._load_boundary(len(self.session.log))
        return len(self.session.log)
