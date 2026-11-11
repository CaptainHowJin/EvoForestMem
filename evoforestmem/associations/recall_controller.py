from copy import deepcopy
from dataclasses import asdict
import time

from .recall_graph import RecallGraph
from ..structure_update.contracts import fingerprint


class RecallController:
    def __init__(self, session, config, structure, utility_config):
        if utility_config is None or utility_config.mode == 'off' or session.dense is None:
            raise ValueError('associative recall requires online accounting and frozen document embeddings')
        self.session, self.config, self.structure = session, config, structure
        self.utility_config = utility_config
        self.graph = RecallGraph(config, session.dense.vector)
        self.graph.controller = self
        self.recognizer, self.pending, self.decisions, self.round = None, [], [], 0
        session.association_graph, session.association_controller = self.graph, self

    def maintain(self, traces, known_at):
        stream = self.session.stream_controller
        if stream is None:
            return self._maintain(traces, known_at)
        with stream.lock:
            stream.ensure_healthy()
            key = fingerprint([known_at, [t['identity'] for t in traces]])
            saved = stream.association_boundary(key)
            if saved is not None:
                self.restore(saved)
                return self.decisions[-1]
            result = self._maintain(traces, known_at)
            stream.publish_associations(key, self.state())
            return result

    def _maintain(self, traces, known_at):
        tick = time.perf_counter()
        if any(p['ask_time'] > known_at for p in self.pending):
            raise ValueError('cannot publish future question feedback')
        updated = 0
        for row in self.pending:
            updated += self.graph.feedback(row['question'], row['bindings'], row['versions'],
                                           row['outcomes'], known_at)
            if self.recognizer is not None and row['audited']:
                self.recognizer.audited += 1
                self.recognizer.successes += bool(row['changed'])
        self.pending.clear()
        self.round += 1
        row = dict(round=self.round, known_at=known_at, decision='UPDATE' if updated else 'KEEP',
            reason='batch_feedback' if updated else 'no_verified_exposures', updated=updated,
            active=sum(map(len, self.graph._adjacency.values())), candidates=[],
            seconds=time.perf_counter()-tick, policy=self.config.learning_policy)
        self.decisions.append(row)
        return row

    def state(self):
        return dict(schema='associative-recall-controller-v4', graph=self.graph.state(),
            utility_config=asdict(self.utility_config), pending=deepcopy(self.pending),
            decisions=deepcopy(self.decisions), round=self.round)

    def restore(self, value):
        if (value.get('schema') != 'associative-recall-controller-v4'
                or value['utility_config'] != asdict(self.utility_config)):
            raise ValueError('fresh associative recall controller state required')
        self.graph.restore(value['graph'])
        self.graph.controller = self
        self.pending, self.decisions, self.round = deepcopy(value['pending']), deepcopy(value['decisions']), value['round']
        self.session.association_graph = self.graph
        if self.session.stream_controller is not None:
            self.session.stream_controller._graph_bindings_dirty = True
