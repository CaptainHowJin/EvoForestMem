from __future__ import annotations
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import time
from ..forest.update import PeriodUpdater
from ..index.dense import DenseIndex
from ..index.sparse import SparseIndex
from ..log.message import Message
from ..log.store import MessageLog
from ..retrieve.pipeline import PipelineConfig

class SummaryRefreshInLoopError(RuntimeError):
    pass

class StreamingSession:


    def __init__(self, updater: PeriodUpdater, *, embedder: Any=None, dense: DenseIndex | None=None, summariser: Any=None, config: PipelineConfig | Mapping[str, Any] | None=None, analyser: Any=None, views: Sequence[str] | None=None, artifact_keys: Mapping[str, Sequence[str]] | None=None, refresh_bridges: bool=False, dense_vectors: Mapping[str, Any] | None=None, maintenance_shadow: bool=True, current_head_enabled: bool=True, render_names: tuple[Mapping[str, str], Mapping[str, str]] | None=None, retriever_mode: str='pipeline', corpus: str | None=None, closure_config: Any=None, evidence_planner: Any=None, count_tokens: Any=None, card_store: Any=None, card_client: Any=None, card_policy: str='fixed', card_snapshot_dir: Any=None, reuse_card_snapshots: bool=False, artifact_routing: str='auto', artifact_min_messages: int=2, routing_summary_client: Any=None, routing_summary_config: Any=None, routing_summary_cache: Any=None, summary_mode: str='budgeted', full_raw_search: bool=False) -> None:

        self.updater = updater
        if artifact_routing not in {'auto', 'tag', 'legacy'}:
            raise ValueError('artifact_routing must be auto, tag or legacy')
        self.artifact_routing = artifact_routing
        self._tag_router = None
        from ..routers.artifact import ArtifactRouter
        from ..routers.tag_artifact import TagArtifactRouter
        existing = [r for r in updater.builder.routers if r.view == 'artifact']
        empty_default = len(existing) == 1 and type(existing[0]) is ArtifactRouter and (not existing[0].keys)
        enable = artifact_routing == 'tag' or (artifact_routing == 'auto' and retriever_mode == 'forest_first' and empty_default and (artifact_keys is None) and (not len(updater.log)) and (views is None or 'artifact' in views) and (not hasattr(updater.builder, 'snapshot')) and (updater._fork_config is None))
        if enable:
            if views is not None and 'artifact' not in views:
                raise ValueError('tag routing conflicts with retrieval views excluding artifact')
            if len(updater.log) or artifact_keys is not None:
                raise ValueError('tag routing requires a fresh replay without frozen artifact bindings')
            if 'artifact' not in updater.builder.config.enabled_views:
                raise ValueError('tag routing requires the artifact view to be enabled')
            if existing and (not empty_default):
                raise ValueError('cannot replace a custom/static artifact router; use legacy or auto')
            if hasattr(updater.builder, 'snapshot') or updater._fork_config is not None:
                raise ValueError('tag routing currently requires ordinary trees without LSM or forks')
            self._tag_router = TagArtifactRouter(min_messages=artifact_min_messages)
            updater.builder.routers = tuple((r for r in updater.builder.routers if r.view != 'artifact')) + (self._tag_router,)
            updater.builder._per_view.setdefault('artifact', 0)
            updater.builder._unrouted.setdefault('artifact', 0)
            artifact_keys = self._tag_router.query_keys
        self.closure_config = closure_config
        self.evidence_planner = evidence_planner
        self.count_tokens = count_tokens
        self.card_client = card_client
        self.card_policy = card_policy
        self.card_snapshot_dir = card_snapshot_dir
        self.reuse_card_snapshots = reuse_card_snapshots
        self._card_snapshot_loads = 0
        if reuse_card_snapshots and card_snapshot_dir is None:
            raise ValueError('reusing cards requires a snapshot directory')
        self.card_store = card_store
        if maintenance_shadow:
            if self.updater.maintenance is None:
                from ..forest.maintenance import FmcConfig, ForestMaintenanceController
                self.updater.maintenance = ForestMaintenanceController(config=FmcConfig())
        else:
            self.updater.maintenance = None
        self.maintenance_shadow = maintenance_shadow
        self.current_head_enabled = current_head_enabled
        self.embedder = embedder
        if dense is not None and embedder is None:
            raise ValueError('a live dense channel needs the embedder that encodes it')
        self.dense = dense
        self.dense_vectors = dense_vectors
        self.summariser = summariser
        self.analyser = analyser
        self.views = views
        self.artifact_keys = artifact_keys
        self.refresh_bridges = refresh_bridges
        self._render_names = render_names
        self.retriever_mode = retriever_mode
        if config is None:
            from ..retrieve.pipeline import apply_corpus_profile, load_yaml
            config = PipelineConfig.from_dict(apply_corpus_profile(load_yaml(None), corpus))
        elif isinstance(config, Mapping):
            config = PipelineConfig.from_dict(config)
        self._base_config = config
        self.config = config
        self.routing_summaries = None
        self.full_raw_search = full_raw_search or summary_mode == 'hierarchical_full'
        if self.full_raw_search and retriever_mode != 'forest_first':
            raise ValueError('full raw search requires forest_first')
        if routing_summary_client is not None:
            if retriever_mode != 'forest_first' or summariser is not None:
                raise ValueError('budgeted routing summaries require forest_first without a recursive summariser')
            if len(updater.log) or hasattr(updater.builder, 'snapshot') or updater._fork_config is not None:
                raise ValueError('routing summaries require a fresh ordinary-tree replay')
            from ..forest.hierarchical_summaries import HierarchicalSummaryConfig, HierarchicalSummaryStore
            cfg_type, store_type = (HierarchicalSummaryConfig, HierarchicalSummaryStore)
            if isinstance(routing_summary_config, Mapping):
                routing_summary_config = cfg_type(**routing_summary_config)
            self.routing_summaries = store_type(client=routing_summary_client, count_tokens=count_tokens, config=routing_summary_config, cache_dir=routing_summary_cache, **{'embedder': embedder, 'vector_lookup': self.dense.vector})
        if self._tag_router is not None:
            if config.norm_spans > 0 and updater.norms is None:
                from ..forest.norms import NormIndex, NormConfig
                updater.norms = NormIndex(updater.log, config=NormConfig(max_results=config.norm_spans))
        self._snapshot_episodes = None
        self._snapshot_episode_epoch = -1
        self.sparse = SparseIndex()
        self._epoch = 0
        self._arrival_seconds: list[float] = []
        self._node_index: Any = None
        self._roots: Any = None
        self._bridges: Any = None
        self._dirty_node_trees: set[str] = set()
        self._dnm_delta_queue: set[tuple[str, str, str]] = set()
        self.updater.node_index_checkpoint = self.save_node_index_shards
        self.stream_controller = None

    @property
    def log(self) -> MessageLog:
        return self.updater.log

    def ingest(self, message: Message, *, explicit_events=(), message_lookup=None) -> Any:
        if self.stream_controller is not None:
            return self.stream_controller.ingest(message, explicit_events=explicit_events, message_lookup=message_lookup)
        return self._ingest(message, explicit_events=explicit_events, message_lookup=message_lookup)

    def _ingest(self, message: Message, *, explicit_events: Iterable[Mapping[str, Any]]=(), message_lookup: Mapping[str, Message] | None=None, stream_vector=None) -> Any:

        ingest_started = time.perf_counter()
        self._epoch += 1
        already = message.msg_id in self.updater.log
        explicit_events = tuple(explicit_events)
        report = self.updater.update(self._epoch, [message], explicit_events=explicit_events, message_lookup=message_lookup)
        if getattr(self.routing_summaries, 'mode', None) == 'hierarchical_full':
            self.routing_summaries.observe_revisions(explicit_events, self.log)
        if already or report.messages == 0:
            return report
        stored = self.log.get(message.msg_id)
        self.sparse.add(stored.msg_id, stored.text)
        if self.dense is not None:
            vec = stream_vector
            if vec is None and self.dense_vectors is not None:
                vec = self.dense_vectors.get(stored.msg_id)
            if vec is None:
                vec = self.embedder.encode([stored.text], kind='document')[0]
            self.dense.add(stored.msg_id, np.asarray(vec, dtype=np.float32))
        if self.routing_summaries is not None:
            self.routing_summaries.observe_arrival(stored.msg_id, self.updater.placement, self.updater.store.trees)
        return report

    def refresh_routing_summaries(self):

        if self.routing_summaries is None:
            return {'calls': 0, 'attempts': 0}
        if self.stream_controller is not None:
            return self.stream_controller.refresh_summaries()
        return self.routing_summaries.refresh(trees=self.updater.store.trees, log=self.log)

    def ingest_all(self, messages: Iterable[Message], *, message_lookup: Mapping[str, Message] | None=None) -> int:
        n = 0
        for message in messages:
            self.ingest(message, message_lookup=message_lookup)
            n += 1
        return n

    def retriever(self) -> Any:
        if self.stream_controller is not None:
            from ..structure_update.streaming import LockedRetriever
            return LockedRetriever(self.stream_controller)
        return self._retriever()

    def _retriever(self) -> Any:

        up = self.updater
        from dataclasses import replace as _dc_replace
        config = self._base_config
        if self._node_index is None and config.s2_source == 'dense':
            config = _dc_replace(config, s2_source='leaf')
        lsm_snapshot = None
        if hasattr(up.builder, 'snapshot'):
            lsm_snapshot = up.builder.snapshot()
        retrieval_trees = lsm_snapshot.trees if lsm_snapshot is not None else up.store.trees
        retrieval_placement = lsm_snapshot.placement if lsm_snapshot is not None else up.placement
        from ..retrieve.forest_first import ForestFirstRetriever
        statistics = None
        if config.retrieval_strategy == 'hierarchical_v1':
            from ..retrieve.sufficiency import SufficiencyStatistics
            if not hasattr(self, '_sufficiency_statistics'):
                self._sufficiency_statistics = SufficiencyStatistics()
            statistics = self._sufficiency_statistics.sync(up.log)
        episodes = up.episodes
        if self._tag_router is not None and config.duration_headers and (episodes is None):
            if self._snapshot_episode_epoch != self._epoch:
                from ..forest.episodes import EpisodeTable
                self._snapshot_episodes = EpisodeTable(up.log)
                self._snapshot_episode_epoch = self._epoch
            episodes = self._snapshot_episodes
        return ForestFirstRetriever(log=up.log, placement=retrieval_placement, sparse=self.sparse, dense=self.dense, embedder=self.embedder, trees=retrieval_trees, roots=self._roots, analyser=self.analyser, config=config, views=self.views, artifact_keys=self.artifact_keys, artifact_soft_keys=self._tag_router.soft_keys if self._tag_router else None, validity=up.ledger, norms=up.norms, episodes=episodes, render_names=self._render_names, routing_summaries=self.routing_summaries, full_raw_search=self.full_raw_search, association_graph=getattr(self, 'association_graph', None), sufficiency_statistics=statistics, navigation_centroids=self.stream_controller.statistics if self.stream_controller is not None else None)

    def answer(self, question: str) -> Any:

        return self.retriever().search(question)

    def save_node_index_shards(self, path: str) -> int:

        self._refresh_node_index()
        if self._node_index is None or not self._dirty_node_trees:
            return 0
        saved = self._node_index.save_shards(path, tree_ids=self._dirty_node_trees)
        dnm = self.updater.maintenance.dnm if self.updater.maintenance is not None else None
        if dnm is not None:
            for state in dnm.states.values():
                if state.tree_id in self._dirty_node_trees:
                    state.dirty_summary_row = False
        self._dirty_node_trees.clear()
        return saved

    def stats(self) -> dict[str, Any]:
        up = self.updater
        return {'epoch': self._epoch, 'retriever_mode': self.retriever_mode, 'version_authority': 'validity_ledger', 'artifact_routing_requested': self.artifact_routing, 'arrival_to_queryable_ms': {'p50': float(np.percentile(self._arrival_seconds, 50) * 1000) if self._arrival_seconds else None, 'p95': float(np.percentile(self._arrival_seconds, 95) * 1000) if self._arrival_seconds else None, 'total_sec': sum(self._arrival_seconds)}, 'card_snapshots_reused': self._card_snapshot_loads, 'routing_summaries': self.routing_summaries.report() if self.routing_summaries else None, 'artifact_routing': [r.stats() for r in up.builder.routers if r.view == 'artifact'], 'trees_by_view': {v: sum((t.view == v for t in up.store.trees.values())) for v in ('channel', 'actor', 'artifact')}, 'placements_by_view': dict(up.builder._per_view), 'messages': len(up.log), 'trees': len(up.store.trees), 'placements': up.placement.n_placements, 'claims': len(up.ledger), 'sparse_docs': len(self.sparse), 'dense_docs': len(self.dense) if self.dense is not None else 0, 'node_index': len(self._node_index) if self._node_index is not None else 0, 'roots': len(self._roots) if self._roots is not None else 0, 'norms': up.norms.stats.norms if up.norms is not None else 0, 'episodes': len(up.episodes.episodes) if up.episodes is not None else len(self._snapshot_episodes.episodes) if self._snapshot_episode_epoch == self._epoch else 0, 'episode_cache_epoch': self._snapshot_episode_epoch, 'episode_policy': 'caller_supplied' if up.episodes is not None else 'snapshot_batch', 'dnm_delta_queue': len(self._dnm_delta_queue)}
