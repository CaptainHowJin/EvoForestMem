from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemTreeNode:


    node_id: str
    tree_id: str
    level: int
    time_start: float
    time_end: float
    summary: str
    summary_dirty: bool
    child_ids: list[str]
    item_count: int
    parent_id: str | None = None
    prev_leaf_id: str | None = None
    next_leaf_id: str | None = None


    embedding: list[float] | None = None


    @property
    def ts_start(self) -> float:
        return self.time_start

    @property
    def ts_end(self) -> float:
        return self.time_end

    @property
    def is_leaf(self) -> bool:
        return self.level == 0

    @property
    def msg_ids(self) -> tuple[str, ...]:

        return tuple(self.child_ids) if self.level == 0 else ()

    @property
    def children(self) -> tuple[str, ...]:

        return () if self.level == 0 else tuple(self.child_ids)


@dataclass
class MemTree:


    tree_id: str
    view: str
    key: str
    label: str
    k: int
    root_node_id: str
    nodes: dict[str, MemTreeNode]
    payload_ids_ordered: list[str]
    leaf_by_payload: dict[str, str]
    centroid: list[float] | None
    version: int
    lifecycle_state: str
    deactivated: bool
    replacement_tree_ids: list[str]
    created_at: float
    updated_at: float


    summaries_built: bool = False


    next_index_by_level: dict[int, int] = field(default_factory=dict)
    dirty_l0_node_ids: set[str] = field(default_factory=set)
    dirty_internal_node_ids_by_level: dict[int, set[str]] = field(default_factory=dict)


    append_sealed: bool = False


    def parent_of(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        return node.parent_id if node is not None else None

    def path_to_root(self, node_id: str) -> list[str]:

        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = node_id
        while current is not None and current in self.nodes and current not in seen:
            seen.add(current)
            chain.append(current)
            current = self.nodes[current].parent_id
        chain.reverse()
        return chain

    def leaf_node_id(self, msg_id: str) -> str | None:

        return self.leaf_by_payload.get(msg_id)

    def leaf_positions(self) -> dict[str, int]:

        return {msg_id: i for i, msg_id in enumerate(self.payload_ids_ordered)}

    @property
    def leaf_count(self) -> int:
        return len(self.payload_ids_ordered)


@dataclass(frozen=True)
class TreeCard:


    tree_id: str
    view: str
    label: str
    time_start: float
    time_end: float
    root_summary: str
    item_count: int


@dataclass(frozen=True)
class SummaryRequest:


    request_id: str
    tree_id: str
    node_id: str
    view: str
    tree_label: str
    level: int
    time_start: float
    time_end: float
    input_text: str
    is_root: bool = False
    channel_thread: bool = False


@dataclass(frozen=True)
class SummaryResult:


    request_id: str
    tree_id: str
    node_id: str
    summary: str
    elapsed_sec: float
    error: str | None = None


@dataclass(frozen=True)
class TreeBuildResult:


    tree_id: str
    view: str
    label: str
    node_count: int
    leaf_count: int
    summary_calls: int
    elapsed_sec: float
    error: str | None = None
