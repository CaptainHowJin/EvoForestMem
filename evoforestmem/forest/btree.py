from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

from .tree_types import (
    MemTree,
    MemTreeNode,
    SummaryRequest,
)


TextResolver = Callable[[str], str]


PlacementHook = Callable[[MemTree, str, str, int], None]


UnplacementHook = Callable[[MemTree, str, str], None]


_VALIDATE: bool = os.environ.get("EVOFORESTMEM_TREE_VALIDATE", "") == "1"


def validation_enabled() -> bool:
    return _VALIDATE


def set_validation(enabled: bool) -> bool:

    global _VALIDATE
    previous = _VALIDATE
    _VALIDATE = bool(enabled)
    return previous


@contextmanager
def validating(enabled: bool = True) -> Iterator[None]:
    previous = set_validation(enabled)
    try:
        yield
    finally:
        set_validation(previous)


def _maybe_validate(tree: MemTree) -> None:
    if _VALIDATE:
        validate_tree_structure(tree, strict=True)


FASTPATH_STATS: dict[str, int] = {"hit": 0, "walk": 0, "walk_steps": 0}


def reset_fastpath_stats() -> None:
    FASTPATH_STATS.update(hit=0, walk=0, walk_steps=0)


def _new_node_id(tree_id: str, level: int, index: int) -> str:
    return f"{tree_id}:L{level}:{index}"


def _reset_tree_to_empty_root(tree: MemTree) -> None:


    root_id = _new_node_id(tree.tree_id, 1, _next_node_index(tree, 1))
    now = time.time()
    tree.nodes = {
        root_id: MemTreeNode(
            node_id=root_id,
            tree_id=tree.tree_id,
            level=1,
            time_start=now,
            time_end=now,
            summary="",
            summary_dirty=True,
            child_ids=[],
            item_count=0,
            parent_id=None,
        )
    }
    tree.root_node_id = root_id
    tree.payload_ids_ordered = []
    tree.leaf_by_payload = {}
    clear_dirty_queues(tree)
    tree.dirty_internal_node_ids_by_level.setdefault(1, set()).add(root_id)


def _bucket_indices(n: int, k: int) -> list[list[int]]:

    if n == 0:
        return []
    effective_k = max(2, k)
    buckets: list[list[int]] = []
    i = 0
    remaining = n
    while remaining > 0:


        take = effective_k if remaining - effective_k != 1 else effective_k - 1
        take = min(take, remaining)
        buckets.append(list(range(i, i + take)))
        i += take
        remaining -= take
    return buckets


def _compute_effective_k(n_leaves: int, k: int) -> int:

    if n_leaves <= k:
        return max(2, n_leaves)
    return max(2, k)


def _next_node_index(tree: MemTree, level: int) -> int:

    counter = tree.next_index_by_level.get(level)
    if counter is None:


        prefix = f"{tree.tree_id}:L{level}:"
        max_index = -1
        for node_id in tree.nodes:
            if not node_id.startswith(prefix):
                continue
            try:
                max_index = max(max_index, int(node_id.rsplit(":", 1)[1]))
            except (TypeError, ValueError):
                continue
        counter = max_index + 1
    tree.next_index_by_level[level] = counter + 1
    return counter


def _mark_node_dirty(tree: MemTree, node_id: str) -> None:
    node = tree.nodes.get(node_id)
    if node is None:
        return
    if node.level == 0:


        tree.dirty_l0_node_ids.add(node.node_id)
        return
    node.summary_dirty = True
    _refresh_internal_node_from_children(tree, node)
    tree.dirty_internal_node_ids_by_level.setdefault(node.level, set()).add(node.node_id)


def _payload_sort_key(*, time_start: float, time_end: float, payload_id: str) -> tuple[float, float, str]:
    return (float(time_start), float(time_end), str(payload_id))


def _leaf_sort_key(node: MemTreeNode) -> tuple[float, float, str]:
    payload_id = node.child_ids[0] if node.child_ids else node.node_id
    return _payload_sort_key(time_start=node.time_start, time_end=node.time_end, payload_id=payload_id)


def _refresh_internal_node_from_children(tree: MemTree, node: MemTreeNode) -> None:
    child_nodes = [tree.nodes[cid] for cid in node.child_ids if cid in tree.nodes]
    if not child_nodes:
        node.item_count = 0
        return
    node.time_start = min(child.time_start for child in child_nodes)
    node.time_end = max(child.time_end for child in child_nodes)
    node.item_count = sum(child.item_count for child in child_nodes)


def _rightmost_node_id_at_level(tree: MemTree, target_level: int) -> str | None:
    node = tree.nodes.get(tree.root_node_id)
    if node is None or node.level < target_level:
        return None
    while node.level > target_level:
        if not node.child_ids:
            return None
        next_node = tree.nodes.get(node.child_ids[-1])
        if next_node is None:
            return None
        node = next_node
    return node.node_id


def _min_internal_fanout(tree: MemTree) -> int:
    return max(1, math.ceil(tree.k / 2))


def _leftmost_leaf_id(tree: MemTree) -> str | None:
    node = tree.nodes.get(tree.root_node_id)
    if node is None:
        return None
    while node.level > 0:
        if not node.child_ids:
            return None
        node = tree.nodes.get(node.child_ids[0])
        if node is None:
            return None
    return node.node_id


def _find_l0_node_by_payload(tree: MemTree, payload_id: str) -> str | None:

    node_id = tree.leaf_by_payload.get(payload_id)
    if node_id is not None and node_id in tree.nodes:
        return node_id
    for node_id, node in tree.nodes.items():
        if node.level == 0 and node.child_ids == [payload_id]:
            return node_id
    return None


def _find_insertion_neighbors(
    tree: MemTree,
    *,
    time_start: float,
    time_end: float,
    payload_id: str,
    arrival_order=None,
) -> tuple[str | None, str | None]:
    target_key = _payload_sort_key(
        time_start=time_start,
        time_end=time_end,
        payload_id=payload_id,
    )


    leaf_key = _leaf_sort_key
    if arrival_order is not None:
        target_key = (float(time_start), float(time_end), arrival_order[payload_id])
        leaf_key = lambda node: (float(node.time_start), float(node.time_end),
                                 arrival_order[node.child_ids[0]])


    if not tree.payload_ids_ordered:
        FASTPATH_STATS["hit"] += 1
        return None, None
    last_id = _rightmost_node_id_at_level(tree, 0)
    last = tree.nodes.get(last_id) if last_id is not None else None
    if last is not None and last.next_leaf_id is None and leaf_key(last) <= target_key:
        FASTPATH_STATS["hit"] += 1
        return last_id, None

    FASTPATH_STATS["walk"] += 1
    current_id = _leftmost_leaf_id(tree)
    previous_id: str | None = None
    while current_id is not None:
        FASTPATH_STATS["walk_steps"] += 1
        current = tree.nodes.get(current_id)
        if current is None:
            break
        if leaf_key(current) > target_key:
            return previous_id, current_id
        previous_id = current_id
        current_id = current.next_leaf_id
    return previous_id, None


def _insert_leaf_into_chain(
    tree: MemTree,
    leaf_node_id: str,
    *,
    after_leaf_id: str | None,
    before_leaf_id: str | None,
) -> None:
    leaf = tree.nodes[leaf_node_id]
    leaf.prev_leaf_id = after_leaf_id
    leaf.next_leaf_id = before_leaf_id
    if after_leaf_id and after_leaf_id in tree.nodes:
        tree.nodes[after_leaf_id].next_leaf_id = leaf_node_id
    if before_leaf_id and before_leaf_id in tree.nodes:
        tree.nodes[before_leaf_id].prev_leaf_id = leaf_node_id


def _split_internal_node_upwards(
    tree: MemTree, node_id: str, *, seal_right_edge: bool = False,
) -> None:
    current_id = node_id
    while True:
        current = tree.nodes.get(current_id)
        if current is None or current.level == 0 or len(current.child_ids) <= tree.k:
            return


        if tree.append_sealed and current.parent_id is not None:
            parent = tree.nodes.get(current.parent_id)
            if parent is not None:
                try:
                    current_index = parent.child_ids.index(current.node_id)
                except ValueError:
                    current_index = -1
                if 0 <= current_index < len(parent.child_ids) - 1:
                    right = tree.nodes.get(parent.child_ids[current_index + 1])
                    if (right is not None and right.level == current.level
                            and len(right.child_ids) < tree.k):
                        moved_id = current.child_ids.pop()
                        right.child_ids.insert(0, moved_id)
                        moved = tree.nodes.get(moved_id)
                        if moved is not None:
                            moved.parent_id = right.node_id
                        _refresh_internal_node_from_children(tree, current)
                        _refresh_internal_node_from_children(tree, right)
                        _refresh_internal_node_from_children(tree, parent)
                        _mark_node_dirty(tree, current.node_id)
                        _mark_node_dirty(tree, right.node_id)
                        _mark_node_dirty(tree, parent.node_id)
                        return


        split_at = tree.k if tree.append_sealed else len(current.child_ids) // 2
        right_child_ids = current.child_ids[split_at:]
        current.child_ids = current.child_ids[:split_at]
        _refresh_internal_node_from_children(tree, current)
        _mark_node_dirty(tree, current.node_id)

        sibling_id = _new_node_id(tree.tree_id, current.level, _next_node_index(tree, current.level))
        sibling = MemTreeNode(
            node_id=sibling_id,
            tree_id=tree.tree_id,
            level=current.level,
            time_start=current.time_start,
            time_end=current.time_end,
            summary="",
            summary_dirty=True,
            child_ids=list(right_child_ids),
            item_count=0,
            parent_id=current.parent_id,
        )
        for child_id in right_child_ids:
            child = tree.nodes.get(child_id)
            if child is not None:
                child.parent_id = sibling_id
        tree.nodes[sibling_id] = sibling
        _refresh_internal_node_from_children(tree, sibling)
        _mark_node_dirty(tree, sibling_id)

        if current.node_id == tree.root_node_id:
            new_root_id = _new_node_id(tree.tree_id, current.level + 1, _next_node_index(tree, current.level + 1))
            new_root = MemTreeNode(
                node_id=new_root_id,
                tree_id=tree.tree_id,
                level=current.level + 1,
                time_start=min(current.time_start, sibling.time_start),
                time_end=max(current.time_end, sibling.time_end),
                summary="",
                summary_dirty=True,
                child_ids=[current.node_id, sibling_id],
                item_count=current.item_count + sibling.item_count,
                parent_id=None,
            )
            current.parent_id = new_root_id
            sibling.parent_id = new_root_id
            tree.nodes[new_root_id] = new_root
            tree.root_node_id = new_root_id
            _mark_node_dirty(tree, new_root_id)
            return

        parent = tree.nodes.get(current.parent_id or "")
        if parent is None:
            return
        try:
            parent_index = parent.child_ids.index(current.node_id)
        except ValueError:
            parent_index = len(parent.child_ids) - 1
        parent.child_ids.insert(parent_index + 1, sibling_id)
        sibling.parent_id = parent.node_id
        _refresh_internal_node_from_children(tree, parent)
        _mark_node_dirty(tree, parent.node_id)
        current_id = parent.node_id


def _insert_child_into_parent(
    tree: MemTree,
    *,
    parent_id: str,
    child_id: str,
    child_index: int,
    seal_right_edge: bool = False,
) -> None:
    parent = tree.nodes.get(parent_id)
    child = tree.nodes.get(child_id)
    if parent is None or child is None:
        return
    clamped_index = max(0, min(child_index, len(parent.child_ids)))
    parent.child_ids.insert(clamped_index, child_id)
    child.parent_id = parent_id
    _refresh_internal_node_from_children(tree, parent)
    _mark_node_dirty(tree, parent_id)
    _split_internal_node_upwards(tree, parent_id, seal_right_edge=seal_right_edge)


def _attach_leaf_to_tree(
    tree: MemTree,
    leaf_node_id: str,
    *,
    after_leaf_id: str | None,
    before_leaf_id: str | None,
    seal_right_edge: bool = False,
) -> None:
    leaf = tree.nodes.get(leaf_node_id)
    root = tree.nodes.get(tree.root_node_id)
    if leaf is None:
        return

    if root is None:
        tree.root_node_id = leaf_node_id
        leaf.parent_id = None
        return

    if root.level > 0 and not root.child_ids and root.item_count == 0:
        del tree.nodes[root.node_id]
        tree.root_node_id = leaf_node_id
        leaf.parent_id = None
        return

    if root.level == 0:
        new_root_id = _new_node_id(tree.tree_id, 1, _next_node_index(tree, 1))
        left_id, right_id = (leaf_node_id, root.node_id) if before_leaf_id == root.node_id else (root.node_id, leaf_node_id)
        new_root = MemTreeNode(
            node_id=new_root_id,
            tree_id=tree.tree_id,
            level=1,
            time_start=min(tree.nodes[left_id].time_start, tree.nodes[right_id].time_start),
            time_end=max(tree.nodes[left_id].time_end, tree.nodes[right_id].time_end),
            summary="",
            summary_dirty=True,
            child_ids=[left_id, right_id],
            item_count=tree.nodes[left_id].item_count + tree.nodes[right_id].item_count,
            parent_id=None,
        )
        tree.nodes[left_id].parent_id = new_root_id
        tree.nodes[right_id].parent_id = new_root_id
        tree.nodes[new_root_id] = new_root
        tree.root_node_id = new_root_id
        _mark_node_dirty(tree, new_root_id)
        return

    anchor_leaf_id = after_leaf_id or before_leaf_id
    if anchor_leaf_id is None:
        parent_id = _rightmost_node_id_at_level(tree, 1)
        if parent_id is None:
            return
        _insert_child_into_parent(
            tree,
            parent_id=parent_id,
            child_id=leaf_node_id,
            child_index=len(tree.nodes[parent_id].child_ids),
            seal_right_edge=seal_right_edge,
        )
        return

    anchor_leaf = tree.nodes.get(anchor_leaf_id)
    if anchor_leaf is None or anchor_leaf.parent_id is None:
        return
    parent = tree.nodes.get(anchor_leaf.parent_id)
    if parent is None:
        return
    try:
        anchor_index = parent.child_ids.index(anchor_leaf_id)
    except ValueError:
        anchor_index = len(parent.child_ids)
    child_index = anchor_index + 1 if after_leaf_id is not None else anchor_index
    _insert_child_into_parent(
        tree,
        parent_id=parent.node_id,
        child_id=leaf_node_id,
        child_index=child_index,
        seal_right_edge=seal_right_edge,
    )


def _unlink_leaf_chain(tree: MemTree, leaf_node: MemTreeNode) -> None:
    if leaf_node.prev_leaf_id and leaf_node.prev_leaf_id in tree.nodes:
        tree.nodes[leaf_node.prev_leaf_id].next_leaf_id = leaf_node.next_leaf_id
    if leaf_node.next_leaf_id and leaf_node.next_leaf_id in tree.nodes:
        tree.nodes[leaf_node.next_leaf_id].prev_leaf_id = leaf_node.prev_leaf_id


def _parent_and_siblings(tree: MemTree, node_id: str) -> tuple[MemTreeNode | None, int, MemTreeNode | None, MemTreeNode | None]:
    node = tree.nodes.get(node_id)
    if node is None or node.parent_id is None:
        return None, -1, None, None
    parent = tree.nodes.get(node.parent_id)
    if parent is None:
        return None, -1, None, None
    try:
        index = parent.child_ids.index(node_id)
    except ValueError:
        return parent, -1, None, None
    left = tree.nodes.get(parent.child_ids[index - 1]) if index > 0 else None
    right = tree.nodes.get(parent.child_ids[index + 1]) if index + 1 < len(parent.child_ids) else None
    return parent, index, left, right


def _borrow_or_merge_underflow(tree: MemTree, node_id: str) -> None:
    node = tree.nodes.get(node_id)
    if node is None:
        return

    if node.node_id == tree.root_node_id:
        if node.level > 0 and len(node.child_ids) == 1:
            child_id = node.child_ids[0]
            child = tree.nodes.get(child_id)
            if child is not None:
                child.parent_id = None
                tree.root_node_id = child_id
            del tree.nodes[node.node_id]
            if child is not None:
                _mark_node_and_ancestors_dirty(tree, child_id)
        elif node.level > 0 and not node.child_ids:
            node.item_count = 0
            node.time_start = time.time()
            node.time_end = node.time_start
            _mark_node_and_ancestors_dirty(tree, node.node_id)
        else:
            _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    parent, index, left, right = _parent_and_siblings(tree, node_id)
    if parent is None or index < 0:
        return


    if not node.child_ids:
        parent.child_ids.remove(node.node_id)
        del tree.nodes[node.node_id]
        tree.dirty_internal_node_ids_by_level.get(node.level, set()).discard(node.node_id)
        _refresh_internal_node_from_children(tree, parent)
        _borrow_or_merge_underflow(tree, parent.node_id)
        return
    if len(node.child_ids) >= _min_internal_fanout(tree):
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    donor = left if left is not None and len(left.child_ids) > _min_internal_fanout(tree) else None
    borrow_from_left = donor is not None
    if donor is None and right is not None and len(right.child_ids) > _min_internal_fanout(tree):
        donor = right

    if donor is not None:
        if borrow_from_left:
            moved_child_id = donor.child_ids.pop()
            node.child_ids.insert(0, moved_child_id)
        else:
            moved_child_id = donor.child_ids.pop(0)
            node.child_ids.append(moved_child_id)
        moved_child = tree.nodes.get(moved_child_id)
        if moved_child is not None:
            moved_child.parent_id = node.node_id
        _refresh_internal_node_from_children(tree, donor)
        _refresh_internal_node_from_children(tree, node)
        _mark_node_and_ancestors_dirty(tree, donor.node_id)
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    merge_target = left if left is not None else right
    if merge_target is None:
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    if merge_target is left:
        merged_children = merge_target.child_ids + node.child_ids
        remove_id = node.node_id
        keep = merge_target
    else:
        merged_children = node.child_ids + merge_target.child_ids
        remove_id = merge_target.node_id
        keep = node

    if len(merged_children) > tree.k:
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    keep.child_ids = merged_children
    for child_id in merged_children:
        child = tree.nodes.get(child_id)
        if child is not None:
            child.parent_id = keep.node_id
    _refresh_internal_node_from_children(tree, keep)
    parent.child_ids.remove(remove_id)
    del tree.nodes[remove_id]
    _mark_node_and_ancestors_dirty(tree, keep.node_id)
    if parent.node_id in tree.nodes:
        _refresh_internal_node_from_children(tree, parent)
        _borrow_or_merge_underflow(tree, parent.node_id)


def delete_payload(
    tree: MemTree,
    msg_id: str,
    *,
    on_unplace: UnplacementHook | None = None,
) -> bool:

    if tree.append_sealed:
        raise ValueError(
            "delete_payload is incompatible with append_sealed=True; "
            "rebuild with the default B-tree profile before deleting"
        )
    leaf_id = _find_l0_node_by_payload(tree, msg_id)
    if leaf_id is None:
        return False
    leaf = tree.nodes.get(leaf_id)
    if leaf is None:
        return False
    _unlink_leaf_chain(tree, leaf)
    if msg_id in tree.payload_ids_ordered:
        tree.payload_ids_ordered.remove(msg_id)
    tree.leaf_by_payload.pop(msg_id, None)
    parent_id = leaf.parent_id
    del tree.nodes[leaf_id]
    tree.dirty_l0_node_ids.discard(leaf_id)
    if parent_id is not None and parent_id in tree.nodes:
        parent = tree.nodes[parent_id]
        if leaf_id in parent.child_ids:
            parent.child_ids.remove(leaf_id)
        _refresh_internal_node_from_children(tree, parent)
        _borrow_or_merge_underflow(tree, parent_id)
    elif tree.root_node_id == leaf_id:
        if tree.nodes:
            leftmost = _leftmost_leaf_id(tree)
            if leftmost is not None:
                tree.root_node_id = leftmost
                tree.nodes[leftmost].parent_id = None
        else:
            _reset_tree_to_empty_root(tree)
    tree.version += 1
    tree.updated_at = time.time()
    if on_unplace is not None:
        on_unplace(tree, msg_id, leaf_id)
    _maybe_validate(tree)
    return True


def _mark_node_and_ancestors_dirty(tree: MemTree, node_id: str) -> None:
    current_id: str | None = node_id
    visited: set[str] = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        current = tree.nodes.get(current_id)
        if current is None:
            break
        _mark_node_dirty(tree, current_id)
        current_id = current.parent_id


def clear_dirty_queues(tree: MemTree) -> None:
    tree.dirty_l0_node_ids.clear()
    tree.dirty_internal_node_ids_by_level.clear()


def rebuild_dirty_queues_from_flags(tree: MemTree) -> None:

    clear_dirty_queues(tree)
    for node in tree.nodes.values():
        if node.summary_dirty and node.level > 0:
            tree.dirty_internal_node_ids_by_level.setdefault(node.level, set()).add(node.node_id)


def _build_single_leaf_hierarchy(
    *,
    tree_id: str,
    view: str,
    tree_label: str,
    level0_ids: list[str],
    nodes: dict[str, MemTreeNode],
    effective_k: int,
    level1_groups: Sequence[Sequence[int]] | None = None,
) -> dict[int, list[SummaryRequest]]:
    requests_by_level: dict[int, list[SummaryRequest]] = {0: []}
    current_level_ids = list(level0_ids)
    level = 1
    while len(current_level_ids) > 1:
        next_level_ids: list[str] = []
        req_this_level: list[SummaryRequest] = []
        groups = (
            [list(group) for group in level1_groups]
            if level == 1 and level1_groups is not None
            else _bucket_indices(len(current_level_ids), effective_k)
        )
        if level == 1 and level1_groups is not None:
            flattened = sorted(index for group in groups for index in group)
            if flattened != list(range(len(current_level_ids))):
                raise ValueError("level1_groups must partition every L0 index exactly once")
            if any(not group or len(group) > effective_k for group in groups):
                raise ValueError("level1_groups contains an empty or over-fanout group")
        for group_num, indices in enumerate(groups):
            child_ids = [current_level_ids[i] for i in indices]
            ts = min(nodes[cid].time_start for cid in child_ids)
            te = max(nodes[cid].time_end for cid in child_ids)
            item_count = sum(nodes[cid].item_count for cid in child_ids)
            node_id = _new_node_id(tree_id, level, group_num)
            node = MemTreeNode(
                node_id=node_id,
                tree_id=tree_id,
                level=level,
                time_start=ts,
                time_end=te,
                summary="",
                summary_dirty=True,
                child_ids=child_ids,
                item_count=item_count,
                parent_id=None,
            )
            for child_id in child_ids:
                nodes[child_id].parent_id = node_id
            nodes[node_id] = node
            next_level_ids.append(node_id)
            req_this_level.append(SummaryRequest(
                request_id=f"{tree_id}:{node_id}",
                tree_id=tree_id,
                node_id=node_id,
                view=view,
                tree_label=tree_label,
                level=level,
                time_start=ts,
                time_end=te,
                input_text="",
                channel_thread=view == "channel" and level1_groups is not None,
            ))
        requests_by_level[level] = req_this_level
        current_level_ids = next_level_ids
        level += 1
    return requests_by_level


def rebuild_hierarchy_from_level0(
    tree: MemTree,
    *,
    level1_groups: Sequence[Sequence[int]],
) -> tuple[MemTree, dict[int, list[SummaryRequest]]]:

    import copy

    rebuilt = copy.deepcopy(tree)
    level0_ids = [tree.leaf_by_payload[msg_id] for msg_id in tree.payload_ids_ordered]
    rebuilt.nodes = {node_id: copy.deepcopy(tree.nodes[node_id]) for node_id in level0_ids}
    for node in rebuilt.nodes.values():
        node.parent_id = None
        node.summary = ""
        node.summary_dirty = False

    requests = _build_single_leaf_hierarchy(
        tree_id=rebuilt.tree_id,
        view=rebuilt.view,
        tree_label=rebuilt.label,
        level0_ids=level0_ids,
        nodes=rebuilt.nodes,
        effective_k=max(2, rebuilt.k),
        level1_groups=level1_groups,
    )
    requests.pop(0, None)
    if len(level0_ids) == 1:
        rebuilt.root_node_id = level0_ids[0]
    else:
        root_level = max(node.level for node in rebuilt.nodes.values())
        roots = [node.node_id for node in rebuilt.nodes.values()
                 if node.level == root_level and node.parent_id is None]
        if len(roots) != 1:
            raise ValueError(f"rebuilt hierarchy has {len(roots)} roots")
        rebuilt.root_node_id = roots[0]

    rebuilt.next_index_by_level = {}
    for node in rebuilt.nodes.values():
        rebuilt.next_index_by_level[node.level] = max(
            rebuilt.next_index_by_level.get(node.level, 0),
            int(node.node_id.rsplit(":", 1)[1]) + 1,
        )
    rebuilt.summaries_built = False
    rebuilt.updated_at = time.time()
    rebuilt.version += 1
    rebuilt.dirty_l0_node_ids = set(level0_ids)
    rebuilt.dirty_internal_node_ids_by_level = {
        level: {request.node_id for request in level_requests}
        for level, level_requests in requests.items()
        if level_requests
    }
    validate_tree_structure(rebuilt)
    return rebuilt, requests


def validate_tree_structure(tree: MemTree, *, strict: bool = False) -> None:

    if tree.root_node_id not in tree.nodes:
        raise ValueError(f"tree {tree.tree_id} has missing root_node_id={tree.root_node_id}")
    root = tree.nodes[tree.root_node_id]
    if root.parent_id is not None:
        raise ValueError(f"tree {tree.tree_id} root {tree.root_node_id} must have parent_id=None")
    l0_nodes: list[MemTreeNode] = []
    metadata_errors: list[str] = []
    for node_id, node in tree.nodes.items():
        if node.level == 0:
            l0_nodes.append(node)
            if len(node.child_ids) != 1:
                raise ValueError(
                    f"tree {tree.tree_id} L0 node {node_id} must have exactly one payload child, "
                    f"got {len(node.child_ids)}"
                )
            payload_id = node.child_ids[0]
            if tree.leaf_by_payload.get(payload_id) != node_id:
                raise ValueError(
                    f"tree {tree.tree_id} leaf index disagrees for msg_id={payload_id}: "
                    f"indexed at {tree.leaf_by_payload.get(payload_id)}, node is {node_id}"
                )
            if node.item_count != 1:
                raise ValueError(
                    f"tree {tree.tree_id} L0 node {node_id} item_count must be 1, got {node.item_count}"
                )
            if node.parent_id is None and node.node_id != tree.root_node_id:
                raise ValueError(
                    f"tree {tree.tree_id} non-root L0 node {node_id} has no parent_id"
                )
            continue
        if not node.child_ids:
            if node_id == tree.root_node_id and node.item_count == 0:
                continue
            raise ValueError(f"tree {tree.tree_id} internal node {node_id} has no children")
        if len(node.child_ids) > tree.k:
            raise ValueError(
                f"tree {tree.tree_id} internal node {node_id} fanout {len(node.child_ids)} exceeds k={tree.k}"
            )
        if (strict and node_id != tree.root_node_id
                and len(node.child_ids) < _min_internal_fanout(tree)
                and not tree.append_sealed):
            raise ValueError(
                f"tree {tree.tree_id} internal node {node_id} fanout {len(node.child_ids)} underflows "
                f"min={_min_internal_fanout(tree)} for k={tree.k}"
            )
        child_nodes = [tree.nodes[child_id] for child_id in node.child_ids
                       if child_id in tree.nodes]
        if len(child_nodes) == len(node.child_ids):
            expected_count = sum(child.item_count for child in child_nodes)
            expected_start = min(child.time_start for child in child_nodes)
            expected_end = max(child.time_end for child in child_nodes)
            if node.item_count != expected_count:
                metadata_errors.append(
                    f"tree {tree.tree_id} internal node {node_id} item_count "
                    f"{node.item_count} != child total {expected_count}")
            if node.time_start != expected_start or node.time_end != expected_end:
                metadata_errors.append(
                    f"tree {tree.tree_id} internal node {node_id} time bounds disagree")
            if strict:
                for left, right in zip(child_nodes, child_nodes[1:]):
                    if left.time_end > right.time_start:
                        metadata_errors.append(
                            f"tree {tree.tree_id} internal node {node_id} children out of order")
        for child_id in node.child_ids:
            child = tree.nodes.get(child_id)
            if child is None:
                raise ValueError(
                    f"tree {tree.tree_id} internal node {node_id} references non-node child_id={child_id}"
                )
            if child.level != node.level - 1:
                raise ValueError(
                    f"tree {tree.tree_id} edge {node_id}->{child_id} crosses invalid levels "
                    f"{node.level}->{child.level}"
                )
            if child.parent_id != node_id:
                raise ValueError(
                    f"tree {tree.tree_id} child {child_id} has parent_id={child.parent_id}, "
                    f"expected {node_id}"
                )


    reachable: set[str] = set()
    leaf_depths: set[int] = set()
    stack: list[tuple[str, int]] = [(tree.root_node_id, 0)]
    while stack:
        node_id, depth = stack.pop()
        if node_id in reachable:
            raise ValueError(f"tree {tree.tree_id} node {node_id} reached twice (cycle or shared child)")
        reachable.add(node_id)
        node = tree.nodes.get(node_id)
        if node is None:
            raise ValueError(f"tree {tree.tree_id} walk reached missing node {node_id}")
        if node.level == 0:
            leaf_depths.add(depth)
            continue
        stack.extend((cid, depth + 1) for cid in node.child_ids)
    if len(reachable) != len(tree.nodes):
        orphans = set(tree.nodes) - reachable
        raise ValueError(
            f"tree {tree.tree_id} has {len(orphans)} node(s) unreachable from the root: "
            f"{sorted(orphans)[:5]}"
        )
    if len(leaf_depths) > 1:
        raise ValueError(
            f"tree {tree.tree_id} leaves sit at differing depths {sorted(leaf_depths)}"
        )
    if metadata_errors:
        raise ValueError(metadata_errors[0])


    l0_by_id = {n.node_id: n for n in l0_nodes}
    leftmost = next((n for n in l0_nodes if n.prev_leaf_id is None), None)
    if l0_nodes and leftmost is None:
        raise ValueError(
            f"tree {tree.tree_id} L0 chain has no node with prev_leaf_id=None"
        )
    visited_chain: list[MemTreeNode] = []
    current: MemTreeNode | None = leftmost
    visited_ids: set[str] = set()
    while current is not None:
        if current.node_id in visited_ids:
            raise ValueError(
                f"tree {tree.tree_id} L0 chain cycle detected at {current.node_id}"
            )
        visited_ids.add(current.node_id)

        if visited_chain:
            prev_node = visited_chain[-1]
            if current.prev_leaf_id != prev_node.node_id:
                raise ValueError(
                    f"tree {tree.tree_id} L0 chain back-pointer broken at {current.node_id}: "
                    f"prev={current.prev_leaf_id}, expected={prev_node.node_id}"
                )
            if prev_node.next_leaf_id != current.node_id:
                raise ValueError(
                    f"tree {tree.tree_id} L0 chain forward-pointer broken at {prev_node.node_id}: "
                    f"next={prev_node.next_leaf_id}, expected={current.node_id}"
                )
        visited_chain.append(current)
        nxt_id = current.next_leaf_id
        current = l0_by_id.get(nxt_id) if nxt_id else None
        if nxt_id and nxt_id not in l0_by_id:
            raise ValueError(
                f"tree {tree.tree_id} L0 chain points to non-L0/missing node {nxt_id}"
            )
    if len(visited_chain) != len(l0_nodes):
        unreachable = set(l0_by_id.keys()) - visited_ids
        raise ValueError(
            f"tree {tree.tree_id} L0 chain covers {len(visited_chain)}/{len(l0_nodes)} nodes; "
            f"unreachable: {sorted(unreachable)[:5]}"
        )


    for i in range(1, len(visited_chain)):
        prev_times = (visited_chain[i - 1].time_start, visited_chain[i - 1].time_end)
        curr_times = (visited_chain[i].time_start, visited_chain[i].time_end)
        if curr_times < prev_times:
            raise ValueError(
                f"tree {tree.tree_id} L0 chain ordering violation between "
                f"{visited_chain[i-1].node_id} and {visited_chain[i].node_id}: "
                f"{prev_times} > {curr_times}"
            )


    chain_payloads = [n.child_ids[0] for n in visited_chain]
    if chain_payloads != list(tree.payload_ids_ordered):
        raise ValueError(
            f"tree {tree.tree_id} payload_ids_ordered disagrees with the leaf chain "
            f"({len(tree.payload_ids_ordered)} recorded vs {len(chain_payloads)} chained)"
        )
    if len(tree.leaf_by_payload) != len(l0_nodes):
        raise ValueError(
            f"tree {tree.tree_id} leaf index holds {len(tree.leaf_by_payload)} entries "
            f"for {len(l0_nodes)} L0 nodes"
        )


def build_tree_from_payloads(
    tree_id: str,
    view: str,
    key: str,
    label: str,
    k: int,
    payloads: Sequence[tuple[str, float, float]],
    *,
    level1_groups: Sequence[Sequence[int]] | None = None,
    append_sealed: bool = False,
) -> tuple[MemTree, dict[int, list[SummaryRequest]]]:

    now = time.time()
    if not payloads:

        root_id = _new_node_id(tree_id, 1, 0)
        root_node = MemTreeNode(
            node_id=root_id,
            tree_id=tree_id,
            level=1,
            time_start=now,
            time_end=now,
            summary="",
            summary_dirty=True,
            child_ids=[],
            item_count=0,
        )
        tree = MemTree(
            tree_id=tree_id,
            view=view,
            key=key,
            label=label,
            k=k,
            root_node_id=root_id,
            nodes={root_id: root_node},
            payload_ids_ordered=[],
            leaf_by_payload={},
            centroid=None,
            version=1,
            lifecycle_state="active",
            deactivated=False,
            replacement_tree_ids=[],
            created_at=now,
            updated_at=now,
            next_index_by_level={1: 1},
            append_sealed=append_sealed,
        )
        return tree, {}

    effective_k = _compute_effective_k(len(payloads), k)
    nodes: dict[str, MemTreeNode] = {}


    ordered = sorted(
        payloads,
        key=lambda p: _payload_sort_key(time_start=p[1], time_end=p[2], payload_id=p[0]),
    )

    payload_ids = [p[0] for p in ordered]
    leaf_by_payload: dict[str, str] = {}
    level0_ids: list[str] = []
    for leaf_num, (payload_id, ts, te) in enumerate(ordered):
        node_id = _new_node_id(tree_id, 0, leaf_num)
        prev_leaf_id = _new_node_id(tree_id, 0, leaf_num - 1) if leaf_num > 0 else None
        next_leaf_id = _new_node_id(tree_id, 0, leaf_num + 1) if leaf_num + 1 < len(ordered) else None
        nodes[node_id] = MemTreeNode(
            node_id=node_id,
            tree_id=tree_id,
            level=0,
            time_start=float(ts),
            time_end=float(te),
            summary="",
            summary_dirty=False,
            child_ids=[payload_id],
            item_count=1,
            parent_id=None,
            prev_leaf_id=prev_leaf_id,
            next_leaf_id=next_leaf_id,
        )
        level0_ids.append(node_id)
        leaf_by_payload[payload_id] = node_id

    requests_by_level = _build_single_leaf_hierarchy(
        tree_id=tree_id,
        view=view,
        tree_label=label,
        level0_ids=level0_ids,
        nodes=nodes,
        effective_k=effective_k,
        level1_groups=level1_groups,
    )
    requests_by_level.pop(0, None)
    root_node_id = level0_ids[0]
    if len(level0_ids) > 1:
        root_level = max(node.level for node in nodes.values())
        root_node_id = _new_node_id(tree_id, root_level, 0)

    next_index_by_level: dict[int, int] = {}
    for node in nodes.values():
        next_index_by_level[node.level] = max(
            next_index_by_level.get(node.level, 0),
            int(node.node_id.rsplit(":", 1)[1]) + 1,
        )

    tree = MemTree(
        tree_id=tree_id,
        view=view,
        key=key,
        label=label,


        k=max(2, k),
        root_node_id=root_node_id,
        nodes=nodes,
        payload_ids_ordered=payload_ids,
        leaf_by_payload=leaf_by_payload,
        centroid=None,
        version=1,
        lifecycle_state="active",
        deactivated=False,
        replacement_tree_ids=[],
        created_at=now,
        updated_at=now,
        next_index_by_level=next_index_by_level,
        dirty_l0_node_ids=set(level0_ids),
        dirty_internal_node_ids_by_level={
            level: {req.node_id for req in reqs}
            for level, reqs in requests_by_level.items()
            if level > 0 and reqs
        },
        append_sealed=append_sealed,
    )
    _maybe_validate(tree)
    return tree, requests_by_level


def empty_tree(
    tree_id: str, view: str, key: str, label: str, k: int, *, append_sealed: bool = False,
) -> MemTree:

    return build_tree_from_payloads(
        tree_id, view, key, label, k, (), append_sealed=append_sealed,
    )[0]


def insert_payload(
    tree: MemTree,
    msg_id: str,
    ts_start: float,
    ts_end: float | None = None,
    *,
    on_place: PlacementHook | None = None,
    arrival_order=None,
) -> str:

    existing = tree.leaf_by_payload.get(msg_id)
    if existing is not None and existing in tree.nodes:
        return existing

    ts = float(ts_start)
    te = float(ts_end if ts_end is not None else ts_start)
    after_leaf_id, before_leaf_id = _find_insertion_neighbors(
        tree,
        time_start=ts,
        time_end=te,
        payload_id=msg_id,
        arrival_order=arrival_order,
    )
    leaf_id = _new_node_id(tree.tree_id, 0, _next_node_index(tree, 0))
    tree.nodes[leaf_id] = MemTreeNode(
        node_id=leaf_id,
        tree_id=tree.tree_id,
        level=0,
        time_start=ts,
        time_end=te,
        summary="",
        summary_dirty=False,
        child_ids=[msg_id],
        item_count=1,
        parent_id=None,
        prev_leaf_id=None,
        next_leaf_id=None,
    )
    _insert_leaf_into_chain(
        tree,
        leaf_id,
        after_leaf_id=after_leaf_id,
        before_leaf_id=before_leaf_id,
    )
    _attach_leaf_to_tree(
        tree,
        leaf_id,
        after_leaf_id=after_leaf_id,
        before_leaf_id=before_leaf_id,
        seal_right_edge=tree.append_sealed and before_leaf_id is None,
    )


    if before_leaf_id is None:
        insert_at = len(tree.payload_ids_ordered)
    else:
        anchor_msg_id = tree.nodes[before_leaf_id].child_ids[0]
        try:
            insert_at = tree.payload_ids_ordered.index(anchor_msg_id)
        except ValueError:
            insert_at = 0
    tree.payload_ids_ordered.insert(insert_at, msg_id)
    tree.leaf_by_payload[msg_id] = leaf_id
    _mark_node_and_ancestors_dirty(tree, leaf_id)
    tree.version += 1
    tree.updated_at = time.time()
    if on_place is not None:
        on_place(tree, msg_id, leaf_id, insert_at)
    _maybe_validate(tree)
    return leaf_id


def collect_dirty_requests(
    tree: MemTree,
    text_of: TextResolver | None = None,
) -> dict[int, list[SummaryRequest]]:

    requests_by_level: dict[int, list[SummaryRequest]] = {}

    dirty_node_ids: list[str] = []
    for level in sorted(tree.dirty_internal_node_ids_by_level):
        dirty_node_ids.extend(
            node_id
            for node_id in sorted(tree.dirty_internal_node_ids_by_level.get(level, set()))
            if node_id in tree.nodes
        )

    if not dirty_node_ids:
        dirty_node_ids = [
            node.node_id
            for node in sorted(tree.nodes.values(), key=lambda n: (n.level, n.node_id))
            if node.summary_dirty and node.level > 0
        ]

    channel_thread = tree.view == "channel" and _has_noncontiguous_level1(tree)
    for node_id in dirty_node_ids:
        node = tree.nodes.get(node_id)
        if node is None or node.level == 0:
            continue
        if not node.summary_dirty:
            continue
        if not node.child_ids:
            node.summary = ""
            node.summary_dirty = False
            tree.dirty_internal_node_ids_by_level.get(node.level, set()).discard(node.node_id)
            continue

        input_text = _build_level0_input(tree, node, text_of) if node.level == 1 else ""

        req = SummaryRequest(
            request_id=f"{tree.tree_id}:{node.node_id}",
            tree_id=tree.tree_id,
            node_id=node.node_id,
            view=tree.view,
            tree_label=tree.label,
            level=node.level,
            time_start=node.time_start,
            time_end=node.time_end,
            input_text=input_text,
            is_root=node.node_id == tree.root_node_id,
            channel_thread=channel_thread,
        )
        requests_by_level.setdefault(node.level, []).append(req)

    return requests_by_level


def _has_noncontiguous_level1(tree: MemTree) -> bool:

    positions = {
        tree.leaf_by_payload[payload_id]: index
        for index, payload_id in enumerate(tree.payload_ids_ordered)
    }
    for node in tree.nodes.values():
        if node.level != 1 or len(node.child_ids) < 2:
            continue
        indexes = [positions[child_id] for child_id in node.child_ids if child_id in positions]
        if indexes and indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            return True
    return False


def _build_level0_input(
    tree: MemTree,
    node: MemTreeNode,
    text_of: TextResolver | None,
) -> str:

    if text_of is None:
        return ""
    parts: list[str] = []
    for child_id in node.child_ids:
        child = tree.nodes.get(child_id)
        if child is None or child.level != 0 or not child.child_ids:
            continue
        parts.append(text_of(child.child_ids[0]))
    return "\n\n".join(p for p in parts if p)


def fill_upper_level_inputs(
    tree: MemTree,
    requests_by_level: dict[int, list[SummaryRequest]],
    results_by_node: dict[str, str],
    text_of: TextResolver | None = None,
) -> dict[int, list[SummaryRequest]]:

    updated: dict[int, list[SummaryRequest]] = {}
    for level, reqs in requests_by_level.items():
        if level == 0:
            updated[level] = reqs
            continue
        new_reqs: list[SummaryRequest] = []
        for req in reqs:
            node = tree.nodes.get(req.node_id)
            if node is None:
                new_reqs.append(req)
                continue
            if node.level == 1:
                input_text = _build_level0_input(tree, node, text_of)
            else:
                child_summaries = [
                    results_by_node.get(cid, tree.nodes[cid].summary if cid in tree.nodes else "")
                    for cid in node.child_ids
                ]
                input_text = "\n\n".join(s for s in child_summaries if s)
            new_reqs.append(SummaryRequest(
                request_id=req.request_id,
                tree_id=req.tree_id,
                node_id=req.node_id,
                view=req.view,
                tree_label=req.tree_label,
                level=req.level,
                time_start=req.time_start,
                time_end=req.time_end,
                input_text=input_text,
                is_root=req.is_root,
                channel_thread=req.channel_thread,
            ))
        updated[level] = new_reqs
    return updated
