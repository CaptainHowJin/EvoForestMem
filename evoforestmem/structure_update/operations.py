from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from ..forest import btree
from ..forest.tree_types import MemTreeNode
from .contracts import StructureDelta, fingerprint


@dataclass(frozen=True)
class LocalOperation:
    kind: str
    tree_id: str
    node_id: str
    sibling_id: str = ""
    cut: int = 0

    @property
    def identity(self):
        return fingerprint(asdict(self))


def leaf_order(tree):
    out, stack = [], [tree.root_node_id]
    while stack:
        node = tree.nodes[stack.pop()]
        if node.level == 0:
            out.extend(node.child_ids)
        else:
            stack.extend(reversed(node.child_ids))
    return out


def validate(tree):
    btree.validate_tree_structure(tree)
    if leaf_order(tree) != tree.payload_ids_ordered:
        raise ValueError("hierarchy order disagrees with stable leaf order")


def preview(tree, op, *, local=False):
    if op.tree_id != tree.tree_id or op.node_id not in tree.nodes:
        raise ValueError("local operation refers to a missing node")
    if local:
        from .streaming_local import clone_tree
        candidate = clone_tree(tree)
    else:
        candidate = deepcopy(tree)
    node = candidate.nodes[op.node_id]
    if node.level == 0:
        raise ValueError("optional operations cannot alter raw leaves")
    changed = set()
    if op.kind == "split":
        if not 2 <= op.cut <= len(node.child_ids)-2:
            raise ValueError("early split must retain at least two children per side")
        right = _sibling(candidate, node, node.child_ids[op.cut:])
        node.child_ids = node.child_ids[:op.cut]
        if node.parent_id is None:
            root = _sibling(candidate, node, [node.node_id, right.node_id], level=node.level+1)
            root.parent_id = None
            candidate.root_node_id = root.node_id
            node.parent_id = right.parent_id = root.node_id
            changed.add(root.node_id)
        else:
            parent = candidate.nodes[node.parent_id]
            parent.child_ids.insert(parent.child_ids.index(node.node_id)+1, right.node_id)
            changed.add(parent.node_id)
        changed.update((node.node_id, right.node_id))
    elif op.kind in {"boundary", "merge"}:
        right = candidate.nodes.get(op.sibling_id)
        parent = candidate.nodes.get(node.parent_id)
        if (right is None or parent is None or right.parent_id != node.parent_id or
                right.level != node.level or
                parent.child_ids.index(right.node_id) != parent.child_ids.index(node.node_id)+1):
            raise ValueError("operation requires adjacent same-level siblings")
        children = node.child_ids + right.child_ids
        if op.kind == "merge":
            if len(children) > tree.k:
                raise ValueError("merge overflows fan-out")
            node.child_ids = children
            parent.child_ids.remove(right.node_id)
            del candidate.nodes[right.node_id]
        else:
            if not max(2, len(children)-tree.k) <= op.cut <= min(tree.k, len(children)-2):
                raise ValueError("illegal boundary position")
            if op.cut == len(node.child_ids):
                raise ValueError("boundary operation must move children")
            node.child_ids, right.child_ids = children[:op.cut], children[op.cut:]
            changed.add(right.node_id)
        changed.update((node.node_id, parent.node_id))
    else:
        raise ValueError("unknown local operation")

    for nid in sorted(changed, key=lambda n: candidate.nodes[n].level):
        n = candidate.nodes[nid]
        for cid in n.child_ids:
            candidate.nodes[cid].parent_id = nid
        btree._refresh_internal_node_from_children(candidate, n)
        btree._mark_node_and_ancestors_dirty(candidate, nid)
    for nid in (sorted(changed, key=lambda n: (candidate.nodes[n].level, n)) if local else list(changed)):
        btree._split_internal_node_upwards(candidate, nid)
    root = candidate.nodes[candidate.root_node_id]
    while root.level > 1 and len(root.child_ids) == 1:
        candidate.root_node_id = root.child_ids[0]
        del candidate.nodes[root.node_id]
        root = candidate.nodes[candidate.root_node_id]
        root.parent_id = None

    inspected = (list(candidate.nodes.changed.items()) if local else candidate.nodes.items())
    affected = {nid for nid, n in inspected
                if nid not in tree.nodes or n.child_ids != tree.nodes[nid].child_ids}
    for nid in list(affected):
        affected.update(candidate.path_to_root(nid))
    for nid in sorted(affected, key=lambda n: candidate.nodes[n].level):
        n = candidate.nodes[nid]
        if n.level:
            btree._refresh_internal_node_from_children(candidate, n)
            btree._mark_node_dirty(candidate, nid)
    candidate.version = tree.version + 1
    candidate.centroid = None
    if not local:
        candidate.dirty_l0_node_ids.intersection_update(candidate.nodes)
        for ids in candidate.dirty_internal_node_ids_by_level.values():
            ids.intersection_update(candidate.nodes)

    if local:
        from .streaming_local import validate_local
        validate_local(tree, candidate)
    else:
        validate(candidate)
    if not local and (candidate.payload_ids_ordered != tree.payload_ids_ordered or candidate.leaf_by_payload != tree.leaf_by_payload):
        raise AssertionError("optional operation changed view membership or raw identity")
    inspected = (list(candidate.nodes.changed.items()) if local else candidate.nodes.items())
    modified = affected | {nid for nid,n in inspected
                           if nid not in tree.nodes or n.parent_id!=tree.nodes[nid].parent_id}
    retired = set(candidate.nodes.deleted) if local else set(tree.nodes)-set(candidate.nodes)
    if local:
        delta = StructureDelta(op.kind, tree.tree_id, tree.version, candidate.version,
                               tuple(sorted(modified)), tuple(sorted(retired)), tuple(sorted(affected-changed)))
        return candidate, delta
    rollback = {"nodes":{nid:asdict(tree.nodes[nid]) for nid in sorted((modified|retired)&tree.nodes.keys())},
                "created":sorted(set(candidate.nodes)-set(tree.nodes)),
                "metadata":{key:deepcopy(value) for key,value in vars(tree).items()
                            if key!="nodes" and value!=getattr(candidate,key)}}
    delta = StructureDelta(op.kind, tree.tree_id, tree.version, candidate.version,
                           tuple(sorted(modified)), tuple(sorted(retired)),
                           tuple(sorted(affected-changed)), rollback)
    return candidate, delta


def rollback(tree,delta):

    if tree.tree_id!=delta.tree_id or tree.version!=delta.after_version:
        raise ValueError("rollback requires the matching committed structure version")
    result=deepcopy(tree)
    for nid in delta.rollback["created"]:
        result.nodes.pop(nid,None)
    for nid,value in delta.rollback["nodes"].items():
        result.nodes[nid]=MemTreeNode(**deepcopy(value))
    for key,value in delta.rollback["metadata"].items():
        setattr(result,key,deepcopy(value))
    validate(result)
    return result


def _sibling(tree, node, children, level=None):
    level = node.level if level is None else level
    nid = btree._new_node_id(tree.tree_id, level, btree._next_node_index(tree, level))
    result = MemTreeNode(nid, tree.tree_id, level, node.time_start, node.time_end,
                         "", True, list(children), 0, node.parent_id)
    tree.nodes[nid] = result
    for cid in children:
        tree.nodes[cid].parent_id = nid
    btree._refresh_internal_node_from_children(tree, result)
    btree._mark_node_dirty(tree, nid)
    return result
