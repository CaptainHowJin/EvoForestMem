from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reply_roots(
    payload_ids: Sequence[str], reply_to: Mapping[str, str | None]
) -> dict[str, str]:

    allowed = set(payload_ids)
    roots: dict[str, str] = {}
    for start in payload_ids:
        if start in roots:
            continue
        path: list[str] = []
        position: dict[str, int] = {}
        current = start
        while current not in roots and current not in position:
            position[current] = len(path)
            path.append(current)
            parent = reply_to.get(current)
            if not parent or parent not in allowed:
                roots[current] = current
                break
            current = parent
        if current in roots:
            root = roots[current]
        else:
            cycle = path[position[current]:]
            root = min(cycle)
            for member in cycle:
                roots[member] = root
        for member in reversed(path):
            roots.setdefault(member, root)
    return roots


def thread_groups(
    payload_ids: Sequence[str],
    reply_to: Mapping[str, str | None],
    *,
    k: int,
    min_thread: int = 2,
) -> list[list[int]]:

    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if min_thread < 2:
        raise ValueError(f"min_thread must be >= 2, got {min_thread}")
    if not payload_ids:
        return []

    roots = reply_roots(payload_ids, reply_to)
    by_root: dict[str, list[int]] = defaultdict(list)
    for index, payload_id in enumerate(payload_ids):
        by_root[roots[payload_id]].append(index)

    groups: list[list[int]] = []
    fallback: list[int] = []
    ordered_threads = sorted(by_root.values(), key=lambda indexes: indexes[0])
    for indexes in ordered_threads:
        if len(indexes) < min_thread:
            fallback.extend(indexes)
            continue
        groups.extend(indexes[offset:offset + k] for offset in range(0, len(indexes), k))

    fallback.sort()
    groups.extend(fallback[offset:offset + k] for offset in range(0, len(fallback), k))
    return groups
