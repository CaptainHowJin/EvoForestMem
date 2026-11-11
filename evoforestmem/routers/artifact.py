from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ..log.message import Message
from ..retrieve.s0_analyse import _contains_run, norm_artifact_tokens


def _alias_runs(key: str, row: Mapping) -> tuple[tuple[str, ...], ...]:

    raw_aliases = row.get("aliases") or ()
    runs: list[tuple[str, ...]] = []
    tokens = row.get("tokens")
    if tokens and all(isinstance(value, str) for value in tokens):
        runs.append(tuple(str(value) for value in tokens if str(value)))
    elif tokens:
        for value in tokens:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                runs.append(tuple(str(token) for token in value if str(token)))
    for alias in raw_aliases:
        run = norm_artifact_tokens(str(alias))
        if run:
            runs.append(run)
    if not runs:
        fallback = norm_artifact_tokens(str(row.get("label") or key).replace("_", " "))
        if fallback:
            runs.append(fallback)
    return tuple(dict.fromkeys(runs))


class ArtifactRouter:
    view = "artifact"

    def __init__(self, registry=None, *, lexicon: str | Path | None = None,
                 entries: Mapping[str, Mapping] | None = None, batch: str = "004") -> None:
        if entries is None:
            if lexicon is None:


                entries = {}
            else:
                with open(lexicon, encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh) or {}
                entries = (doc.get("artifact_lexicon") or {}).get(str(batch), {})
        self._labels: dict[str, str] = {}
        self._aliases: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._first: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
        member_keys: dict[str, list[str]] = {}
        for key, row in sorted(entries.items()):
            name = str(key)
            runs = _alias_runs(name, row)
            if not runs:
                continue
            self._aliases[name] = runs
            label = " ".join(str(row.get("label") or name).split())
            self._labels[name] = label[:60]
            for run in runs:
                self._first.setdefault(run[0], []).append((name, run))
            for member in row.get("members") or ():
                bucket = member_keys.setdefault(str(member), [])
                if name not in bucket:
                    bucket.append(name)
        self._members = {mid: tuple(keys) for mid, keys in member_keys.items()}
        self._routed = 0
        self._unrouted = 0

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._aliases))

    def matching_keys(self, text: str) -> tuple[str, ...]:

        toks = norm_artifact_tokens(text)
        found: list[str] = []
        seen: set[str] = set()
        for token in dict.fromkeys(toks):
            for key, run in self._first.get(token, ()):
                if key not in seen and _contains_run(toks, run):
                    seen.add(key)
                    found.append(key)
        return tuple(found)

    def route(self, msg: Message) -> tuple[str, ...]:

        keys = self._members.get(msg.msg_id, ())
        if keys:
            self._routed += 1
        else:
            self._unrouted += 1
        return keys

    def label_for(self, key: str) -> str:
        return self._labels.get(key, key)[:60]

    def stats(self) -> dict[str, int]:
        return {"keys": len(self._aliases), "members": len(self._members),
                "routed": self._routed, "unrouted": self._unrouted}
