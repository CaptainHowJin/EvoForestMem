from __future__ import annotations

import re
from collections import defaultdict

IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")
URL = re.compile(r"https?://[^\s<>\"\)]+", re.I)
ALIAS = re.compile(r'\b([A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+)\s*\((?:aka|also called)\s+["\']?([^\)"\']+)', re.I)


class IncrementalArtifactRouter:
    view = "artifact"

    def __init__(self):
        self.members: dict[str, tuple[str, ...]] = {}
        self.provenance: dict[str, tuple[dict, ...]] = {}
        self.aliases = defaultdict(set)
        self.hints: dict[str, tuple[str, ...]] = {}
        self.labels = {}
        self.alias_sources = {}
        self._first = defaultdict(set)

    def route(self, msg):
        if msg.msg_id in self.members:
            return self.members[msg.msg_id]
        keys, records = set(), []
        for identifier in IDENTIFIER.findall(msg.text):
            if not (any(c.isdigit() for c in identifier) or "_" in identifier or identifier.isupper()):
                continue
            key = f"{msg.channel_key}:{identifier.lower()}"
            keys.add(key)
            self.labels[key] = identifier
            records.append({"key": key, "source": msg.msg_id, "kind": "literal_identifier"})


        for url in URL.findall(msg.text):
            url = url.rstrip(".,;")
            key = "url:" + url
            keys.add(key)
            self.labels[key] = url
            records.append({"key": key, "source": msg.msg_id, "kind": "literal_url"})

        for identifier, alias in ALIAS.findall(msg.text):
            key = f"{msg.channel_key}:{identifier.lower()}"
            keys.add(key)
            self.labels[key] = identifier
            alias = alias.strip().lower()
            self.aliases[msg.channel_key, alias].add(key)
            self.alias_sources.setdefault((msg.channel_key, alias, key), msg.msg_id)
            self._first[msg.channel_key, alias.split()[0]].add(alias)
        hints = set()
        words = re.findall(r"[\w-]+", msg.text.lower())
        for word in dict.fromkeys(words):
            for alias in self._first.get((msg.channel_key, word), ()):
                if not re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", msg.text, re.I):
                    continue
                matches = self.aliases[msg.channel_key, alias]
                if len(matches) == 1:
                    keys.update(matches)
                    records.append({"key": next(iter(matches)), "source": msg.msg_id,
                                    "kind": "observed_alias", "alias": alias,
                                    "declaration_source": self.alias_sources[msg.channel_key, alias, next(iter(matches))]})
                else:
                    hints.update(matches)
        self.members[msg.msg_id] = tuple(sorted(keys))
        self.provenance[msg.msg_id] = tuple(records)
        self.hints[msg.msg_id] = tuple(sorted(hints))
        return self.members[msg.msg_id]

    def label_for(self, key):
        return self.labels.get(key, key)

    def stats(self):
        return {"mode": "incremental_scoped", "keys": len(self.labels),
                "members": sum(bool(v) for v in self.members.values()),
                "unresolved": sum(bool(v) for v in self.hints.values()),
                "unrouted": sum(not v for v in self.members.values())}
