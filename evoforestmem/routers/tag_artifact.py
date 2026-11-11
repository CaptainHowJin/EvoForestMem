import hashlib
import json
import unicodedata

from .incremental_artifact import IncrementalArtifactRouter
from ..retrieve.s0_analyse import norm_artifact_tokens


class TagArtifactRouter(IncrementalArtifactRouter):
    def __init__(self, *, min_messages=2):
        super().__init__()
        if min_messages < 1:
            raise ValueError("artifact min_messages must be at least 1")
        self.min_messages = min_messages
        self.query_keys = {}
        self.soft_keys = set()
        self.active_keys = set()
        self.pending = {}
        self._backfills = {}
        self._tag_members = 0

    def route(self, msg):
        if msg.msg_id in self.members:
            return tuple(k for k in self.members[msg.msg_id] if k in self.active_keys)
        keys = set(super().route(msg))
        records = list(self.provenance[msg.msg_id])
        tagged = False
        if msg.tags.tag_source in {"llm", "llm+rule"}:
            for raw in msg.tags.deliverables[:12]:
                label = " ".join(unicodedata.normalize("NFKC", raw).casefold().split())[:160]
                if not label or not norm_artifact_tokens(label):
                    continue
                identity = json.dumps([msg.channel_key, label], ensure_ascii=False)
                key = "tag:" + hashlib.sha256(identity.encode()).hexdigest()[:32]
                if key in keys:
                    continue
                keys.add(key)
                self.labels[key] = label
                records.append({"key": key, "source": msg.msg_id,
                                "kind": "inferred_deliverable", "identity_confirmed": False,
                                "scope": msg.channel_key})
                tagged = True
        self.members[msg.msg_id] = tuple(sorted(keys))
        self.provenance[msg.msg_id] = tuple(records)
        self._tag_members += int(tagged)
        for key in keys:
            if key not in self.active_keys:
                pending = self.pending.setdefault(key, [])
                pending.append((msg.msg_id, msg.ts))
                if len(pending) < self.min_messages:
                    continue
                self.active_keys.add(key)
                self._backfills[key] = tuple(pending[:-1])
                del self.pending[key]
                if key.startswith("tag:"):
                    self.soft_keys.add(key)
                tokens = norm_artifact_tokens(self.labels[key])
                if tokens:
                    self.query_keys[key] = (tuple(tokens),)
        return tuple(k for k in self.members[msg.msg_id] if k in self.active_keys)

    def take_backfill(self, key):

        return self._backfills.pop(key, ())

    def stats(self):
        return {**super().stats(), "mode": "tag_scoped", "tag_members": self._tag_members,
                "tag_groups": sum(k.startswith("tag:") for k in self.labels),
                "min_messages": self.min_messages, "active_groups": len(self.active_keys),
                "pending_groups": len(self.pending),
                "pending_memberships": sum(len(v) for v in self.pending.values()),
                "identity_policy": "inferred groups are search hints; no cross-channel merge"}
