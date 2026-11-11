from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator

from .message import Message, Tags


def content_id(channel_key: str, speaker_key: str, ts: float, text: str) -> str:

    h = hashlib.sha1()
    h.update(channel_key.encode("utf-8"))
    h.update(b"\x00")
    h.update(speaker_key.encode("utf-8"))
    h.update(b"\x00")
    h.update(f"{ts:.3f}".encode("ascii"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


class MessageLog:


    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._by_id: dict[str, int] = {}
        self._dup_counter: dict[str, int] = {}
        self._latest_ts: float | None = None


    def append(
        self,
        *,
        text: str,
        speaker_key: str,
        channel_key: str,
        ts: float,
        reply_to: str | None = None,
        tags: Tags | None = None,
    ) -> Message:
        base = content_id(channel_key, speaker_key, ts, text)
        msg_id = base
        if msg_id in self._by_id:


            n = self._dup_counter.get(base, 0) + 1
            self._dup_counter[base] = n
            msg_id = f"{base}~{n}"

        msg = Message(
            msg_id=msg_id,
            seq=len(self._messages),
            text=text,
            speaker_key=speaker_key,
            channel_key=channel_key,
            ts=ts,
            reply_to=reply_to,
            tags=tags or Tags(),
        )
        self._by_id[msg_id] = len(self._messages)
        self._messages.append(msg)
        self._latest_ts = ts if self._latest_ts is None else max(self._latest_ts, ts)
        return msg


    def get(self, msg_id: str) -> Message:
        return self._messages[self._by_id[msg_id]]

    def text(self, msg_id: str) -> str:

        return self.get(msg_id).text

    def at(self, seq: int) -> Message:
        return self._messages[seq]

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def latest_ts(self) -> float | None:

        return self._latest_ts

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __contains__(self, msg_id: object) -> bool:
        return msg_id in self._by_id

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def replace_tags(self, msg_id: str, tags: Tags) -> Message:

        idx = self._by_id[msg_id]
        enriched = replace(self._messages[idx], tags=tags)
        self._messages[idx] = enriched
        return enriched

    def by_channel(self, channel_key: str) -> list[Message]:
        return [m for m in self._messages if m.channel_key == channel_key]

    def by_speaker(self, speaker_key: str) -> list[Message]:
        return [m for m in self._messages if m.speaker_key == speaker_key]

    def channels(self) -> list[str]:
        return sorted({m.channel_key for m in self._messages})

    def speakers(self) -> list[str]:
        return sorted({m.speaker_key for m in self._messages})


    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for m in self._messages:
                fh.write(json.dumps(m.to_dict(), ensure_ascii=False))
                fh.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MessageLog":
        log = cls()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                m = Message.from_dict(json.loads(line))
                log._by_id[m.msg_id] = len(log._messages)
                log._messages.append(m)
                log._latest_ts = m.ts if log._latest_ts is None else max(log._latest_ts, m.ts)
        return log


    def stats(self) -> dict:
        if not self._messages:
            return {"n": 0}
        ts = [m.ts for m in self._messages]
        lifecycles: dict[str, int] = {}
        n_tagged = {"mentions": 0, "artifacts": 0, "markers": 0, "deliverables": 0}
        for m in self._messages:
            if m.tags.lifecycle:
                lifecycles[m.tags.lifecycle] = lifecycles.get(m.tags.lifecycle, 0) + 1
            for f in n_tagged:
                if getattr(m.tags, f):
                    n_tagged[f] += 1
        from datetime import datetime, timezone

        fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")  
        return {
            "n": len(self._messages),
            "channels": len(self.channels()),
            "speakers": len(self.speakers()),
            "date_range": [fmt(min(ts)), fmt(max(ts))],
            "with_reply_to": sum(1 for m in self._messages if m.reply_to),
            "tagged": n_tagged,
            "lifecycle": dict(sorted(lifecycles.items(), key=lambda kv: -kv[1])),
        }


def monotonic_violations(messages: Iterable[Message], key: str) -> int:

    last: dict[str, float] = {}
    bad = 0
    for m in messages:
        k = getattr(m, key)
        if k in last and m.ts < last[k]:
            bad += 1
        last[k] = m.ts
    return bad
