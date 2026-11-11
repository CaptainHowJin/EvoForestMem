from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class Tags:


    mentions: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    lifecycle: str | None = None
    deliverables: tuple[str, ...] = ()
    tag_source: str = "rule"
    lifecycle_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.mentions:
            d["mentions"] = list(self.mentions)
        if self.artifacts:
            d["artifacts"] = list(self.artifacts)
        if self.markers:
            d["markers"] = list(self.markers)
        if self.lifecycle:
            d["lifecycle"] = self.lifecycle
        if self.deliverables:
            d["deliverables"] = list(self.deliverables)


        if self.tag_source != "rule":
            d["tag_source"] = self.tag_source
        if self.lifecycle_confidence is not None:
            d["lifecycle_confidence"] = self.lifecycle_confidence
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Tags":
        d = d or {}
        confidence = d.get("lifecycle_confidence")
        return cls(
            mentions=tuple(d.get("mentions", ())),
            artifacts=tuple(d.get("artifacts", ())),
            markers=tuple(d.get("markers", ())),
            lifecycle=d.get("lifecycle"),
            deliverables=tuple(d.get("deliverables", ())),
            tag_source=str(d.get("tag_source", "rule")),
            lifecycle_confidence=(float(confidence) if confidence is not None else None),
        )


@dataclass(frozen=True, slots=True)
class Message:


    msg_id: str
    seq: int
    text: str
    speaker_key: str
    channel_key: str
    ts: float
    reply_to: str | None = None
    tags: Tags = field(default_factory=Tags)

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc)

    @property
    def date(self) -> str:
        return self.dt.strftime("%Y-%m-%d")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "msg_id": self.msg_id,
            "seq": self.seq,
            "text": self.text,
            "speaker_key": self.speaker_key,
            "channel_key": self.channel_key,
            "ts": self.ts,
        }
        if self.reply_to:
            d["reply_to"] = self.reply_to
        tags = self.tags.to_dict()
        if tags:
            d["tags"] = tags
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            msg_id=d["msg_id"],
            seq=d["seq"],
            text=d["text"],
            speaker_key=d["speaker_key"],
            channel_key=d["channel_key"],
            ts=d["ts"],
            reply_to=d.get("reply_to"),
            tags=Tags.from_dict(d.get("tags")),
        )
