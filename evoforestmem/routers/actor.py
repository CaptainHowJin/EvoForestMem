from __future__ import annotations

import re
from typing import Callable

from evoforestmem.log.message import Message
from evoforestmem.log.registry import Registry


ActorFallback = Callable[[Message], tuple[str, ...]]

_CUE = re.compile(
    r"(?:\b(?:own(?:s|ed|ing)?|lead(?:s|ing)?|drive(?:s|n|ing)?|responsible|assigned?|"
    r"approve(?:s|d|ing)?|manage(?:s|d|ing)?|report(?:s|ed|ing)?|describ(?:e|es|ed|ing)|"
    r"role|title)\b|(?:can|could|would|please)\s+@?[\w.-]+\s+(?:own|lead|drive|take|"
    r"handle|confirm|approve|review)|(?:负责|主导|牵头|承担|分配给|指派给|汇报给|审批|"
    r"确认|职位|角色|请|能否|可以))",
    re.IGNORECASE,
)


def is_substantive_about(msg: Message, mention: str) -> bool:

    if not mention or not msg.text:
        return False
    text = msg.text
    hit = re.search(re.escape(mention), text, re.IGNORECASE)
    if hit is None:
        return False
    window = text[max(0, hit.start() - 80): min(len(text), hit.end() + 80)]
    return _CUE.search(window) is not None


class ActorRouter:


    view = "actor"

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        fallback: ActorFallback | None = None,
        include_mentions: bool = False,
    ) -> None:
        self._registry = registry
        self._fallback = fallback
        self._include_mentions = include_mentions
        self._routed = 0
        self._unrouted = 0
        self._from_fallback = 0
        self._mention_placements = 0

    def route(self, msg: Message) -> tuple[str, ...]:
        keys: list[str] = []
        if msg.speaker_key:
            keys.append(msg.speaker_key)
            self._routed += 1
        elif self._fallback is not None:
            extra = [k for k in self._fallback(msg) if k]
            if extra:
                keys.extend(extra)
                self._from_fallback += 1
        if self._include_mentions:


            for mention in msg.tags.mentions:
                if not is_substantive_about(msg, mention):
                    continue
                key = self._registry.key(mention) if self._registry is not None else mention


                if self._registry is not None and key not in self._registry.keys:
                    continue
                if key and key not in keys:
                    keys.append(key)
                    self._mention_placements += 1
        if not keys:
            self._unrouted += 1
        return tuple(keys)

    def label_for(self, key: str) -> str:
        return self._registry.display(key) if self._registry is not None else key

    def stats(self) -> dict[str, int]:
        return {
            "routed": self._routed,
            "unrouted": self._unrouted,
            "from_fallback": self._from_fallback,
            "mention_placements": self._mention_placements,
        }
