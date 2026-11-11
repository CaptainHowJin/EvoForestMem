from __future__ import annotations

from typing import Callable

from evoforestmem.log.message import Message
from evoforestmem.log.registry import Registry


ChannelFallback = Callable[[Message], tuple[str, ...]]


class ChannelRouter:


    view = "channel"

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        fallback: ChannelFallback | None = None,
    ) -> None:
        self._registry = registry
        self._fallback = fallback
        self._routed = 0
        self._unrouted = 0
        self._from_fallback = 0

    def route(self, msg: Message) -> tuple[str, ...]:
        if msg.channel_key:
            self._routed += 1
            return (msg.channel_key,)
        if self._fallback is not None:
            keys = tuple(k for k in self._fallback(msg) if k)
            if keys:
                self._from_fallback += 1
                return keys


        self._unrouted += 1
        return ()

    def label_for(self, key: str) -> str:
        return self._registry.display(key) if self._registry is not None else key

    def stats(self) -> dict[str, int]:
        return {
            "routed": self._routed,
            "unrouted": self._unrouted,
            "from_fallback": self._from_fallback,
        }
