from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from evoforestmem.log.message import Message
from evoforestmem.log.registry import Registry

from .actor import ActorRouter
from .artifact import ArtifactRouter
from .channel import ChannelRouter

__all__ = [
    "Router", "ActorRouter", "ArtifactRouter", "ChannelRouter",
    "build", "default_routers",
]


@runtime_checkable
class Router(Protocol):


    view: str

    def route(self, msg: Message) -> tuple[str, ...]:

        ...

    def label_for(self, key: str) -> str:

        ...

    def stats(self) -> dict[str, int]:
        ...


def build(view: str, registry: Registry | None = None, **kw) -> Router:
    if view == "channel":
        return ChannelRouter(registry, **kw)
    if view == "actor":
        return ActorRouter(registry, **kw)
    if view == "artifact":
        return ArtifactRouter(registry, **kw)
    raise ValueError(f"unknown view: {view!r}")


def default_routers(
    views: Sequence[str] = ("channel", "actor"),
    *,
    registries: dict[str, Registry] | None = None,
    router_kwargs: dict[str, dict] | None = None,
) -> tuple[Router, ...]:

    registries = registries or {}
    router_kwargs = router_kwargs or {}
    return tuple(build(v, registries.get(v), **router_kwargs.get(v, {})) for v in views)
