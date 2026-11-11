from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_SLUG_STRIP = re.compile(r"[^\w\s-]", re.UNICODE)
_SLUG_SPACE = re.compile(r"[\s_-]+")


def slug(name: str) -> str:

    s = unicodedata.normalize("NFKC", name).strip().lower()
    s = _SLUG_STRIP.sub("", s)
    s = _SLUG_SPACE.sub("_", s)
    return s or "unknown"


@dataclass
class Registry:


    kind: str = "speaker"
    _by_key: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def add(self, display: str) -> str:
        key = slug(display)


        self._by_key.setdefault(key, display)
        self._counts[key] = self._counts.get(key, 0) + 1
        return key

    def key(self, display: str) -> str:
        return slug(display)

    def display(self, key: str) -> str:
        return self._by_key.get(key, key)

    def count(self, key: str) -> int:
        return self._counts.get(key, 0)

    @property
    def keys(self) -> list[str]:
        return sorted(self._by_key)

    @property
    def names(self) -> set[str]:
        return set(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)


    def probable_aliases(self) -> list[tuple[str, str, str]]:

        out: list[tuple[str, str, str]] = []
        by_tokens: dict[frozenset[str], list[str]] = {}
        for key in self._by_key:
            toks = frozenset(key.split("_"))
            if len(toks) > 1:
                by_tokens.setdefault(toks, []).append(key)
        for toks, keys in by_tokens.items():
            if len(keys) > 1:
                keys.sort(key=lambda k: -self.count(k))
                for other in keys[1:]:
                    out.append((keys[0], other, "token-permutation"))
        return out


    def to_dict(self) -> dict:
        return {"kind": self.kind, "by_key": self._by_key, "counts": self._counts}

    @classmethod
    def from_dict(cls, d: dict) -> "Registry":
        r = cls(kind=d.get("kind", "speaker"))
        r._by_key = dict(d.get("by_key", {}))
        r._counts = {k: int(v) for k, v in (d.get("counts") or {}).items()}
        return r

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def SpeakerRegistry() -> Registry:
    return Registry(kind="speaker")


def ChannelRegistry() -> Registry:
    return Registry(kind="channel")
