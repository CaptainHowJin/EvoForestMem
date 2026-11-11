from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from .embedder import Embedder, l2_normalize

VECTORS_FILE = "vectors.npy"
IDS_FILE = "ids.json"
MANIFEST_FILE = "manifest.json"
ACTIVE_FILE = "active.npy"


class StaleIndexError(RuntimeError):
    pass


def fingerprint(items: Iterable[str]) -> str:

    h = hashlib.sha256()
    for it in items:
        h.update(it.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@runtime_checkable
class VectorSearcher(Protocol):


    dim: int

    def __len__(self) -> int: ...

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]: ...


@dataclass(frozen=True, slots=True)
class DenseManifest:


    model: str
    dim: int
    count: int


    corpus_hash: str


    id_hash: str
    active_count: int | None = None
    active_hash: str | None = None
    normalized: bool = True
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dim": self.dim,
            "count": self.count,
            "corpus_hash": self.corpus_hash,
            "id_hash": self.id_hash,
            "active_count": self.active_count if self.active_count is not None else self.count,
            "active_hash": self.active_hash or self.id_hash,
            "normalized": self.normalized,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DenseManifest":
        return cls(
            model=d["model"],
            dim=int(d["dim"]),
            count=int(d["count"]),
            corpus_hash=d["corpus_hash"],
            id_hash=d.get("id_hash", d["corpus_hash"]),
            active_count=int(d.get("active_count", d["count"])),
            active_hash=d.get("active_hash", d.get("id_hash", d["corpus_hash"])),
            normalized=bool(d.get("normalized", True)),
            created_at=d.get("created_at", ""),
        )


class DenseIndex:


    def __init__(
        self,
        dim: int,
        *,
        model: str = "",
        normalize: bool = True,
        capacity: int = 1024,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.model = model
        self.normalize = normalize
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._buf = np.zeros((max(capacity, 1), dim), dtype=np.float32)
        self._active = np.zeros(max(capacity, 1), dtype=bool)
        self._n = 0

        self._hash = hashlib.sha256()


    def _reserve(self, extra: int) -> None:
        need = self._n + extra
        if need <= self._buf.shape[0]:
            return
        cap = self._buf.shape[0]
        while cap < need:
            cap *= 2
        grown = np.zeros((cap, self.dim), dtype=np.float32)
        grown[: self._n] = self._buf[: self._n]
        self._buf = grown
        active = np.zeros(cap, dtype=bool)
        active[: self._n] = self._active[: self._n]
        self._active = active

    def _prepare(self, matrix: np.ndarray, n: int) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float32).reshape(n, self.dim)
        return l2_normalize(m) if self.normalize else m

    def add(self, doc_id: str, vector: np.ndarray) -> int:

        if doc_id in self._pos:
            raise ValueError(f"duplicate doc_id: {doc_id!r}")
        self._reserve(1)
        self._buf[self._n] = self._prepare(vector, 1)[0]
        self._active[self._n] = True
        self._pos[doc_id] = self._n
        self._ids.append(doc_id)
        self._hash.update(doc_id.encode("utf-8"))
        self._hash.update(b"\x00")
        self._n += 1
        return self._n - 1

    def add_many(self, doc_ids: Sequence[str], matrix: np.ndarray) -> int:

        doc_ids = list(doc_ids)
        if not doc_ids:
            return 0
        block = self._prepare(matrix, len(doc_ids))
        seen: set[str] = set()
        for d in doc_ids:
            if d in self._pos or d in seen:
                raise ValueError(f"duplicate doc_id: {d!r}")
            seen.add(d)
        self._reserve(len(doc_ids))
        self._buf[self._n : self._n + len(doc_ids)] = block
        self._active[self._n : self._n + len(doc_ids)] = True
        for d in doc_ids:
            self._pos[d] = self._n
            self._ids.append(d)
            self._hash.update(d.encode("utf-8"))
            self._hash.update(b"\x00")
            self._n += 1
        return len(doc_ids)

    def add_pairs(self, pairs: Iterable[tuple[str, np.ndarray]]) -> int:
        n = 0
        for doc_id, vec in pairs:
            self.add(doc_id, vec)
            n += 1
        return n

    def replace(self, doc_id: str, vector: np.ndarray) -> None:

        row = self._pos.get(doc_id)
        if row is None:
            raise KeyError(f"replace: unknown doc_id {doc_id!r}")
        self._buf[row] = self._prepare(vector, 1)[0]

    def remove(self, doc_id: str) -> int:

        row = self._pos.pop(doc_id, None)
        if row is None:
            raise KeyError(f"remove: unknown doc_id {doc_id!r}")
        self._active[row] = False
        return row

    def compact_rows(self) -> dict[int, int]:

        live = np.flatnonzero(self._active[: self._n])
        remap = {int(old): new for new, old in enumerate(live)}
        if len(live) == self._n:
            return remap
        ids = [self._ids[int(row)] for row in live]
        vectors = self._buf[live].copy()
        capacity = max(len(ids), 1)
        self._buf = np.zeros((capacity, self.dim), dtype=np.float32)
        self._active = np.zeros(capacity, dtype=bool)
        if ids:
            self._buf[:len(ids)] = vectors
            self._active[:len(ids)] = True
        self._ids = ids
        self._pos = {doc_id: row for row, doc_id in enumerate(ids)}
        self._n = len(ids)
        self._hash = hashlib.sha256()
        for doc_id in ids:
            self._hash.update(doc_id.encode("utf-8"))
            self._hash.update(b"\x00")
        return remap


    def __len__(self) -> int:
        return self._n

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._pos

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    @property
    def active(self) -> np.ndarray:

        return self._active[: self._n]

    @property
    def active_ids(self) -> tuple[str, ...]:
        return tuple(doc_id for doc_id, active in zip(self._ids, self.active) if active)

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active))

    @property
    def matrix(self) -> np.ndarray:

        return self._buf[: self._n]

    def row(self, doc_id: str) -> int:
        return self._pos[doc_id]

    def vector(self, doc_id: str) -> np.ndarray:
        return self._buf[self._pos[doc_id]].copy()

    def fingerprint(self) -> str:

        return self._hash.copy().hexdigest()


    def _query(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.size != self.dim:
            raise ValueError(f"query has {q.size} dims, index has {self.dim}")
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-12 else q

    def scores(self, query: np.ndarray,
               visible: np.ndarray | None = None) -> np.ndarray:

        if self._n == 0:
            return np.zeros(0, dtype=np.float32)
        sc = self.matrix @ self._query(query)
        effective = self.active.copy()
        if visible is not None:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape[0] != self._n:
                raise ValueError(
                    f"visible mask has {visible.shape[0]} rows, index has {self._n}")
            effective &= visible
        sc = np.where(effective, sc, np.float32("-inf"))
        return sc

    def score_map(self, query: np.ndarray,
                  visible: np.ndarray | None = None) -> dict[str, float]:

        return {d: float(s) for d, s in zip(self._ids, self.scores(query, visible))
                if np.isfinite(s)}

    def score_subset(self, query: np.ndarray, doc_ids: Iterable[str]) -> dict[str, float]:

        ids = [mid for mid in dict.fromkeys(doc_ids) if mid in self._pos]
        if not ids:
            return {}
        rows = [self._pos[mid] for mid in ids]
        values = self._buf[rows] @ self._query(query)
        return dict(zip(ids, map(float, values)))

    def search(self, query: np.ndarray, k: int = 10,
               visible: np.ndarray | None = None) -> list[tuple[str, float]]:

        sc = self.scores(query, visible)
        out = [(self._ids[i], float(sc[i])) for i in top_rows(sc, k)]
        return [(d, s) for d, s in out if np.isfinite(s)]


    def manifest(self, corpus_hash: str | None = None) -> DenseManifest:
        idh = self.fingerprint()
        return DenseManifest(
            model=self.model,
            dim=self.dim,
            count=self._n,
            corpus_hash=corpus_hash or idh,
            id_hash=idh,
            active_count=self.active_count,
            active_hash=fingerprint(self.active_ids),
            normalized=self.normalize,
            created_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def save(self, path: str | Path, *, corpus_hash: str | None = None) -> Path:

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / VECTORS_FILE, np.ascontiguousarray(self.matrix))
        np.save(out / ACTIVE_FILE, np.ascontiguousarray(self.active))
        (out / IDS_FILE).write_text(json.dumps(self._ids, ensure_ascii=False), encoding="utf-8")
        (out / MANIFEST_FILE).write_text(
            json.dumps(self.manifest(corpus_hash).to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return out

    @classmethod
    def read_manifest(cls, path: str | Path) -> DenseManifest:
        return DenseManifest.from_dict(
            json.loads((Path(path) / MANIFEST_FILE).read_text(encoding="utf-8"))
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expect_corpus: str | None = None,
        expect_model: str | None = None,
    ) -> "DenseIndex":

        src = Path(path)
        man = cls.read_manifest(src)
        ids: list[str] = json.loads((src / IDS_FILE).read_text(encoding="utf-8"))
        vecs = np.load(src / VECTORS_FILE)
        active_path = src / ACTIVE_FILE
        active = np.load(active_path) if active_path.is_file() else np.ones(man.count, dtype=bool)

        if (vecs.shape != (man.count, man.dim) or len(ids) != man.count
                or active.shape != (man.count,)):
            raise StaleIndexError(
                f"{src}: manifest says {man.count}x{man.dim}, found "
                f"{vecs.shape} vectors and {len(ids)} ids"
            )
        recomputed = fingerprint(ids)
        if recomputed != man.id_hash:
            raise StaleIndexError(f"{src}: ids.json does not match manifest id_hash")
        active = np.asarray(active, dtype=bool)
        active_ids = tuple(doc_id for doc_id, keep in zip(ids, active) if keep)
        if man.active_count is not None and int(np.count_nonzero(active)) != man.active_count:
            raise StaleIndexError(f"{src}: active mask count does not match manifest")
        if man.active_hash is not None and fingerprint(active_ids) != man.active_hash:
            raise StaleIndexError(f"{src}: active mask does not match manifest")
        if expect_model is not None and expect_model != man.model:
            raise StaleIndexError(
                f"{src}: built with model {man.model!r}, caller expects {expect_model!r}"
            )
        if expect_corpus is not None and expect_corpus != man.corpus_hash:
            raise StaleIndexError(
                f"{src}: built over corpus {man.corpus_hash[:12]}..., caller has "
                f"{expect_corpus[:12]}...; rebuild the index"
            )

        idx = cls(man.dim, model=man.model, normalize=man.normalized, capacity=max(man.count, 1))


        idx._buf[: man.count] = vecs.astype(np.float32, copy=False)
        idx._active[: man.count] = active
        idx._ids = list(ids)
        idx._pos = {d: i for i, d in enumerate(ids) if active[i]}
        idx._n = man.count
        for d in ids:
            idx._hash.update(d.encode("utf-8"))
            idx._hash.update(b"\x00")
        return idx

    def is_fresh_for(self, corpus_hash: str) -> bool:
        return self.fingerprint() == corpus_hash


def top_rows(scores: np.ndarray, k: int) -> np.ndarray:

    n = int(scores.shape[0])
    k = max(0, min(int(k), n))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k < n:


        threshold = float(scores[np.argpartition(-scores, k - 1)[:k]].min())
        cand = np.flatnonzero(scores >= threshold)
        if cand.shape[0] == 0:
            cand = np.arange(n)
    else:
        cand = np.arange(n)

    return cand[np.lexsort((cand, -scores[cand]))][:k].astype(np.int64, copy=False)


def build_index(
    embedder: Embedder,
    items: Iterable[tuple[str, str]],
    *,
    chunk: int = 512,
) -> DenseIndex:

    idx = DenseIndex(embedder.dim, model=getattr(embedder, "name", ""))
    ids: list[str] = []
    texts: list[str] = []
    for doc_id, text in items:
        ids.append(doc_id)
        texts.append(text)
        if len(ids) >= chunk:
            idx.add_many(ids, embedder.encode(texts))
            ids, texts = [], []
    if ids:
        idx.add_many(ids, embedder.encode(texts))
    return idx
