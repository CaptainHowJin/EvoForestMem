from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from .sparse import tokenize

Kind = Literal["document", "query"]


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
}

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_SEED = 20260730


def text_hash(text: str) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:

    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)


    np.maximum(norms, 1e-12, out=norms)
    return (m / norms).astype(np.float32, copy=False)


@runtime_checkable
class Embedder(Protocol):


    name: str

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray: ...


def encode_one(embedder: Embedder, text: str, *, kind: Kind = "document") -> np.ndarray:

    return embedder.encode([text], kind=kind)[0]


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "rows": self.rows}


def embedding_identity(model: str, max_seq_length: int | None = None) -> str:

    return model if max_seq_length is None else f"{model}@maxseq={int(max_seq_length)}"


class EmbeddingCache:


    def __init__(self, root: str | Path, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim
        slug = model.replace("/", "__").replace(":", "_")
        self.path = Path(root) / slug
        self.repaired = 0
        self._keys: dict[str, int] = {}
        self._rows: list[np.ndarray] = []
        self._load()


    @property
    def _meta_file(self) -> Path:
        return self.path / "meta.json"

    @property
    def _keys_file(self) -> Path:
        return self.path / "keys.txt"

    @property
    def _vec_file(self) -> Path:
        return self.path / "vectors.f32"

    def _load(self) -> None:
        if not self._meta_file.is_file():
            return
        meta = json.loads(self._meta_file.read_text(encoding="utf-8"))
        if int(meta.get("dim", -1)) != self.dim:
            raise ValueError(
                f"cache at {self.path} holds {meta.get('dim')}-d vectors for "
                f"{meta.get('model')!r}, but {self.model!r} is {self.dim}-d; "
                "delete the directory or point --embed-cache elsewhere"
            )
        keys = self._keys_file.read_text(encoding="utf-8").split() if self._keys_file.is_file() else []
        flat = np.fromfile(self._vec_file, dtype=np.float32) if self._vec_file.is_file() else np.empty(0, np.float32)
        n_vec = flat.size // self.dim
        n = min(len(keys), n_vec)
        self.repaired = max(len(keys), n_vec) - n
        mat = flat[: n * self.dim].reshape(n, self.dim)
        for i, key in enumerate(keys[:n]):
            if key not in self._keys:
                self._keys[key] = i
        self._rows = [mat[i] for i in range(n)]


    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def lookup(self, keys: Sequence[str]) -> list[np.ndarray | None]:
        return [self._rows[self._keys[k]] if k in self._keys else None for k in keys]

    def put(self, keys: Sequence[str], matrix: np.ndarray) -> int:

        mat = np.asarray(matrix, dtype=np.float32).reshape(len(keys), self.dim)
        fresh_keys: list[str] = []
        fresh_rows: list[np.ndarray] = []
        seen: set[str] = set()
        for i, key in enumerate(keys):
            if key in self._keys or key in seen:
                continue
            seen.add(key)
            fresh_keys.append(key)
            fresh_rows.append(mat[i])
        if not fresh_keys:
            return 0

        self.path.mkdir(parents=True, exist_ok=True)
        if not self._meta_file.is_file():
            self._meta_file.write_text(
                json.dumps({"model": self.model, "dim": self.dim}, indent=2) + "\n",
                encoding="utf-8",
            )
        block = np.ascontiguousarray(np.stack(fresh_rows), dtype=np.float32)

        with open(self._vec_file, "ab") as fh:
            fh.write(block.tobytes(order="C"))
        with open(self._keys_file, "a", encoding="utf-8") as fh:
            fh.write("".join(f"{k}\n" for k in fresh_keys))

        for key, row in zip(fresh_keys, fresh_rows):
            self._keys[key] = len(self._rows)
            self._rows.append(row)
        return len(fresh_keys)


class HashEmbedder:


    def __init__(self, dim: int = 64, *, seed: int = DEFAULT_SEED, name: str | None = None) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self._dim = dim
        self.seed = seed
        self.name = name or f"hash-{dim}d-s{seed}"
        self._salt = seed.to_bytes(8, "little", signed=False)

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8, salt=self._salt[:8]).digest()
        v = int.from_bytes(h, "little")


        return (v >> 1) % self._dim, (1.0 if v & 1 else -1.0)

    def encode(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for r, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                j, sign = self._bucket(tok)
                out[r, j] += sign * (1.0 + math.log(tf))
        return l2_normalize(out)


class LocalEmbedder:


    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: str | Path | None = None,
        query_prefix: str | None = None,
        dim: int | None = None,
        seed: int = DEFAULT_SEED,
        max_seq_length: int | None = None,
    ) -> None:
        self.model = model
        self.max_seq_length = (
            int(max_seq_length) if max_seq_length is not None else None
        )
        if self.max_seq_length is not None and self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        self.name = embedding_identity(model, self.max_seq_length)
        self.batch_size = batch_size
        self.seed = seed
        self._device = device
        self._dim = dim if dim is not None else KNOWN_DIMS.get(model)
        self._model_obj: Any = None


        self._encode_lock = threading.RLock()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache: EmbeddingCache | None = None
        self._hits = 0
        self._misses = 0
        if query_prefix is not None:
            self.query_prefix = query_prefix
        elif model.startswith("BAAI/bge-") and "-en" in model:
            self.query_prefix = BGE_QUERY_PREFIX
        elif model.startswith("Qwen/Qwen3-Embedding"):
            self.query_prefix = QWEN3_QUERY_PREFIX
        else:
            self.query_prefix = ""


    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        return self._device

    def _encoder(self) -> Any:
        if self._model_obj is not None:
            return self._model_obj
        with self._encode_lock:


            if self._model_obj is None:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.manual_seed(self.seed)
                model = SentenceTransformer(self.model, device=self.device)
                if self.max_seq_length is not None:
                    model.max_seq_length = self.max_seq_length
                model.eval()
                self._model_obj = model
        return self._model_obj

    @property
    def dim(self) -> int:
        if self._dim is None:
            enc = self._encoder()
            getter = getattr(enc, "get_embedding_dimension", None) or enc.get_sentence_embedding_dimension
            self._dim = int(getter())
        return self._dim

    @property
    def cache(self) -> EmbeddingCache | None:
        if self._cache_dir is not None and self._cache is None:
            self._cache = EmbeddingCache(self._cache_dir, self.name, self.dim)
        return self._cache

    def stats(self) -> CacheStats:
        c = self.cache
        return CacheStats(hits=self._hits, misses=self._misses, rows=len(c) if c else 0)


    def encode(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:


        with self._encode_lock:
            texts = list(texts)
            if not texts:
                return np.zeros((0, self.dim), dtype=np.float32)

            prepared = [self.query_prefix + t for t in texts] if kind == "query" else list(texts)
            keys = [text_hash(f"{kind}\x00{t}") for t in prepared]

            cache = self.cache
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            missing: list[int] = list(range(len(texts)))
            if cache is not None:
                missing = []
                for i, vec in enumerate(cache.lookup(keys)):
                    if vec is None:
                        missing.append(i)
                    else:
                        out[i] = vec
                self._hits += len(texts) - len(missing)
            self._misses += len(missing)

            if missing:
                fresh = self._forward([prepared[i] for i in missing])
                for slot, i in enumerate(missing):
                    out[i] = fresh[slot]
                if cache is not None:
                    cache.put([keys[i] for i in missing], fresh)
            return out

    def _forward(self, texts: list[str]) -> np.ndarray:
        import torch

        enc = self._encoder()
        with torch.inference_mode():
            vecs = enc.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )


        return l2_normalize(np.asarray(vecs, dtype=np.float32).reshape(len(texts), self.dim))


class LocalServiceEmbedder(LocalEmbedder):


    def __init__(self, model=DEFAULT_MODEL, *, base_url, **kwargs):
        from urllib.parse import urlparse
        from urllib.request import urlopen
        parsed = urlparse(base_url)
        if parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('embedding service must be local')
        self.base_url = base_url.rstrip('/')
        with urlopen(self.base_url + '/models', timeout=30) as response:
            models = json.load(response)['data']
        matches = [row for row in models if row['id'] == model]
        if len(matches) != 1 or not matches[0].get('max_model_len'):
            raise ValueError('embedding service must expose the model and physical window')
        if matches[0].get('backend') != 'evoforestmem-sentence-transformers-v1':
            raise ValueError('embedding service differs from the reference SentenceTransformer pipeline')
        self.served_window = int(matches[0]['max_model_len'])
        super().__init__(model, max_seq_length=self.served_window, **kwargs)
        self.name += '@local-sentence-transformers-v1'
        self._tokenizer = None
        self.service_usage = {'calls': 0, 'tokens': 0}

    @property
    def device(self):
        return 'gpu-sentence-transformers-service'

    def _token_count(self, text):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
        return len(self._tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def document_chunks(self, text):
        from ..forest.hierarchical_summaries import split_losslessly
        return [(part, self._token_count(part)) for part in split_losslessly(
            text, lambda part: self._token_count(part) <= self.served_window)]

    def _forward(self, texts):
        from urllib.request import Request, urlopen
        if any(self._token_count(text) > self.served_window for text in texts):
            raise ValueError('embedding input exceeds physical window; refusing truncation')
        payload = {'model': self.model, 'input': texts, 'encoding_format': 'float'}
        request = Request(self.base_url + '/embeddings', json.dumps(payload).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=600) as response:
            data = json.load(response)
        rows = sorted(data['data'], key=lambda row: row['index'])
        if [row['index'] for row in rows] != list(range(len(texts))):
            raise ValueError('embedding service omitted or duplicated input rows')
        matrix = np.asarray([row['embedding'] for row in rows], dtype=np.float32)
        if matrix.shape != (len(texts), self.dim) or not np.isfinite(matrix).all():
            raise ValueError('invalid embedding matrix')
        self.service_usage['calls'] += 1
        self.service_usage['tokens'] += int(data.get('usage', {}).get('prompt_tokens', 0))
        return l2_normalize(matrix)


def build(name: str = "local", **kw: Any) -> Embedder:

    if name == "hash":
        return HashEmbedder(**kw)
    if name == "local":
        return LocalEmbedder(**kw)
    return LocalEmbedder(name, **kw)


def encode_batched(
    embedder: Embedder,
    texts: Iterable[str],
    *,
    chunk: int = 512,
    kind: Kind = "document",
    progress: Callable[[int], None] | None = None,
) -> np.ndarray:

    blocks: list[np.ndarray] = []
    buf: list[str] = []
    for t in texts:
        buf.append(t)
        if len(buf) >= chunk:
            blocks.append(embedder.encode(buf, kind=kind))
            if progress is not None:
                progress(len(buf))
            buf = []
    if buf:
        blocks.append(embedder.encode(buf, kind=kind))
        if progress is not None:
            progress(len(buf))
    if not blocks:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.vstack(blocks).astype(np.float32, copy=False)
