from __future__ import annotations

import heapq
import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Literal

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")

IdfScheme = Literal["lucene", "okapi"]


def tokenize(text: str) -> list[str]:

    return _TOKEN.findall(text.lower())


class SparseIndex:


    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 idf_scheme: IdfScheme = "lucene", epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.idf_scheme: IdfScheme = idf_scheme
        self.epsilon = epsilon
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._df: Counter[str] = Counter()
        self._doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._total_len = 0
        self._positions: dict[str, int] = {}
        self._frequencies: list[Counter] = []
        self._prefix_len = [0]


        self._active_query_cache: ContextVar[dict[str, object] | None] = ContextVar(
            f"sparse_query_cache_{id(self)}", default=None
        )
        self._active_max_docs: ContextVar[int | None] = ContextVar(
            f"sparse_max_docs_{id(self)}", default=None
        )


    def add(self, doc_id: str, text: str) -> int:
        idx = len(self._doc_ids)
        toks = tokenize(text)
        tf = Counter(toks)
        self._positions[doc_id] = idx
        self._frequencies.append(tf)
        for term, freq in tf.items():
            self._postings[term].append((idx, freq))
            self._df[term] += 1
        self._doc_ids.append(doc_id)
        self._doc_len.append(len(toks))
        self._total_len += len(toks)
        self._prefix_len.append(self._total_len)
        return idx

    def add_many(self, pairs: Iterable[tuple[str, str]]) -> int:
        n = 0
        for doc_id, text in pairs:
            self.add(doc_id, text)
            n += 1
        return n


    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def avgdl(self) -> float:
        n = self._doc_count()
        if not n:
            return 0.0


        if self._active_max_docs.get() is None:
            return self._total_len / n
        return self._prefix_len[n] / n

    def _doc_count(self) -> int:
        limit = self._active_max_docs.get()
        return len(self._doc_ids) if limit is None else max(0, min(limit, len(self._doc_ids)))

    def _term_df(self, term: str) -> int:
        limit = self._active_max_docs.get()
        if limit is None:
            return self._df.get(term, 0)
        postings = self._postings.get(term, ())
        return bisect_left(postings, (limit, -1))

    def _idf(self, term: str, avg_idf: float | None = None) -> float:
        n = self._doc_count()
        df = self._term_df(term)
        if df == 0:
            return 0.0
        if self.idf_scheme == "okapi":


            idf = math.log(n - df + 0.5) - math.log(df + 0.5)
            if idf < 0 and avg_idf is not None:
                return self.epsilon * avg_idf
            return idf
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _average_idf(self) -> float:

        n = self._doc_count()
        if not self._df:
            return 0.0
        dfs = [self._term_df(term) for term in self._df]
        dfs = [df for df in dfs if df]
        if not dfs:
            return 0.0
        total = sum(math.log(n - df + 0.5) - math.log(df + 0.5) for df in dfs)
        return total / len(dfs)


    def _compute_score_indices(self, query: str) -> dict[int, float]:

        n_docs = self._doc_count()
        if not n_docs:
            return {}
        avg_idf = self._average_idf() if self.idf_scheme == "okapi" else None
        avgdl = self.avgdl or 1.0
        acc: dict[int, float] = defaultdict(float)


        for term in tokenize(query):
            if term not in self._postings:
                continue
            idf = self._idf(term, avg_idf)
            for idx, freq in self._postings[term]:
                if idx >= n_docs:
                    break
                dl = self._doc_len[idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                acc[idx] += idf * freq * (self.k1 + 1.0) / denom
        return dict(acc)

    def _score_indices(self, query: str) -> dict[int, float]:

        cache = self._active_query_cache.get()
        scores = cache.get("scores") if cache is not None else None
        key = (query, self._doc_count())
        if isinstance(scores, dict) and key in scores:
            cache["hits"] = int(cache.get("hits", 0)) + 1
            return scores[key]
        result = self._compute_score_indices(query)
        if isinstance(scores, dict):
            scores[key] = result
            cache["misses"] = int(cache.get("misses", 0)) + 1
        return result

    @contextmanager
    def query_cache(self):

        state: dict[str, object] = {"scores": {}, "hits": 0, "misses": 0}
        token = self._active_query_cache.set(state)
        try:
            yield state
        finally:
            self._active_query_cache.reset(token)

    @contextmanager
    def snapshot(self, max_docs: int | None):

        token = self._active_max_docs.set(
            None if max_docs is None else max(0, min(int(max_docs), len(self._doc_ids)))
        )
        try:
            yield self
        finally:
            self._active_max_docs.reset(token)

    def scores(self, query: str) -> dict[str, float]:

        return {self._doc_ids[i]: score for i, score in self._score_indices(query).items()}

    def score_subset(self, query: str, doc_ids: Iterable[str]) -> dict[str, float]:

        n = self._doc_count()
        avgdl = self.avgdl or 1.0
        avg_idf = self._average_idf() if self.idf_scheme == "okapi" else None
        terms = [(term, self._idf(term, avg_idf)) for term in tokenize(query)]
        result = {}
        for mid in dict.fromkeys(doc_ids):
            idx = self._positions.get(mid)
            if idx is None or idx >= n:
                continue
            norm = self.k1 * (1.0 - self.b + self.b * self._doc_len[idx] / avgdl)
            tf = self._frequencies[idx]
            result[mid] = sum(idf * tf[term] * (self.k1 + 1.0) / (tf[term] + norm)
                              for term, idf in terms if tf[term])
        return result

    def postings_work(self, query: str) -> int:
        return sum(self._term_df(term) for term in tokenize(query))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:

        if k <= 0:
            return []
        scores = self._score_indices(query)
        rows = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self._doc_ids[index], score) for index, score in rows]

    def score_all(self, query: str) -> list[float]:

        sc = self.scores(query)
        return [sc.get(d, 0.0) for d in self._doc_ids]
