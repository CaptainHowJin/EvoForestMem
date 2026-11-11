from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..log.store import MessageLog
from .s0_analyse import norm_tokens


_NORMATIVE = frozenset({"mandate", "ratified", "decided", "effective_now",
                        "supersede", "signoff"})


_SCAFFOLD = frozenset(norm_tokens(
    "what which who whom whose when where why how many much should would could can may "
    "the a an of for to in on at by with from and or but if then than that this these "
    "those is are was were be been being do does did done have has had will shall must "
    "please help me i we you they it there here about into over under again more most "
    "team project group everyone colleague colleagues member members question answer "
    "option options following situation case scenario system platform module"
))

Verdict = str


@dataclass(frozen=True, slots=True)
class SufficiencyConfig:


    supported_coverage: float = 0.60

    unsupported_coverage: float = 0.25

    min_term_idf: float = 1.0


    announce_supported: bool = False


DEFAULT_SUFFICIENCY_CONFIG = SufficiencyConfig()


@dataclass(frozen=True, slots=True)
class Sufficiency:
    verdict: Verdict
    coverage: float
    n_terms: int
    n_covered: int
    missing: tuple[str, ...]
    normative_spans: int
    n_spans: int

    @property
    def announce(self) -> bool:
        return self.verdict != "supported"

    def render(self) -> list[str]:

        if not self.announce:
            return []
        missing = ", ".join(self.missing[:6])
        rule = ("no retrieved memory states a rule, approval, mandate or decision "
                "covering them" if self.normative_spans == 0
                else f"{self.normative_spans} retrieved memory(ies) state a rule, "
                     f"approval or decision")
        tail = (f" Key terms with no match: {missing}." if missing else "")
        return [f"    (evidence check: {self.n_covered} of {self.n_terms} key terms from "
                f"the question appear in the {self.n_spans} retrieved memories; {rule}."
                f"{tail} Anything not stated above is unsupported by this memory.)"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "coverage": round(self.coverage, 4),
            "n_terms": self.n_terms,
            "n_covered": self.n_covered,
            "missing": list(self.missing),
            "normative_spans": self.normative_spans,
            "n_spans": self.n_spans,
        }


class SufficiencyStatistics:

    def __init__(self):
        self.df = Counter()
        self.count = 0
        self.documents_read = 0
        self._snapshot = None

    def sync(self, log):
        if len(log) < self.count:
            raise ValueError('sufficiency prefix moved backwards')
        while self.count < len(log):
            self.df.update(set(norm_tokens(log.at(self.count).text)))
            self.count += 1
            self.documents_read += 1
            self._snapshot = None
        return self

    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = dict(self.df), self.count, min(self.df.values(), default=0)
        return self._snapshot


class _IDFView:
    def __init__(self, statistics):
        self.df, n, self.minimum = statistics.snapshot()
        self.n = max(1, n)

    def get(self, word, default=None):
        c = self.df.get(word)
        return math.log(1.0 + (self.n - c + .5) / (c + .5)) if c else default


class SufficiencyGate:


    def __init__(self, log: MessageLog, *,
                 config: SufficiencyConfig = DEFAULT_SUFFICIENCY_CONFIG, statistics=None):
        self.log = log
        self.config = config
        if statistics is not None:
            self._idf = _IDFView(statistics)
            n = max(1, statistics.count)


            c = self._idf.minimum
            self._max_idf = math.log(1.0 + (n-c+.5)/(c+.5)) if c else 1.0
            return
        df: Counter[str] = Counter()
        for m in log:
            df.update(set(norm_tokens(m.text)))
        n = max(1, len(log))
        self._idf = {w: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}


        self._max_idf = max(self._idf.values(), default=1.0)


    def _key_terms(self, question: str) -> list[str]:
        out: list[str] = []
        for token in dict.fromkeys(norm_tokens(question)):
            if len(token) <= 2 or token in _SCAFFOLD:
                continue
            if self._idf.get(token, self._max_idf) < self.config.min_term_idf:
                continue
            out.append(token)
        return out

    def assess(self, question: str, cited: Sequence[str]) -> Sufficiency:
        terms = self._key_terms(question)
        present: set[str] = set()
        normative = 0
        for msg_id in cited:
            if msg_id not in self.log:
                continue
            msg = self.log.get(msg_id)
            present |= set(norm_tokens(msg.text))
            if _NORMATIVE & set(msg.tags.markers):
                normative += 1
        if not terms:
            return Sufficiency("supported", 1.0, 0, 0, (), normative, len(cited))
        weights = {t: self._idf.get(t, self._max_idf) for t in terms}
        total = sum(weights.values()) or 1.0
        covered = [t for t in terms if t in present]
        coverage = sum(weights[t] for t in covered) / total
        missing = tuple(sorted((t for t in terms if t not in present),
                               key=lambda t: -weights[t]))
        if coverage >= self.config.supported_coverage:
            verdict: Verdict = "supported"
        elif coverage < self.config.unsupported_coverage:
            verdict = "unsupported"
        else:
            verdict = "partial"
        return Sufficiency(verdict=verdict, coverage=coverage, n_terms=len(terms),
                           n_covered=len(covered), missing=missing,
                           normative_spans=normative, n_spans=len(cited))


    @staticmethod
    def as_block(verdict: Sufficiency) -> Any:
        return _SufficiencyBlock(verdict)


@dataclass(frozen=True, slots=True)
class _SufficiencyBlock:


    verdict: Sufficiency

    @property
    def msg_ids(self) -> tuple[str, ...]:
        return ()

    def render(self) -> list[str]:
        return self.verdict.render()


def build(log: MessageLog, *,
          config: SufficiencyConfig = DEFAULT_SUFFICIENCY_CONFIG) -> SufficiencyGate:
    return SufficiencyGate(log, config=config)
