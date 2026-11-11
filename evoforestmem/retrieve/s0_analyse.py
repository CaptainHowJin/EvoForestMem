from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ..log import tags as _tags
from ..log.registry import Registry, slug


_WORD = re.compile(r"[a-z0-9]+")


_SUFFIXES = ("ities", "ment", "ing", "ive", "ion", "ies", "ity", "ed", "es", "al",
             "ly", "s")
_STEM_FLOOR = 3


@functools.lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    while True:
        for suf in _SUFFIXES:
            if token.endswith(suf) and len(token) - len(suf) >= _STEM_FLOOR:
                token = token[: -len(suf)]
                break
        else:
            return token


def fold(text: str) -> str:

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def norm_tokens(text: str, *, do_stem: bool = True) -> tuple[str, ...]:
    toks = _WORD.findall(fold(text))
    return tuple(stem(t) for t in toks) if do_stem else tuple(toks)


def norm_artifact_tokens(text: str) -> tuple[str, ...]:

    return tuple("diagnos" if t in {"diagnosi", "diagnostic"} else t
                 for t in norm_tokens(text))


def _contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:

    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return False
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and tuple(haystack[i : i + m]) == tuple(needle):
            return True
    return False


class Lexicon:


    def __init__(self, phrases_by_channel: Mapping[str, Sequence[str]], *, do_stem: bool = True):
        self.do_stem = do_stem
        self._phrases: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._surface: dict[str, tuple[str, ...]] = {}
        for name, phrases in phrases_by_channel.items():
            key = slug(name)
            norm = []
            surf = []
            for p in phrases:
                t = norm_tokens(p, do_stem=do_stem)
                if t:
                    norm.append(t)
                    surf.append(p)


            order = sorted(range(len(norm)), key=lambda i: -len(norm[i]))
            self._phrases[key] = tuple(norm[i] for i in order)
            self._surface[key] = tuple(surf[i] for i in order)

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted(self._phrases))

    def __len__(self) -> int:
        return sum(len(v) for v in self._phrases.values())

    def phrases(self, channel_key: str) -> tuple[str, ...]:

        return self._surface.get(channel_key, ())

    def bind(self, question: str) -> dict[str, list[str]]:

        toks = norm_tokens(question, do_stem=self.do_stem)
        out: dict[str, list[str]] = {}
        for key, phrases in self._phrases.items():
            hits = [
                self._surface[key][i]
                for i, p in enumerate(phrases)
                if _contains_run(toks, p)
            ]
            if hits:
                out[key] = hits
        return out


def load_lexicon(
    batch: str = "004",
    *,
    path: str | Path | None = None,
    node: str = "channel_lexicon",
    do_stem: bool = True,
) -> Lexicon:

    import yaml

    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "embench.yaml"
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    table = (doc.get(node) or {}).get(batch)
    if table is None:
        raise KeyError(f"no {node!r} entry for batch {batch!r} in {path}")
    return Lexicon(table, do_stem=do_stem)


_NAME = r"[A-ZÀ-Ý][\wÀ-ɏ'’-]*(?:\s+[A-ZÀ-Ý][\wÀ-ɏ'’-]*){0,2}"


_ASKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("paren", re.compile(r"\bI\s*\(\s*([^)]{2,40}?)\s*\)")),
    ("i_am", re.compile(rf"\bI\s+am\s+({_NAME})")),
    ("im", re.compile(rf"\bI['’]m\s+({_NAME})")),
    ("this_is", re.compile(rf"\bThis\s+is\s+({_NAME})")),
)


_ASKER_TAIL = re.compile(
    r"\s+(?:from|of|in|at|with|on|the|a|an|and|who|working|responsible)\b.*$", re.IGNORECASE
)

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")

_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_MONTH_DATE = re.compile(
    rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(20\d{{2}}))?\b", re.IGNORECASE
)
_MONTH_ONLY = re.compile(rf"\b({'|'.join(_MONTHS)})\b(?!\s+\d)", re.IGNORECASE)
_DURATION = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+(?:days|weeks|months)\b|\bnumber\s+of\s+days\b"
    r"|\bfrom\s+(?:initiation|start|kick[- ]?off)\s+to\b|\bstart\s+to\s+finish\b"
    r"|\bhow\s+much\s+time\b|\bduration\b|\btime\s+(?:elapsed|gap|interval)\b",
    re.IGNORECASE,
)


_EFFORT = re.compile(
    r"\bhow\s+many\s+(?:work(?:ing)?[- ]?days?|person[- ]?days?|man[- ]?days?"
    r"|business\s+days?)\b"
    r"|\b(?:work(?:ing)?[- ]?days?|person[- ]?days?|man[- ]?days?|business\s+days?)\b"
    r"|\bhow\s+much\s+effort\b|\beffort\s+(?:did|was|in\s+days)\b"
    r"|\bactually\s+take\b|\bcombined\s+(?:effort|time)\b|\btotal\s+effort\b",
    re.IGNORECASE,
)
_RELATIVE = re.compile(
    r"\b(?:before|after|since|until|prior\s+to|following|earlier|later|"
    r"next\s+(?:quarter|week|month)|last\s+(?:quarter|week|month)|"
    r"most\s+recent(?:ly)?|latest|currently|so\s+far)\b",
    re.IGNORECASE,
)


_CONNECTIVE = re.compile(
    r"(?:^|(?<=[,;.])\s*|\s+)"
    r"(how long (?:after|before)|"
    r"how long (?:did it take|was it|had|elapsed)|"
    r"how many (?:days|weeks|months) (?:did it take|passed|elapsed|until|before)|"
    r"after|before|once|following|since|prior to|until|then|subsequently)"
    r"\b",
    re.IGNORECASE,
)


_SUBORDINATORS = frozenset({"after", "before", "once", "following", "since",
                            "prior to", "then", "subsequently"})
_COMBINED = frozenset({"how long after", "how long before"})


def _is_interrogative(conn: str) -> bool:
    return conn.startswith("how ")


_ADVISORY = re.compile(
    r"\bhelp me\b|\bplease help\b|\bcould you help\b|\bhelp (?:me )?(?:draft|organi[sz]e|prepare)\b"
    r"|\bdraft (?:a|the|my|an)\b|\bhow should I\b|\bwhat should I\b|\bwhat aspects\b"
    r"|\bfrom what angles\b|\bprepare (?:my|a|the)\b|\bsuggestions should I\b"
    r"|\borgani[sz]e my thoughts\b|\bwhat (?:points|angles)\b|\bmy remarks\b",
    re.IGNORECASE,
)

_SKILL = re.compile(
    r"\b(?:skills?|expertise|strengths?|capabilit(?:y|ies)|experience|proficien(?:t|cy)|"
    r"good at|speciali[sz](?:e|es|ed|ation)|technical background|best suited)\b",
    re.IGNORECASE,
)

_MIN_CLAUSE_TOKENS = 3


@dataclass(frozen=True, slots=True)
class TimeExpr:


    kind: str
    text: str
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "text": self.text}
        if self.value:
            d["value"] = self.value
        return d


@dataclass(frozen=True, slots=True)
class Clause:


    text: str
    index: int
    connective: str | None = None
    channels: tuple[str, ...] = ()
    people: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "connective": self.connective,
            "channels": list(self.channels),
            "people": list(self.people),
        }


QUESTION_TYPES = frozenset({"simple", "temporal", "multi_hop", "preference", "aggregate"})
TEMPORAL_OPS = frozenset({"none", "duration", "before_after", "as_of", "audit"})
RETRIEVAL_VIEWS = ("channel", "actor", "artifact")

_CURRENT_STATE = re.compile(
    r"\b(?:current|currently|now|latest|in effect|in force|final version|so far)\b",
    re.IGNORECASE,
)
_AS_OF_DATE = re.compile(r"\bas\s+of\s+(\d{4}-\d{1,2}-\d{1,2})\b", re.IGNORECASE)


_AUDIT_INTENT = re.compile(
    r"\b(?:originally|previously|go back to|what changed|what was changed|"
    r"as it was|before the revision|prior version|superseded|audit|"
    r"what was the (?:old|original|previous|prior))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class QuerySpec:


    question: str
    asker: str | None = None
    people: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    time_expressions: tuple[TimeExpr, ...] = ()
    target_depth: int = 0
    is_multi_partition: bool = False
    clauses: tuple[Clause, ...] = ()

    variants: tuple[str, ...] = ()
    question_type: str = "simple"
    scope: dict[str, bool] = field(default_factory=lambda: {
        "channel": True, "actor": True, "artifact": True,
    })

    budgets: dict[str, int] = field(default_factory=dict)
    sub_queries: dict[str, str] = field(default_factory=dict)
    temporal_op: str = "none"
    as_of_date: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def query_for(self, view: str) -> str:

        return str(self.sub_queries.get(view) or self.question)

    def in_scope(self, view: str) -> bool:
        return bool(self.scope.get(view, True))

    @property
    def bound(self) -> bool:

        return bool(self.channels or self.people or self.asker)

    @property
    def has_duration_question(self) -> bool:
        return any(t.kind == "duration" for t in self.time_expressions)

    @property
    def has_skill_question(self) -> bool:

        return bool(self.people and _SKILL.search(self.question))

    @property
    def has_effort_question(self) -> bool:

        return any(t.kind == "effort" for t in self.time_expressions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asker": self.asker,
            "people": list(self.people),
            "channels": list(self.channels),
            "artifacts": list(self.artifacts),
            "time_expressions": [t.to_dict() for t in self.time_expressions],
            "skill_question": self.has_skill_question,
            "target_depth": self.target_depth,
            "is_multi_partition": self.is_multi_partition,
            "clauses": [c.to_dict() for c in self.clauses],
            "variants": list(self.variants),
            "question_type": self.question_type,
            "scope": dict(self.scope),
            "budgets": dict(self.budgets),
            "sub_queries": dict(self.sub_queries),
            "temporal_op": self.temporal_op,
            "as_of_date": self.as_of_date,
            "evidence": self.evidence,
        }


class QueryClassifier(Protocol):


    def classify(self, question: str, *, channels: Sequence[str],
                 people: Sequence[str]) -> Mapping[str, Sequence[str]]: ...


class RegisterExpander(Protocol):


    def expand(self, question: str, spec: "QuerySpec") -> Sequence[str]: ...


class QueryAnalyser:


    def __init__(
        self,
        *,
        speakers: Registry,
        lexicon: Lexicon | None = None,
        classifier: QueryClassifier | None = None,
        expander: RegisterExpander | None = None,
    ):
        self.speakers = speakers
        self.lexicon = lexicon
        self.classifier = classifier
        self.expander = expander


        self._by_fold: dict[str, str] = {}
        self._by_collapsed: dict[str, str] = {}
        self._by_tokens: dict[frozenset[str], str] = {}
        self._name_patterns: list[tuple[str, re.Pattern[str]]] = []
        for key in speakers.keys:
            f = fold(speakers.display(key))
            self._by_fold.setdefault(f, key)
            self._by_collapsed.setdefault(f.replace(" ", ""), key)
            toks = frozenset(f.split())
            if len(toks) > 1:
                self._by_tokens.setdefault(toks, key)
            self._name_patterns.append((key, re.compile(rf"\b{re.escape(f)}\b")))


    def resolve_name(self, surface: str) -> str | None:

        f = fold(surface).strip(" .,'’")
        if not f:
            return None
        return (
            self._by_fold.get(f)
            or self._by_collapsed.get(f.replace(" ", ""))
            or self._by_tokens.get(frozenset(f.split()))
        )

    def find_asker(self, question: str) -> tuple[str | None, str | None, str | None]:

        for rule, pat in _ASKER_PATTERNS:
            m = pat.search(question)
            if not m:
                continue
            surface = m.group(1).strip()
            if rule != "paren":
                surface = _ASKER_TAIL.sub("", surface).strip()
            return self.resolve_name(surface), surface, rule
        return None, None, None

    def find_people(self, text: str) -> tuple[str, ...]:
        folded = fold(text)
        return tuple(key for key, pat in self._name_patterns if pat.search(folded))

    def find_channels(self, text: str) -> tuple[tuple[str, ...], dict[str, list[str]]]:
        if self.lexicon is None:
            return (), {}
        hits = self.lexicon.bind(text)


        ranked = sorted(
            hits,
            key=lambda k: (-len(hits[k]), -max(len(p) for p in hits[k]), k),
        )
        return tuple(ranked), hits


    def find_time(self, question: str) -> tuple[TimeExpr, ...]:
        out: list[TimeExpr] = []
        seen: set[tuple[str, str]] = set()

        def push(kind: str, text: str, value: str | None = None) -> None:
            k = (kind, text.lower())
            if k not in seen:
                seen.add(k)
                out.append(TimeExpr(kind=kind, text=text, value=value))

        for m in _ISO_DATE.finditer(question):
            y, mo, d = m.groups()
            push("date", m.group(0), f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
        for m in _MONTH_DATE.finditer(question):
            month, day, year = m.groups()
            value = (
                f"{int(year):04d}-{_MONTHS.index(month.lower()) + 1:02d}-{int(day):02d}"
                if year else None
            )
            push("date", m.group(0), value)
        for m in _MONTH_ONLY.finditer(question):
            push("month", m.group(0))
        for m in _DURATION.finditer(question):
            push("duration", m.group(0))
        for m in _EFFORT.finditer(question):
            push("effort", m.group(0))
        for m in _RELATIVE.finditer(question):
            push("relative", m.group(0))
        return tuple(out)

    def connectives(self, question: str) -> tuple[str, ...]:

        return tuple(m.group(1).lower() for m in _CONNECTIVE.finditer(question))

    def split_clauses(self, question: str) -> tuple[Clause, ...]:

        bounds: list[tuple[int, int, str]] = []
        for m in _CONNECTIVE.finditer(question):
            bounds.append((m.start(1), m.end(1), m.group(1)))
        if not bounds:
            return ()

        pieces: list[tuple[str, str | None]] = []
        head = question[: bounds[0][0]].strip(" ,;.")
        if len(head.split()) >= _MIN_CLAUSE_TOKENS:
            pieces.append((head, None))
        for i, (_s, e, conn) in enumerate(bounds):
            stop = bounds[i + 1][0] if i + 1 < len(bounds) else len(question)
            body = question[e:stop].strip(" ,;.")
            if len(body.split()) >= _MIN_CLAUSE_TOKENS:
                pieces.append((body, conn.lower()))

        clauses: list[Clause] = []
        for i, (body, conn) in enumerate(pieces):
            ch, _ = self.find_channels(body)
            clauses.append(
                Clause(text=body, index=i, connective=conn,
                       channels=ch, people=self.find_people(body))
            )
        return tuple(clauses)


    @staticmethod
    def _two_event_shape(connectives: Sequence[str]) -> bool:

        sub_at = -1
        for i, conn in enumerate(connectives):
            if conn in _COMBINED:
                return True
            if conn in _SUBORDINATORS:
                if sub_at < 0:
                    sub_at = i
            elif _is_interrogative(conn) and 0 <= sub_at < i:
                return True
        return False

    def analyse(self, question: str) -> QuerySpec:
        asker_key, asker_surface, asker_rule = self.find_asker(question)
        people = self.find_people(question)
        channels, channel_hits = self.find_channels(question)
        times = self.find_time(question)
        connectives = self.connectives(question)
        clauses = self.split_clauses(question)

        evidence: dict[str, Any] = {}
        if asker_surface is not None:
            evidence["asker_rule"] = asker_rule
            evidence["asker_surface"] = asker_surface
            evidence["asker_resolved"] = asker_key is not None
        if channel_hits:
            evidence["channel_phrases"] = {k: v[:3] for k, v in channel_hits.items()}


        if self.classifier is not None and not (channels or people or asker_key):
            hint = self.classifier.classify(
                question,
                channels=self.lexicon.channels if self.lexicon else (),
                people=tuple(self.speakers.keys),
            )
            got_ch = tuple(c for c in hint.get("channels", ()) if isinstance(c, str))
            got_pp = tuple(p for p in hint.get("people", ()) if isinstance(p, str))
            channels = channels or got_ch
            people = people or got_pp
            evidence["llm_fallback"] = True
            evidence["llm_hint"] = {"channels": list(got_ch), "people": list(got_pp)}
        elif self.classifier is not None:
            evidence["llm_fallback"] = False


        multi = len(channels) >= 2 or (
            any(t.kind == "duration" for t in times) and self._two_event_shape(connectives)
        )
        if connectives:
            evidence["connectives"] = list(connectives)


        if _ADVISORY.search(question):
            depth = 2
        elif multi or any(t.kind == "duration" for t in times):
            depth = 1
        else:
            depth = 0

        temporal_op = "none"
        as_of_date: float | None = None
        if any(t.kind in {"duration", "effort"} for t in times):
            temporal_op = "before_after" if self._two_event_shape(connectives) else "duration"
        explicit_as_of = _AS_OF_DATE.search(question)
        if explicit_as_of:
            as_of_date = datetime.strptime(explicit_as_of.group(1), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp()
            temporal_op = "as_of"
            evidence["as_of_rule"] = "explicit_date"
        elif _AUDIT_INTENT.search(question):
            temporal_op = "audit"
            evidence["as_of_rule"] = "audit_intent"
        elif _CURRENT_STATE.search(question):
            temporal_op = "as_of"
            evidence["as_of_rule"] = "current"
        if multi:
            question_type = "multi_hop"
        elif temporal_op != "none" or times:
            question_type = "temporal"
        elif _ADVISORY.search(question):
            question_type = "preference"
        else:
            question_type = "simple"

        variants: tuple[str, ...] = (question,)
        if self.expander is not None:
            extra = tuple(v for v in self.expander.expand(question, QuerySpec(question=question))
                          if v and v != question)
            variants = variants + extra
            evidence["n_variants"] = len(variants)

        return QuerySpec(
            question=question,
            asker=asker_key,
            people=people,
            channels=channels,
            artifacts=_tags.artifacts(question),
            time_expressions=times,
            target_depth=depth,
            is_multi_partition=multi,
            clauses=clauses,
            variants=variants,
            question_type=question_type,
            scope={view: True for view in RETRIEVAL_VIEWS},
            sub_queries={view: question for view in RETRIEVAL_VIEWS},
            temporal_op=temporal_op,
            as_of_date=as_of_date,
            evidence=evidence,
        )

    def analyse_many(self, questions: Iterable[str]) -> list[QuerySpec]:
        return [self.analyse(q) for q in questions]


def build(
    *,
    speakers: Registry | str | Path,
    batch: str = "004",
    lexicon: Lexicon | str | Path | None = None,
    do_stem: bool = True,
    classifier: QueryClassifier | None = None,
    expander: RegisterExpander | None = None,
) -> QueryAnalyser:

    if not isinstance(speakers, Registry):
        speakers = Registry.load(speakers)
    if lexicon is None:
        lex = load_lexicon(batch, do_stem=do_stem)
    elif isinstance(lexicon, Lexicon):
        lex = lexicon
    else:
        lex = load_lexicon(batch, path=lexicon, do_stem=do_stem)
    return QueryAnalyser(speakers=speakers, lexicon=lex,
                         classifier=classifier, expander=expander)
