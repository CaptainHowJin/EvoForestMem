from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..log.message import Message
from ..log.store import MessageLog
from ..retrieve.s0_analyse import norm_tokens
from ._persist import atomic_write as _atomic_write, advance_corpus_digest as _advance_corpus_digest, corpus_fingerprint as _corpus_fingerprint


_OBLIGATION = re.compile(
    r"\b(?:must|must\s+not|shall|shall\s+not|may\s+not|is\s+required\s+to|"
    r"are\s+required\s+to|is\s+mandatory|are\s+mandatory|is\s+forbidden|"
    r"is\s+prohibited|is\s+not\s+allowed|are\s+not\s+allowed|should\s+always|"
    r"should\s+never|is\s+not\s+permitted)\b",
    re.IGNORECASE,
)


_DEFINITION = re.compile(
    r"\b(?:is\s+defined\s+as|are\s+defined\s+as|means\s+that|refers\s+to|"
    r"stands\s+for|the\s+default\s+is|defaults\s+to|is\s+set\s+to|"
    r"the\s+value\s+of\s+\w+\s+is|we\s+define)\b",
    re.IGNORECASE,
)


_CONDITIONAL = re.compile(
    r"\b(?:if|when|whenever|in\s+the\s+case\s+(?:that|where)|should)\b[^.!?]{0,120}"
    r"\b(?:then|must|should|shall|the\s+system\s+(?:should|must|will)|we\s+(?:should|must))\b",
    re.IGNORECASE,
)


_OWNERSHIP = re.compile(
    r"\b(?:is|are|will\s+be|remains?)\s+responsible\s+for\b"
    r"|\b(?:owns|owner\s+of|accountable\s+for|takes?\s+ownership\s+of)\b"
    r"|\bassigned\s+to\b|\bhanded\s+over\s+to\b",
    re.IGNORECASE,
)


_REVOKES = re.compile(
    r"\b(?:supersed\w+|overrides?|replaces?\s+the|no\s+longer\s+(?:applies|valid)|"
    r"instead\s+of\s+the\s+previous|revised\s+to|updated\s+to|changed\s+to|"
    r"from\s+now\s+on|going\s+forward)\b",
    re.IGNORECASE,
)

_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("definition", _DEFINITION),
    ("obligation", _OBLIGATION),
    ("conditional", _CONDITIONAL),
    ("ownership", _OWNERSHIP),
)


_IDENT = re.compile(r"\b(?:[A-Z][A-Z0-9]{1,}(?:[_-][A-Z0-9]+)+|[A-Z]{2,}-\d+|`[^`\n]{2,40}`)")

_STOP = frozenset(norm_tokens(
    "the and for with this that our your their have has had will would could should "
    "from into about over under been being does did done here there what when where "
    "must shall may not need system platform module task team project group everyone"
))


def _key_terms(text: str) -> frozenset[str]:
    return frozenset(w for w in norm_tokens(text) if len(w) > 2 and w not in _STOP)


NORM_SCHEMA_VERSION = "evoforestmem.norm-index.v1"
LEGACY_NORM_POLICY_VERSION = "arrival-idf-single-predecessor-v1"
NORM_POLICY_VERSION = "ledger-authority-discovery-v2"


def norm_corpus_fingerprint(messages: Iterable[Message]) -> str:

    return _corpus_fingerprint(messages)


@dataclass(frozen=True, slots=True)
class Norm:


    msg_id: str
    kind: str
    date: str
    ts: float
    speaker_key: str
    channel_key: str
    subjects: tuple[str, ...]
    markers: tuple[str, ...]
    quote: str
    terms: frozenset[str] = frozenset()

    supersedes: str | None = None
    superseded_by: str | None = None
    supersedes_by_subject: tuple[tuple[str, str], ...] = ()
    superseded_by_subject: tuple[tuple[str, str], ...] = ()

    @property
    def active(self) -> bool:
        if self.subjects and self.superseded_by_subject:
            expired = {subject for subject, _ in self.superseded_by_subject}
            return any(subject not in expired for subject in self.subjects)
        return self.superseded_by is None

    def active_for(self, subject: str) -> bool:
        return subject not in dict(self.superseded_by_subject)

    def to_dict(self) -> dict[str, Any]:
        return {
            "msg_id": self.msg_id, "kind": self.kind, "date": self.date,
            "speaker": self.speaker_key, "channel": self.channel_key,
            "subjects": list(self.subjects), "markers": list(self.markers),
            "quote": self.quote, "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "supersedes_by_subject": dict(self.supersedes_by_subject),
            "superseded_by_subject": dict(self.superseded_by_subject),
            "active": self.active,
        }


@dataclass
class NormStats:
    messages: int = 0
    norms: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    with_subject: int = 0
    supersession_links: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages, "norms": self.norms,
                "by_kind": dict(sorted(self.by_kind.items())),
                "with_subject": self.with_subject,
                "supersession_links": self.supersession_links}


@dataclass(frozen=True, slots=True)
class NormConfig:


    supersede_overlap: float = 0.5


    max_results: int = 3
    quote_chars: int = 220


    legacy_replacements: bool = False


DEFAULT_NORM_CONFIG = NormConfig()


def _subjects(message: Message) -> tuple[str, ...]:
    found = [value.strip().lower() for value in message.tags.artifacts if 2 < len(value) < 80]
    found += [value.strip("`").lower() for value in _IDENT.findall(message.text)]
    return tuple(dict.fromkeys(found))[:6]


def _quote(message: Message, pattern: re.Pattern[str], quote_chars: int) -> str:
    hit = pattern.search(message.text)
    if hit is None:
        return " ".join(message.text.split())[:quote_chars]
    lo = max(0, hit.start() - 80)
    hi = min(len(message.text), hit.end() + quote_chars - 80)
    excerpt = " ".join(message.text[lo:hi].split())
    return ("…" if lo else "") + excerpt + ("…" if hi < len(message.text) else "")


def extract_norm(
    message: Message, *, config: NormConfig = DEFAULT_NORM_CONFIG,
) -> Norm | None:

    match = next(((name, pattern) for name, pattern in _KINDS if pattern.search(message.text)), None)
    if match is None:
        return None
    kind, pattern = match
    return Norm(
        msg_id=message.msg_id,
        kind=kind,
        date=message.date,
        ts=message.ts,
        speaker_key=message.speaker_key,
        channel_key=message.channel_key,
        subjects=_subjects(message),
        markers=tuple(message.tags.markers),
        quote=_quote(message, pattern, config.quote_chars),
        terms=_key_terms(message.text),
    )


class NormIndex:


    def __init__(
        self,
        log: MessageLog | None = None,
        *,
        config: NormConfig = DEFAULT_NORM_CONFIG,
        build: bool = True,
    ):
        self.log = log
        self.config = config
        self.norms: dict[str, Norm] = {}
        self.by_subject: dict[str, list[str]] = defaultdict(list)
        self.stats = NormStats()
        self._df: Counter[str] = Counter()
        self._n_docs = 0
        self._idf_cache: dict[str, float] | None = None
        self._seen_msg_ids: set[str] = set()
        self._latest_by_subject: dict[str, str] = {}
        self._corpus_digest = bytes(32)
        self._link_decisions: list[dict[str, Any]] = []
        if build and log is not None:
            self._build_batch(log)


    @classmethod
    def incremental(
        cls, *, config: NormConfig = DEFAULT_NORM_CONFIG,
    ) -> "NormIndex":
        return cls(None, config=config, build=False)

    @property
    def corpus_fingerprint(self) -> str:
        return self._corpus_digest.hex()

    @property
    def idf(self) -> dict[str, float]:
        if self._idf_cache is None:
            n = max(1, self._n_docs)
            self._idf_cache = {
                word: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
                for word, count in self._df.items()
            }
        return self._idf_cache

    def _observe_document(self, message: Message) -> bool:
        if message.msg_id in self._seen_msg_ids:
            return False
        self._seen_msg_ids.add(message.msg_id)
        self._n_docs += 1
        self._df.update(_key_terms(message.text))
        self._idf_cache = None
        self._corpus_digest = _advance_corpus_digest(self._corpus_digest, message)
        self.stats.messages = self._n_docs
        return True

    def _register_norm(self, norm: Norm) -> None:
        self.norms[norm.msg_id] = norm
        for subject in norm.subjects:
            self.by_subject[subject].append(norm.msg_id)
        self.stats.by_kind[norm.kind] = self.stats.by_kind.get(norm.kind, 0) + 1
        if norm.subjects:
            self.stats.with_subject += 1
        self.stats.norms = len(self.norms)

    def _link_one(self, norm_id: str, *, message_text: str) -> None:
        norm = self.norms[norm_id]
        revokes = bool(_REVOKES.search(message_text)) or "supersede" in norm.markers
        chosen_predecessor: str | None = None
        supersedes_by_subject = dict(norm.supersedes_by_subject)
        for subject in norm.subjects:
            previous_id = self._latest_by_subject.get(subject)
            if previous_id and previous_id != norm.msg_id:
                previous = self.norms[previous_id]
                weight = sum(self.idf.get(word, 0.0) for word in previous.terms) or 1.0
                shared = sum(
                    self.idf.get(word, 0.0) for word in previous.terms & norm.terms
                )
                ratio = shared / weight
                previous_subject_links = dict(previous.superseded_by_subject)
                eligible = subject not in previous_subject_links
                linked = eligible and (revokes or ratio >= self.config.supersede_overlap)
                self._link_decisions.append({
                    "previous": previous_id,
                    "current": norm.msg_id,
                    "subject": subject,
                    "ratio": ratio,
                    "revokes": revokes,
                    "linked": linked,
                })
                if linked:
                    if chosen_predecessor is None:
                        chosen_predecessor = previous_id
                    previous_subject_links[subject] = norm.msg_id
                    supersedes_by_subject[subject] = previous_id
                    self.norms[previous_id] = replace(
                        previous,
                        superseded_by=norm.msg_id,
                        superseded_by_subject=tuple(sorted(previous_subject_links.items())),
                    )
            self._latest_by_subject[subject] = norm.msg_id
        if chosen_predecessor is not None:
            self.norms[norm.msg_id] = replace(
                self.norms[norm.msg_id],
                supersedes=chosen_predecessor,
                supersedes_by_subject=tuple(sorted(supersedes_by_subject.items())),
            )
        self.stats.supersession_links = sum(
            1 for item in self.norms.values() if item.superseded_by is not None
        )

    def _build_batch(self, log: MessageLog) -> None:
        for message in log:
            self._observe_document(message)
        extracted = [norm for message in log if (norm := extract_norm(message, config=self.config))]
        for norm in extracted:
            self._register_norm(norm)
        if self.config.legacy_replacements:
            for norm in sorted(extracted, key=lambda item: (item.ts, item.msg_id)):
                self._link_one(norm.msg_id, message_text=log.get(norm.msg_id).text)

    @classmethod
    def from_stream(
        cls,
        messages: Iterable[Message],
        *,
        config: NormConfig = DEFAULT_NORM_CONFIG,
    ) -> "NormIndex":

        index = cls.incremental(config=config)
        for message in messages:
            index.append(message)
        return index

    def append(self, message: Message) -> Norm | None:

        if not self._observe_document(message):
            return None
        norm = extract_norm(message, config=self.config)
        if norm is None:
            return None
        self._register_norm(norm)
        if self.config.legacy_replacements:
            self._link_one(norm.msg_id, message_text=message.text)
        return self.norms[norm.msg_id]

    def fragility(self, *, low: float = 0.45, high: float = 0.55) -> dict[str, Any]:
        candidates = [row for row in self._link_decisions if not row["revokes"]]
        near = [row for row in candidates if low <= row["ratio"] <= high]
        return {
            "comparisons": len(candidates),
            "near_threshold": len(near),
            "low": low,
            "high": high,
            "records": near,
        }


    def lookup(self, question: str, *, limit: int | None = None,
               as_of: float | None = None,
               available: Callable[[str], bool] | None = None) -> list[Norm]:

        limit = self.config.max_results if limit is None else limit
        qterms = _key_terms(question)
        lowered = question.lower()
        visible = (
            self.norms if as_of is None and available is None
            else {mid: n for mid, n in self.norms.items()
                  if (as_of is None or n.ts <= as_of) and (available is None or available(mid))}
        )
        scored: dict[str, float] = {}
        for subject, ids in self.by_subject.items():
            if subject not in lowered:
                continue
            weight = 2.0 + math.log(1 + len(subject.split()))
            for msg_id in ids:
                if msg_id in visible:
                    scored[msg_id] = max(scored.get(msg_id, 0.0), weight)
        for msg_id, norm in visible.items():
            shared = sum(self.idf.get(w, 0.0) for w in (qterms & norm.terms))
            if shared > 0.0:
                scored[msg_id] = scored.get(msg_id, 0.0) + shared / 100.0
        ranked = sorted(
            (self.norms[i] for i in scored),
            key=lambda x: (-scored[x.msg_id], -x.ts, x.msg_id),
        )
        out: list[Norm] = []
        for norm in ranked[:limit]:
            matched = tuple(subject for subject in norm.subjects if subject in lowered)
            if not matched:
                out.append(norm)
                continue
            out.append(replace(norm, subjects=matched))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"stats": self.stats.to_dict(),
                "config": {"supersede_overlap": self.config.supersede_overlap,
                           "legacy_replacements": self.config.legacy_replacements,
                           "max_results": self.config.max_results},
                "corpus_fingerprint": self.corpus_fingerprint}

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for norm in sorted(self.norms.values(), key=lambda item: item.msg_id):
            digest.update(json.dumps(
                self._norm_record(norm), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _norm_record(norm: Norm) -> dict[str, Any]:
        return {
            "msg_id": norm.msg_id,
            "kind": norm.kind,
            "date": norm.date,
            "ts": float(norm.ts),
            "speaker_key": norm.speaker_key,
            "channel_key": norm.channel_key,
            "subjects": list(norm.subjects),
            "markers": list(norm.markers),
            "quote": norm.quote,
            "terms": sorted(norm.terms),
            "supersedes": norm.supersedes,
            "superseded_by": norm.superseded_by,
            "supersedes_by_subject": dict(norm.supersedes_by_subject),
            "superseded_by_subject": dict(norm.superseded_by_subject),
        }

    def save(self, path: str | Path) -> Path:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(self._norm_record(norm), ensure_ascii=False, sort_keys=True) + "\n"
            for norm in sorted(self.norms.values(), key=lambda item: (item.ts, item.msg_id))
        )
        records_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        generation_path = path.with_name(f"{path.stem}.{records_digest[:16]}{path.suffix}")
        if not generation_path.is_file():
            _atomic_write(generation_path, payload)


        _atomic_write(path, payload)
        state = {
            "n_docs": self._n_docs,
            "df": dict(sorted(self._df.items())),
            "seen_msg_ids": sorted(self._seen_msg_ids),
            "latest_by_subject": dict(sorted(self._latest_by_subject.items())),
            "link_decisions": self._link_decisions,
            "config": {
                "legacy_replacements": self.config.legacy_replacements,
                "supersede_overlap": self.config.supersede_overlap,
                "max_results": self.config.max_results,
                "quote_chars": self.config.quote_chars,
            },
        }
        metadata = {
            "schema_version": NORM_SCHEMA_VERSION,
            "policy_version": NORM_POLICY_VERSION,
            "records_file": generation_path.name,
            "records_sha256": records_digest,
            "fingerprint": self.fingerprint(),
            "corpus_fingerprint": self.corpus_fingerprint,
            "state": state,
            "state_sha256": hashlib.sha256(json.dumps(
                state, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        _atomic_write(
            meta_path,
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expect_corpus: str | None = None,
    ) -> "NormIndex":
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != NORM_SCHEMA_VERSION:
            raise ValueError(f"unsupported norm index schema: {metadata.get('schema_version')}")
        if metadata.get("policy_version") not in (NORM_POLICY_VERSION, LEGACY_NORM_POLICY_VERSION):
            raise ValueError(f"unsupported norm policy: {metadata.get('policy_version')}")
        records_path = path.parent / str(metadata.get("records_file") or path.name)
        payload = records_path.read_text(encoding="utf-8")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != metadata.get("records_sha256"):
            raise ValueError("norm index record checksum mismatch")
        corpus = str(metadata.get("corpus_fingerprint") or "")
        if expect_corpus is not None and corpus != expect_corpus:
            raise ValueError("norm index corpus fingerprint mismatch")
        state = metadata.get("state") or {}
        state_digest = hashlib.sha256(json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if state_digest != metadata.get("state_sha256"):
            raise ValueError("norm index continuation-state checksum mismatch")
        raw_config = state.get("config") or {}
        index = cls.incremental(config=NormConfig(
            legacy_replacements=bool(raw_config.get("legacy_replacements",
                metadata.get("policy_version") == LEGACY_NORM_POLICY_VERSION)),
            supersede_overlap=float(raw_config.get("supersede_overlap", 0.5)),
            max_results=int(raw_config.get("max_results", 3)),
            quote_chars=int(raw_config.get("quote_chars", 220)),
        ))
        for line in payload.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            norm = Norm(
                msg_id=str(row["msg_id"]),
                kind=str(row["kind"]),
                date=str(row["date"]),
                ts=float(row["ts"]),
                speaker_key=str(row["speaker_key"]),
                channel_key=str(row["channel_key"]),
                subjects=tuple(str(value) for value in row.get("subjects", ())),
                markers=tuple(str(value) for value in row.get("markers", ())),
                quote=str(row.get("quote") or ""),
                terms=frozenset(str(value) for value in row.get("terms", ())),
                supersedes=row.get("supersedes"),
                superseded_by=row.get("superseded_by"),
                supersedes_by_subject=tuple(sorted(
                    (str(key), str(value))
                    for key, value in (row.get("supersedes_by_subject") or {}).items()
                )),
                superseded_by_subject=tuple(sorted(
                    (str(key), str(value))
                    for key, value in (row.get("superseded_by_subject") or {}).items()
                )),
            )
            index.norms[norm.msg_id] = norm
            for subject in norm.subjects:
                index.by_subject[subject].append(norm.msg_id)
        index._n_docs = int(state.get("n_docs", 0))
        index._df = Counter({str(k): int(v) for k, v in (state.get("df") or {}).items()})
        index._seen_msg_ids = {str(value) for value in state.get("seen_msg_ids", ())}
        index._latest_by_subject = {
            str(k): str(v) for k, v in (state.get("latest_by_subject") or {}).items()
        }
        index._link_decisions = list(state.get("link_decisions") or ())
        index._corpus_digest = bytes.fromhex(corpus)
        index._idf_cache = None
        if index._n_docs != len(index._seen_msg_ids):
            raise ValueError("norm index document-count state mismatch")
        if not set(index.norms) <= index._seen_msg_ids:
            raise ValueError("norm index records reference unseen messages")
        if not set(index._latest_by_subject.values()) <= set(index.norms):
            raise ValueError("norm index latest-subject state references missing norms")
        index.stats = NormStats(messages=index._n_docs)
        for norm in index.norms.values():
            index.stats.by_kind[norm.kind] = index.stats.by_kind.get(norm.kind, 0) + 1
            index.stats.with_subject += int(bool(norm.subjects))
        index.stats.norms = len(index.norms)
        index.stats.supersession_links = sum(
            norm.superseded_by is not None for norm in index.norms.values()
        )
        if index.fingerprint() != metadata.get("fingerprint"):
            raise ValueError("norm index semantic fingerprint mismatch")
        return index


@dataclass(frozen=True, slots=True)
class NormBlock:


    norms: tuple[Norm, ...]

    statuses: dict[str, str] = field(default_factory=dict)

    @property
    def msg_ids(self) -> tuple[str, ...]:
        return tuple(n.msg_id for n in self.norms)

    def render(self) -> list[str]:
        lines: list[str] = []
        for n in self.norms:
            status = self.statuses.get(n.msg_id, "version unverified")
            subjects = ", ".join(n.subjects[:3]) or "unscoped"
            lines.append(
                f"    (normative {n.kind} asserted {n.date} by {n.speaker_key} in "
                f"{n.channel_key}, governing {subjects}; {status})"
            )
        return lines


def build(log: MessageLog, *, config: NormConfig = DEFAULT_NORM_CONFIG) -> NormIndex:
    return NormIndex(log, config=config)
