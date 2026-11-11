from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..forest.episodes import Episode, EpisodeTable, Interval, interval_between
from ..log.store import MessageLog
from .s0_analyse import QuerySpec, norm_tokens


_WORKING = re.compile(
    r"\b(?:work(?:ing)?[- ]?days?|person[- ]?days?|man[- ]?days?|business\s+days?"
    r"|work\s+hours|effort)\b",
    re.IGNORECASE,
)


_GAP = re.compile(
    r"\bhow\s+long\s+(?:after|before)\b"
    r"|\bafter\b[^.?!]{0,200}?\bhow\s+long\b"
    r"|\bhow\s+long\s+did\s+it\s+take\s+(?:before|until|for)\b"
    r"|\bhow\s+(?:long|much\s+time)\b[^.?!]{0,120}\b(?:before|until)\b"
    r"|\b(?:gap|interval)\s+between\b"
    r"|\bhow\s+many\s+days\s+(?:passed|elapsed)\b",
    re.IGNORECASE,
)


_SPAN = re.compile(
    r"\bfrom\s+(?:initiation|start|kick[- ]?off|inception|beginning)\b"
    r"|\bstart\s+to\s+finish\b|\bfrom\s+start\s+to\b"
    r"|\bhow\s+(?:many\s+days|long)\b[^.?!]{0,120}\b(?:last|lasted|take|took|span|spanned)\b"
    r"|\bin\s+total\b|\bhow\s+long\s+did\s+(?:the|this|it)\b",
    re.IGNORECASE,
)


_SUMMED = re.compile(
    r"\b(?:two|both)\s+tasks\b|\bthese\s+two\b|\bthe\s+following\s+two\b"
    r"|\bcombined\b|\bcollectively\b|\baltogether\b|\bin\s+total\b"
    r"|\btotal\s+(?:actual\s+)?(?:number\s+of\s+)?"
    r"(?:work\s+hours|work(?:ing)?[- ]?days?|person[- ]?days?|man[- ]?days?|days)\b",
    re.IGNORECASE,
)


_ASKS_DURATION = re.compile(
    r"\bhow\s+long\b"
    r"|\bhow\s+many\b[^.?!]{0,30}?\b(?:days|weeks|months)\b"
    r"|\bhow\s+much\s+(?:time|effort)\b"
    r"|\b(?:work(?:ing)?[- ]?days?|person[- ]?days?|man[- ]?days?|business\s+days?"
    r"|work\s+hours)\b"
    r"|\bnumber\s+of\s+days\b|\bduration\b|\btime\s+(?:elapsed|gap|interval)\b",
    re.IGNORECASE,
)

Quantity = str
Convention = str
Shape = str


@dataclass(frozen=True, slots=True)
class QuestionShape:
    shape: Shape
    quantity: Quantity
    convention: Convention
    evidence: tuple[str, ...] = ()

    @property
    def fires(self) -> bool:
        return self.shape != "none"

    def to_dict(self) -> dict[str, Any]:
        return {"shape": self.shape, "quantity": self.quantity,
                "convention": self.convention, "evidence": list(self.evidence)}


def classify(question: str, temporal_op: str = "none") -> QuestionShape:

    planner_fires = temporal_op in {"duration", "before_after"}
    if not _ASKS_DURATION.search(question) and not planner_fires:
        return QuestionShape("none", "calendar", "inclusive")
    evidence: list[str] = []
    quantity: Quantity = "calendar"
    if _WORKING.search(question):
        quantity = "working"
        evidence.append("effort_vocabulary")
    if temporal_op == "before_after" or _GAP.search(question):
        evidence.append("planner_before_after" if temporal_op == "before_after"
                        else "gap_between_two_events")
        return QuestionShape("gap", quantity, "exclusive", tuple(evidence))
    if _SUMMED.search(question):
        evidence.append("two_tasks_summed")
        return QuestionShape("sum", quantity, "inclusive", tuple(evidence))
    if _SPAN.search(question):
        evidence.append("extent_of_one_span")
        return QuestionShape("span", quantity, "inclusive", tuple(evidence))


    evidence.append("duration_default")
    return QuestionShape("span", quantity, "inclusive", tuple(evidence))


@dataclass(frozen=True, slots=True)
class DurationFact:


    shape: QuestionShape
    episodes: tuple[Episode, ...]
    value: int
    components: tuple[int, ...] = ()
    confidence: float = 0.0
    notes: tuple[str, ...] = ()

    alternatives: tuple[Episode, ...] = ()

    asserted: bool = True

    @property
    def msg_ids(self) -> tuple[str, ...]:
        return tuple(mid for e in (*self.episodes, *self.alternatives)
                     for mid in e.msg_ids)

    @property
    def unit(self) -> str:
        return ("working days" if self.shape.quantity == "working"
                else f"calendar days ({self.shape.convention})")

    @staticmethod
    def _describe(e: Episode) -> str:
        return (f'"{e.name}" [{e.actor} / {e.channel}] '
                f'{e.start.date} -> {e.end.date} '
                f'(inclusive {e.interval.calendar_days_inclusive}, '
                f'exclusive {e.interval.calendar_days_exclusive}, '
                f'working {e.interval.working_days})')

    def _value_of(self, e: Episode) -> int:
        return e.interval.value(self.shape.quantity, self.shape.convention)

    def render(self) -> list[str]:

        body = " + ".join(self._describe(e) for e in self.episodes)
        component = ""
        if len(self.components) > 1:
            component = " = " + " + ".join(str(c) for c in self.components)
        if self.asserted:
            return [f"    (derived interval - {body}; "
                    f"answer in {self.unit}{component}: {self.value})"]
        if len(self.episodes) > 1:
            lines = [f"    (derived interval - PROVISIONAL sum over "
                     f"{len(self.episodes)} task spans; each boundary pair is the best "
                     f"match, not a certain one. Requested unit: {self.unit}.)"]
            for i, e in enumerate(self.episodes):
                lines.append(f"    (component {i + 1} - {self._describe(e)}; "
                             f"{self.unit}: {self._value_of(e)})")
            lines.append(f"    (provisional total in {self.unit}{component}: {self.value})")
            return lines
        shortlist = (*self.episodes, *self.alternatives)
        lines = [f"    (derived interval - AMBIGUOUS: the question's task matches "
                 f"{len(shortlist)} candidate span(s) that do not agree; ranked, best "
                 f"first. Requested unit: {self.unit}.)"]
        for i, e in enumerate(shortlist):
            lines.append(f"    (candidate {i + 1} - {self._describe(e)}; "
                         f"{self.unit}: {self._value_of(e)})")
        return lines

    def quotes(self) -> list[str]:
        out: list[str] = []
        for e in self.episodes:
            out.append(f"    (boundary start {e.start.date} {e.start.speaker_key}: "
                       f"{e.start.quote})")
            out.append(f"    (boundary end {e.end.date} {e.end.speaker_key}: "
                       f"{e.end.quote})")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.to_dict(),
            "value": self.value,
            "asserted": self.asserted,
            "components": list(self.components),
            "confidence": round(self.confidence, 4),
            "episodes": [e.to_dict() for e in self.episodes],
            "alternatives": [e.to_dict() for e in self.alternatives],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ResolverConfig:


    k_starts: int = 6


    w_end_overlap: float = 1.0

    w_end_bm25: float = 0.5


    w_pair_prior: float = 0.25


    owner_is_filter: bool = True

    min_confidence: float = 0.0


    agreement_window: int = 3


    assert_confidence_tolerant: float = 0.0
    assert_confidence_exact: float = 1.0


    assert_confidence_gap: float = 1.0


    max_alternatives: int = 2


    selector_k: int = 12

    def assert_floor(self, quantity: Quantity, shape: Shape = "span") -> float:
        if shape == "gap":
            return self.assert_confidence_gap
        return (self.assert_confidence_exact if quantity == "working"
                else self.assert_confidence_tolerant)


DEFAULT_RESOLVER_CONFIG = ResolverConfig()


class DurationResolver:


    def __init__(self, *, log: MessageLog, table: EpisodeTable, scores: Any,
                 config: ResolverConfig = DEFAULT_RESOLVER_CONFIG,
                 selector: Any = None):
        self.log = log
        self.table = table
        self.scores = scores
        self.config = config


        self.selector = selector


    def resolve(self, spec: QuerySpec) -> DurationFact | None:
        shape = classify(spec.question, spec.temporal_op)
        if not shape.fires:
            return None
        if shape.shape == "sum":
            return self._resolve_sum(spec, shape)
        if shape.shape == "span":
            return self._resolve_span(spec, shape)
        return None


    def _candidates(self, question: str, people: Sequence[str]) -> list[tuple[float, Episode]]:
        cfg = self.config
        table = self.table
        sc = self.scores.scores(question)
        qt = table.score_terms(question)
        owners = set(people)

        ranked: list[tuple[float, Any]] = []
        for start in table.starts:
            base = sc.get(start.msg_id, 0.0)
            if base <= 0.0:
                continue
            if owners and cfg.owner_is_filter and start.speaker_key not in owners:
                continue
            ranked.append((base, start))
        if not ranked and owners and cfg.owner_is_filter:


            ranked = [(sc.get(s.msg_id, 0.0), s) for s in table.starts
                      if sc.get(s.msg_id, 0.0) > 0.0]
        ranked.sort(key=lambda r: (-r[0], r[1].seq))

        out: list[tuple[float, Episode]] = []
        for base, start in ranked[: cfg.k_starts]:
            start_terms = table.terms_of(start.msg_id)
            identity = (qt & start_terms) or start_terms
            for episode in self.table.by_start.get(start.msg_id, ()):
                end_terms = table.terms_of(episode.end.msg_id)
                score = (
                    base
                    + cfg.w_end_overlap * table.overlap(identity, end_terms)
                    + cfg.w_end_bm25 * sc.get(episode.end.msg_id, 0.0)
                    + cfg.w_pair_prior * episode.pair_score
                )
                out.append((score, episode))
        out.sort(key=lambda r: (-r[0], r[1].start.ts, r[1].end.ts))
        return out

    def _confidence(self, ranked: Sequence[tuple[float, Episode]],
                    shape: QuestionShape) -> float:

        if not ranked:
            return 0.0
        window = self._distinct(ranked, self.config.agreement_window)
        if not window:
            return 0.0
        values = [e.interval.value(shape.quantity, shape.convention) for e in window]
        top = values[0]
        if shape.quantity == "working":
            agree = sum(1 for v in values if v == top)
        else:

            agree = sum(1 for v in values if abs(v - top) <= 1)
        return agree / len(values)

    def _distinct(self, ranked: Sequence[tuple[float, Episode]],
                  limit: int) -> list[Episode]:

        seen: set[tuple[str, str]] = set()
        out: list[Episode] = []
        for _, e in ranked:
            key = (e.start.date, e.end.date)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
            if len(out) >= limit:
                break
        return out

    def _select(self, question: str, ranked: Sequence[tuple[float, Episode]],
                shape: QuestionShape, site: str) -> tuple[Episode, float, tuple[str, ...]]:

        confidence = self._confidence(ranked, shape)
        if self.selector is None or len(ranked) < 2:
            return ranked[0][1], confidence, ()
        shortlist = self._distinct(ranked, self.config.selector_k)
        if len(shortlist) < 2:
            return ranked[0][1], confidence, ()
        chosen = self.selector(question, shortlist, site)
        if chosen is None or not 0 <= int(chosen) < len(shortlist):
            return ranked[0][1], confidence, ("escalation abstained",)


        return shortlist[int(chosen)], 1.0, ("boundary pair selected by escalation",)

    def _resolve_span(self, spec: QuerySpec, shape: QuestionShape) -> DurationFact | None:
        ranked = self._candidates(spec.question, spec.people)
        if not ranked:
            return None
        episode, confidence, note = self._select(spec.question, ranked, shape, "w1_span")
        if confidence < self.config.min_confidence:
            return None
        value = episode.interval.value(shape.quantity, shape.convention)
        asserted = confidence >= self.config.assert_floor(shape.quantity, shape.shape)
        alternatives = ()
        if not asserted:
            alternatives = tuple(
                e for e in self._distinct(ranked, 1 + self.config.max_alternatives)
                if (e.start.date, e.end.date) != (episode.start.date, episode.end.date)
            )[: self.config.max_alternatives]
        return DurationFact(shape=shape, episodes=(episode,), value=value,
                            components=(value,), confidence=confidence,
                            alternatives=alternatives, asserted=asserted,
                            notes=(f"{len(ranked)} candidate episodes",) + note)

    def _resolve_sum(self, spec: QuerySpec, shape: QuestionShape) -> DurationFact | None:

        people = list(dict.fromkeys(spec.people))
        if len(people) < 2:
            fact = self._resolve_span(spec, shape)
            if fact is None:
                return None


            return DurationFact(shape=shape, episodes=fact.episodes, value=fact.value,
                                components=fact.components, confidence=fact.confidence,
                                alternatives=fact.alternatives, asserted=False,
                                notes=fact.notes + ("sum requested but one owner bound; "
                                                    "rendered provisional, not asserted",))
        episodes: list[Episode] = []
        components: list[int] = []
        confidences: list[float] = []
        notes: tuple[str, ...] = ("summed over two named owners",)
        for person in people[:2]:
            ranked = self._candidates(spec.question, [person])
            if not ranked:
                return None
            episode, confidence, note = self._select(spec.question, ranked, shape,
                                                     "w1_sum")
            episodes.append(episode)
            components.append(episode.interval.value(shape.quantity, shape.convention))
            confidences.append(confidence)
            notes = notes + note


        confidence = confidences[0] * confidences[1]
        if confidence < self.config.min_confidence:
            return None


        return DurationFact(shape=shape, episodes=tuple(episodes), value=sum(components),
                            components=tuple(components), confidence=confidence,
                            asserted=False,
                            notes=notes + ("two-task sum below the constant baseline; "
                                           "rendered provisional, not asserted",))


    def from_pair(self, spec: QuerySpec, a_msg_id: str, b_msg_id: str, *,
                  ranked_pairs: Sequence[Any] = (), selected: bool = False) -> DurationFact | None:

        shape = classify(spec.question, spec.temporal_op)
        if shape.shape != "gap":
            return None
        a, b = self.log.get(a_msg_id), self.log.get(b_msg_id)
        if b.ts < a.ts:
            a, b = b, a
        interval = interval_between(a.date, b.date, self.table.corpus_days)
        episode = Episode(
            name=self._pair_label(a_msg_id, b_msg_id),
            aliases=(),
            actor=a.speaker_key,
            channel=a.channel_key,
            start=self.table._boundary(a, "start", "reported"),
            end=self.table._boundary(b, "end", "reported"),
            interval=interval,
        )


        candidates: list[Episode] = []
        values: list[int] = []
        for pair in ranked_pairs[: self.config.agreement_window]:
            pa, pb = self.log.get(pair.a_msg_id), self.log.get(pair.b_msg_id)
            if pb.ts < pa.ts:
                pa, pb = pb, pa
            candidate_interval = interval_between(pa.date, pb.date, self.table.corpus_days)
            values.append(candidate_interval.value(shape.quantity, shape.convention))
            candidates.append(Episode(
                name=self._pair_label(pair.a_msg_id, pair.b_msg_id), aliases=(),
                actor=pa.speaker_key, channel=pa.channel_key,
                start=self.table._boundary(pa, "start", "reported"),
                end=self.table._boundary(pb, "end", "reported"),
                interval=candidate_interval,
            ))
        value = interval.value(shape.quantity, shape.convention)
        if selected:
            confidence = 1.0
        elif values:
            confidence = sum(abs(candidate - value) <= 1 for candidate in values) / len(values)
        else:
            confidence = 0.0
        asserted = confidence >= self.config.assert_floor(shape.quantity, shape.shape)
        alternatives = tuple(
            candidate for candidate in candidates
            if candidate.msg_ids != episode.msg_ids
        )[: self.config.max_alternatives] if not asserted else ()
        return DurationFact(
            shape=shape, episodes=(episode,), value=value, components=(value,),
            confidence=confidence, alternatives=alternatives, asserted=asserted,
            notes=("interval between two distinct events; endpoints excluded",
                   "pair selected by escalation" if selected
                   else "confidence from ranked-pair arithmetic agreement"),
        )

    def _pair_label(self, a_msg_id: str, b_msg_id: str) -> str:
        for mid in (a_msg_id, b_msg_id):
            artifacts = self.log.get(mid).tags.artifacts
            if artifacts:
                return " ".join(str(artifacts[0]).split())[:120]
        return " ".join(self.log.get(a_msg_id).text.split())[:90]


def build(*, log: MessageLog, table: EpisodeTable, scores: Any,
          config: ResolverConfig = DEFAULT_RESOLVER_CONFIG,
          selector: Any = None) -> DurationResolver:
    return DurationResolver(log=log, table=table, scores=scores, config=config,
                            selector=selector)
