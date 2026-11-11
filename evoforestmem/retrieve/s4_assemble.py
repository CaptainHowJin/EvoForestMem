from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping, Sequence

Style = Literal["plain", "structured"]


SectionMode = Literal["on_change", "grouped", "none"]


_CHARS_PER_TOKEN = 4.0


def approx_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


@dataclass(frozen=True, slots=True)
class Span:


    msg_id: str
    score: float = 0.0
    view: str | None = None

    headers: tuple[str, ...] = ()


@dataclass
class Assembled:
    context: str
    provenance: list[str] = field(default_factory=list)
    n_spans: int = 0
    n_requested: int = 0
    tokens: int = 0
    truncated: bool = False
    fallback: bool = False
    depth_used: dict[int, int] = field(default_factory=dict)


    mandatory_infeasible: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_spans": self.n_spans,
            "n_requested": self.n_requested,
            "tokens": self.tokens,
            "truncated": self.truncated,
            "fallback": self.fallback,
            "depth_used": self.depth_used,
            "mandatory_infeasible": self.mandatory_infeasible,
            "provenance": self.provenance,
        }


def _stamp(ts: float, *, seconds: bool = False) -> str:
    fmt = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)


def _one_line(header: str) -> str:

    return " ".join(header.split())


def _render_span(log, span: Span, depth: int, style: Style, *, marks: bool = False,
                 marker_marks: bool = False, rank: int | None = None,
                 render_names: tuple[Mapping[str, str], Mapping[str, str]] | None = None,
                 render_seconds: bool = False) -> str:
    m = log.get(span.msg_id)
    body = log.text(span.msg_id)


    if render_names is not None:
        channel_names, speaker_names = render_names
        channel = channel_names.get(m.channel_key, m.channel_key)
        speaker = speaker_names.get(m.speaker_key, m.speaker_key)
    else:
        channel, speaker = m.channel_key, m.speaker_key

    if style == "plain":


        return f"[{_stamp(m.ts, seconds=render_seconds)}] {speaker}: {body}"

    head = ""
    if depth >= 1 and span.headers:
        chosen = span.headers[-1:] if depth == 1 else span.headers
        head = "".join(f"    ({_one_line(h)})\n" for h in chosen)
    annotations: list[str] = []
    if marks and m.tags.lifecycle:
        annotations.append(m.tags.lifecycle)
    if marker_marks:
        annotations.extend(m.tags.markers)
    mark = "" if not annotations else " " + " ".join(f"[{a}]" for a in annotations)


    number = "" if rank is None else f"#{rank} "
    return (f"{head}{number}[{_stamp(m.ts, seconds=render_seconds)}] "
            f"{channel} / {speaker}{mark}:\n{body}")


def _emit(log, chosen: Sequence[Span], depths: Sequence[int], *, style: Style,
          fallback: bool, sections: SectionMode, marks: bool = False,
          marker_marks: bool = False, rank_numbers: bool = False,
          derived: Sequence[Any] = (),
          render_names: tuple[Mapping[str, str], Mapping[str, str]] | None = None,
          render_seconds: bool = False) -> tuple[list[tuple[Span, int]], list[str]]:

    if sections == "grouped" and style == "structured" and not fallback:
        order = sorted(range(len(chosen)),
                       key=lambda i: (chosen[i].view or "", log.get(chosen[i].msg_id).ts))
    else:


        order = sorted(range(len(chosen)), key=lambda i: log.get(chosen[i].msg_id).ts)
    placed = [(chosen[i], depths[i]) for i in order]


    rank_of: dict[str, int] = {}
    if rank_numbers:
        by_score = sorted(range(len(chosen)), key=lambda i: -chosen[i].score)
        rank_of = {chosen[i].msg_id: r + 1 for r, i in enumerate(by_score)}

    blocks: list[str] = []
    if style == "structured" and not fallback:


        for fact in derived:
            blocks.extend(_one_line(line) for line in fact.render())
    if style == "structured" and not fallback and sections != "none":
        current_view: str | None = object()
        for span, depth in placed:
            if span.view != current_view:
                current_view = span.view
                if span.view:
                    blocks.append(f"--- {span.view} ---")
            blocks.append(_render_span(log, span, depth, style, marks=marks,
                                       marker_marks=marker_marks,
                                       rank=rank_of.get(span.msg_id),
                                       render_names=render_names,
                                       render_seconds=render_seconds))
    else:
        blocks.extend(_render_span(log, s, d, "plain" if fallback else style,
                                   marks=marks and not fallback,
                                   marker_marks=marker_marks and not fallback,
                                   rank=rank_of.get(s.msg_id),


                                   render_names=None if fallback else render_names,
                                   render_seconds=False if fallback else render_seconds)
                      for s, d in placed)
    return placed, blocks


def assemble(
    log,
    spans: Sequence[Span],
    *,
    budget_tokens: int = 3000,
    style: Style = "structured",
    max_depth: int = 2,
    min_spans: int = 1,
    header_budget_frac: float = 0.35,
    section_headers: SectionMode = "on_change",
    lifecycle_marks: bool = False,
    marker_marks: bool = False,
    rank_numbers: bool = False,
    derived: Sequence[Any] = (),
    mandatory_msg_ids: Iterable[str] = (),
    fallback_spans: Iterable[Span] | None = None,
    count_tokens=approx_tokens,
    render_names: tuple[Mapping[str, str], Mapping[str, str]] | None = None,
    render_seconds: bool = False,
) -> Assembled:

    if style == "plain":
        derived = ()
    mandatory = set(mandatory_msg_ids)


    best: dict[str, Span] = {}
    for s in spans:
        prev = best.get(s.msg_id)
        if prev is None or s.score > prev.score:
            best[s.msg_id] = s


    ranked = sorted(best.values(), key=lambda s: (s.msg_id not in mandatory, -s.score))

    used = 0
    truncated = False

    chosen: list[Span] = []
    depths: list[int] = []
    body_cost: list[int] = []


    infeasible: list[str] = []
    for span in ranked:
        cost = count_tokens(_render_span(log, span, 0, style, marks=lifecycle_marks,
                                         marker_marks=marker_marks,
                                         render_names=render_names,
                                         render_seconds=render_seconds)) + 1
        if used + cost > budget_tokens:
            truncated = True
            if span.msg_id in mandatory:
                infeasible.append(span.msg_id)
                chosen.append(span)
                depths.append(0)
                body_cost.append(cost)
                used += cost
            continue
        chosen.append(span)
        depths.append(0)
        body_cost.append(cost)
        used += cost


    if style == "structured" and max_depth > 0 and chosen:
        header_cap = min(budget_tokens - used, int(budget_tokens * header_budget_frac))
        header_used = 0
        for i, span in enumerate(chosen):
            if not span.headers:
                continue
            for depth in range(min(max_depth, 2), 0, -1):
                delta = count_tokens(
                    _render_span(log, span, depth, style, marks=lifecycle_marks,
                                 marker_marks=marker_marks,
                                 render_names=render_names,
                                 render_seconds=render_seconds)
                ) + 1 - body_cost[i]
                if delta <= 0 or header_used + delta <= header_cap:
                    depths[i] = depth
                    header_used += max(0, delta)
                    used += max(0, delta)
                    break

    fallback = False
    if len(chosen) < min_spans and fallback_spans is not None:
        fallback = True


        keep_mandatory = [s for s in chosen if s.msg_id in mandatory]
        chosen, depths, used = [], [], 0
        have: set[str] = set()
        for span in keep_mandatory:
            cost = count_tokens(_render_span(log, span, 0, "plain",
                                             render_names=render_names,
                                             render_seconds=render_seconds)) + 1
            chosen.append(span)
            depths.append(0)
            used += cost
            have.add(span.msg_id)
        for span in fallback_spans:
            if span.msg_id in have:
                continue
            cost = count_tokens(_render_span(log, span, 0, "plain",
                                             render_names=render_names,
                                             render_seconds=render_seconds)) + 1
            if used + cost > budget_tokens:

                truncated = True
                continue
            chosen.append(span)
            depths.append(0)
            used += cost
            have.add(span.msg_id)


    keep = list(range(len(chosen)))
    floor = 1 if fallback else max(1, min_spans)
    while True:
        kept_ids = {chosen[i].msg_id for i in keep}


        rendered = [f for f in derived if set(f.msg_ids) <= kept_ids]
        placed, blocks = _emit(log, [chosen[i] for i in keep], [depths[i] for i in keep],
                               style=style, fallback=fallback, sections=section_headers,
                               marks=lifecycle_marks, marker_marks=marker_marks,
                               rank_numbers=rank_numbers, derived=rendered,
                               render_names=render_names, render_seconds=render_seconds)
        context = "\n\n".join(blocks) if blocks else "(No memories retrieved)"
        total = count_tokens(context)
        if total <= budget_tokens or len(keep) <= floor:
            break
        droppable = [i for i in keep if chosen[i].msg_id not in mandatory]
        if not droppable:
            break
        worst = min(droppable, key=lambda i: (chosen[i].score, -i))
        keep.remove(worst)
        truncated = True

    depth_used: dict[int, int] = {}
    for _, d in placed:
        depth_used[d] = depth_used.get(d, 0) + 1

    return Assembled(
        context=context,
        provenance=[s.msg_id for s, _ in placed],
        n_spans=len(placed),
        n_requested=len(ranked),
        tokens=total,
        truncated=truncated,
        fallback=fallback,
        depth_used=depth_used,
        mandatory_infeasible=infeasible,
    )
