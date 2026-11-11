from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

__all__ = [
    "MAX_WORDS",
    "ROOT_MAX_WORDS",
    "PromptTemplate",
    "SYSTEM_BY_VIEW",
    "USER_TEMPLATE",
    "build_messages",
    "evidence_msg_ids",
    "parse_summary",
    "system_prompt_for",
    "time_range_text",
]


MAX_WORDS: dict[str, int] = {"channel": 200, "actor": 220, "artifact": 180}
ROOT_MAX_WORDS: dict[str, int] = {"actor": 160}
DEFAULT_MAX_WORDS = 160


@dataclass(frozen=True, slots=True)
class PromptTemplate:


    name: str
    system: str
    user: str

    def render(self, **context: object) -> tuple[str, str]:
        return self.system.format(**context), self.user.format(**context)


_SHARED_RULES = """\
Write ONE paragraph of at most {max_words} words. Nothing else: no heading, no bullet \
list, no preamble, no closing remark.

Keep, spelled out exactly as the input spells them:
  - every person's name, and who said, did, asked for or decided what
  - every date, deadline, duration, number, amount, percentage, version and status
  - every named artifact: file, document, dataset, system, tool, ticket, metric, standard
  - every decision, commitment, blocker, disagreement, and question left unanswered

Never:
  - judge, praise or criticise a person, or make unsupported claims about their style, skill or character
  - state a cause, motive, consequence or outcome that the input does not state
  - offer advice, recommendations, lessons, or an assessment of how things went
  - hedge ("appears to", "seems to", "likely") -- state what the input states, or omit it
  - replace named things with category words ("various files", "several issues",
    "some concerns"). Dropping a minor thread entirely is better than generalising it:
    a name that survives is searchable, a category word is not.

Write "X -> Y" for a change of state (draft -> approved, 12% -> 18%). Keep events in \
chronological order."""

_CHANNEL_FOCUS = """\
You are indexing one time interval of the project channel "{tree_label}" — a multi-party \
working group chat. This view's axis is the GROUP: within this interval, capture who \
participated and what each named speaker drove, which workstreams and deliverables \
advanced, every decision and who made it, every commitment, blocker and disagreement, and \
the questions left unanswered. Attribute each decision and commitment to the speaker who \
made it. One channel interval mixes several parallel workstreams — keep them distinct \
rather than blending them into one narrative."""

_CHANNEL_THREAD_FOCUS = """\
You are indexing one reply conversation in the project channel "{tree_label}". Identify \
the question, proposal, or event that opened the conversation; name every participant and \
what each contributed; preserve every decision and who made it, every commitment, blocker, \
disagreement, and unanswered question; and state the conversation's resolution or latest \
status when the input provides one. Follow reply context rather than treating each message \
as an isolated statement. If the input contains independent messages with no reply \
relationship, keep them explicitly separate rather than blending them into one narrative."""

_ACTOR_FOCUS = """\
You are indexing one interval of the person-centric tree for "{tree_label}". Lines authored \
by {tree_label} are their own statements: summarise what they stated, proposed, confirmed, \
decided, assigned, or requested using speech-act wording. Lines authored by other speakers \
are things said TO or ABOUT {tree_label}: attribute every such claim to its speaker \
("<other> asked / assigned to / reported to / described {tree_label}"), and never convert an \
about-claim into a self-decision. Preserve channel attribution. Keep names, dates and \
decisions verbatim. Do not infer a title, formal responsibility, or authority from repeated \
activity alone."""

_ARTIFACT_FOCUS = """\
You are indexing one lifecycle interval of the deliverable "{tree_label}" — a document, \
dataset, system or task tracked across a multi-party group chat. This view's axis is one \
DELIVERABLE across the whole organisation: within this interval, record its state \
transitions as "<state> -> <state>" and who drove each, the version and status changes, \
the decisions and blockers about it, and — because this deliverable is discussed in more \
than one channel — which channels and speakers touched it here. Keep every date, version \
and status verbatim."""

_PREAMBLE = """\
{focus}

What you write is an INDEX ENTRY, not an answer. A later stage uses it only to decide \
which interval of which view to open, and then quotes the original messages verbatim to \
answer the question. So being vague costs a wrong routing decision, and being confident \
about something the input does not say costs a wrong answer that nothing downstream can \
detect.

{rules}

Output the paragraph and nothing else."""


def _system(view: str, focus: str) -> PromptTemplate:
    return PromptTemplate(
        name=f"node_summary_{view}",
        system=_PREAMBLE.format(focus=focus, rules=_SHARED_RULES),
        user="",
    )


SYSTEM_BY_VIEW: dict[str, PromptTemplate] = {
    "channel": _system("channel", _CHANNEL_FOCUS),
    "actor": _system("actor", _ACTOR_FOCUS),


    "artifact": _system("artifact", _ARTIFACT_FOCUS),
}

CHANNEL_THREAD_TEMPLATE = _system("channel_thread", _CHANNEL_THREAD_FOCUS)

_ACTOR_BRANCH_OUTPUT = """\
Return plain English text only in 2 to 4 compact sentences. Keep representative evidence \
message ids inline when present so the root can audit its claims. Do not output Markdown, \
bullets, headings, or raw JSON."""

_ACTOR_ROOT_OUTPUT = """\
You are generating the ROOT summary for the person-centric memory tree of "{tree_label}" — \
a persona card used to route a query to this person and point at the messages that ground \
it. Write compact labelled prose (no JSON, no bullets) in at most {max_words} words, these \
clauses in order, each grounded in the input and omitted (not guessed) when unsupported:
  Role: what {tree_label} does, in the corpus's own words.
  Team / channels: the groups they work in.
  Responsibilities: what they own or drive, each as "owns/drives <what>".
  Authority: decisions they are shown making or approving, only with explicit decision wording.
  Style: recurring phrasings observed in their own messages.
  As seen by others: how other speakers address, assign to, or describe them, attributed to \
the other speaker.
  Evidence: at most three message ids that ground the clauses above, copied exactly from the input.
Distinguish what {tree_label} said from what others said about them. Never claim a title or \
authority without explicit evidence; use "appears to / is treated as" for indirect patterns. \
The final clause must be exactly "Evidence: <id>, <id>, ..." and contain no text after it. \
Do not output Markdown."""

_CHANNEL_ROOT_OUTPUT = """\
You are generating the ROOT summary for the channel-centric memory tree of "{tree_label}" \
— a digest card for the whole channel, used to route a query to this group. Write compact \
labelled prose (no JSON, no bullets) in at most {max_words} words, these clauses in this \
order, each grounded in the input and omitted (not guessed) when unsupported:
  Span: <first date> to <last date>.
  Participants: the speakers who drove work, most active first.
  Workstreams: the recurring deliverables/topics this channel worked on.
  Decisions: each major decision as "<who> decided <what> (<date>)".
  Open threads: questions or blockers still unresolved at the end of the span.
Keep every name, deliverable, date and status verbatim; never generalise a named thing to \
a category word; never add a decision, cause or outcome the input does not state."""

_ARTIFACT_ROOT_OUTPUT = """\
You are generating the ROOT summary for the deliverable-centric memory tree of \
"{tree_label}" — a lifecycle card for the whole deliverable, used to route a query to it. \
Write compact labelled prose (no JSON, no bullets) in at most {max_words} words, these \
clauses in this order, each grounded in the input and omitted when unsupported:
  Aliases: other names this deliverable is called.
  Owner(s): who drove it, most responsible first.
  Channels: the channels it was discussed in (this deliverable spans more than one).
  Current status: its latest state, as of <date>.
  Lifecycle: the state transitions in order, each as "<state> (<date>, <who>)".
  Key decisions: each as "<who> decided <what> (<date>)".
Keep every name, date, version and status verbatim; never generalise a named thing; never \
assert a status or transition the input does not state."""

ROOT_OUTPUT_BY_VIEW: dict[str, str] = {
    "actor": _ACTOR_ROOT_OUTPUT,
    "channel": _CHANNEL_ROOT_OUTPUT,
    "artifact": _ARTIFACT_ROOT_OUTPUT,
}


USER_TEMPLATE = PromptTemplate(
    name="node_summary_user",
    system="",
    user="""{view_word}: {tree_label}
Interval: {time_range}
Level: {level}
{source_line}

{input_text}""",
)

_VIEW_WORD: dict[str, str] = {
    "channel": "Channel",
    "actor": "Person",
    "artifact": "Artifact",
}


def time_range_text(ts_start: float, ts_end: float) -> str:

    try:
        start = datetime.fromtimestamp(ts_start, tz=timezone.utc).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(ts_end, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return ""
    return start if start == end else f"{start} to {end}"


def system_prompt_for(view: str, *, max_words: int | None = None) -> str:
    effective_view = view if view in SYSTEM_BY_VIEW else "channel"
    template = SYSTEM_BY_VIEW[effective_view]
    words = MAX_WORDS.get(effective_view, DEFAULT_MAX_WORDS) if max_words is None else max_words
    return template.system.format(tree_label="{tree_label}", max_words=words)


def build_messages(
    *,
    view: str,
    tree_label: str,
    level: int,
    ts_start: float,
    ts_end: float,
    input_text: str,
    n_children: int | None = None,
    max_words: int | None = None,
    is_root: bool = False,
    actor_branch: str = "self-authored",
    channel_thread: bool = False,
) -> list[dict[str, str]]:

    effective_view = view if view in SYSTEM_BY_VIEW else "channel"
    template = SYSTEM_BY_VIEW[effective_view]
    if view == "channel" and level <= 1 and not is_root and channel_thread:
        template = CHANNEL_THREAD_TEMPLATE
    words = MAX_WORDS.get(effective_view, DEFAULT_MAX_WORDS) if max_words is None else max_words
    if is_root and max_words is None:
        words = ROOT_MAX_WORDS.get(effective_view, words)
    system = template.system.format(tree_label=tree_label, max_words=words)
    if is_root and view in ROOT_OUTPUT_BY_VIEW:


        system = ROOT_OUTPUT_BY_VIEW[view].format(tree_label=tree_label, max_words=words)
    elif view == "actor":
        if actor_branch not in {"self-authored", "mentioned/about", "unified"}:
            raise ValueError(f"unknown actor branch: {actor_branch!r}")
        system += "\n\n" + _ACTOR_BRANCH_OUTPUT.format(tree_label=tree_label)

    if level <= 1:
        kind = "verbatim messages, in order"
    else:
        kind = "summaries of consecutive sub-intervals, in order"
    source_line = f"Source: {n_children} {kind}." if n_children else f"Source: {kind}."

    _, user = USER_TEMPLATE.render(
        view_word=_VIEW_WORD.get(view, "Group"),
        tree_label=tree_label,
        time_range=time_range_text(ts_start, ts_end) or "unknown",
        level=level,
        source_line=source_line,
        input_text=input_text,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_LABEL = re.compile(r"^\s*\*{0,2}\s*summary\s*\*{0,2}\s*[:\-]\s*\*{0,2}\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")
_EVIDENCE = re.compile(r"(?:^|[.;]\s*)Evidence:\s*([^.;]+)\s*$", re.IGNORECASE)
_MSG_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")


def evidence_msg_ids(summary: str, *, limit: int = 3) -> tuple[str, ...]:

    match = _EVIDENCE.search((summary or "").strip())
    if match is None or limit <= 0:
        return ()
    ids: list[str] = []
    for msg_id in _MSG_ID.findall(match.group(1)):
        if msg_id not in ids:
            ids.append(msg_id)
        if len(ids) >= limit:
            break
    return tuple(ids)


def parse_summary(raw: str, *, preserve_json: bool = False) -> str:

    text = (raw or "").strip()
    if not text:
        return ""
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()


    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict) and preserve_json:
                return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            if isinstance(obj, dict):
                value = obj.get("summary")
                if isinstance(value, str):
                    text = value.strip()
                elif value is None:
                    joined = " ".join(v for v in obj.values() if isinstance(v, str))
                    text = joined.strip() or text
    text = _LABEL.sub("", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return _WS.sub(" ", text).strip()
