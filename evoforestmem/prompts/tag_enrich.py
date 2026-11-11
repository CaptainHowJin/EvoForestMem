from __future__ import annotations

import json
from collections.abc import Sequence

from ..log.message import Message
from ..log.tags_vocab import LIFECYCLE_LABELS, MARKER_LABELS

PROMPT_VERSION = "v1"

_SYSTEM = """You annotate workplace messages for a generic long-term memory system.
Return exactly one JSON object and no markdown. Infer the message's deliverables, authority
markers, and lifecycle jointly. Do not invent names. A deliverable is a short canonical
name for the document, task output, release, policy, design, analysis, or other work product
the CURRENT message is about; resolve implicit phrases using only the supplied same-speaker
context. Use stable lower-case wording without articles when possible.

Distinguish actual completion from intent or prediction: "will finish", "should be done",
and "submitting for review" are premature_claim, not complete. partial_complete means only
part of the scoped work is done. cancelled is non-terminal for episode pairing because a
cancelled effort has no delivered completion.

Schema:
{"deliverables":["string"],"markers":["label"],"lifecycle":"label or null","lifecycle_confidence":0.0}
Allowed marker labels: %s
Allowed lifecycle labels: %s
Confidence must be in [0,1]. Empty arrays and null are valid. Never output benchmark or
dataset identifiers unless literally present as the deliverable in the message.""" % (
    ", ".join(sorted(MARKER_LABELS)),
    ", ".join(sorted(LIFECYCLE_LABELS)),
)


def build_messages(message: Message, preceding: Sequence[Message], *,
                   prompt_version: str = PROMPT_VERSION) -> list[dict[str, str]]:
    payload = _payload(message, preceding, prompt_version=prompt_version)
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _payload(message: Message, preceding: Sequence[Message], *,
             prompt_version: str) -> dict:
    return {
        "prompt_version": prompt_version,
        "current": {
            "msg_id": message.msg_id,
            "date": message.date,
            "speaker": message.speaker_key,
            "channel": message.channel_key,
            "text": message.text,
        },
        "same_speaker_preceding_context": [
            {"date": m.date, "channel": m.channel_key, "text": m.text}
            for m in preceding
        ],
    }


_BATCH_SYSTEM = _SYSTEM + """

The user provides an items array. Annotate every item independently and return exactly:
{"results":[{"msg_id":"copied input id","deliverables":[],"markers":[],"lifecycle":null,
"lifecycle_confidence":0.0}]}
Return one result for every input msg_id, in the same order. Never merge evidence or deliverables
between items."""


def build_batch_messages(
    items: Sequence[tuple[Message, Sequence[Message]]], *,
    prompt_version: str = PROMPT_VERSION,
) -> list[dict[str, str]]:

    payload = {
        "prompt_version": prompt_version,
        "items": [
            _payload(message, preceding, prompt_version=prompt_version)
            for message, preceding in items
        ],
    }
    return [
        {"role": "system", "content": _BATCH_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
