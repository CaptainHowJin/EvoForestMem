from __future__ import annotations

LIFECYCLE_LABELS = frozenset({
    "start",
    "progress",
    "complete",
    "premature_claim",
    "revise",
    "archive",
    "blocked",
    "resumed",
    "partial_complete",
    "handoff",
    "cancelled",
})

MARKER_LABELS = frozenset({
    "ratified",
    "decided",
    "effective_now",
    "signoff",
    "mandate",
    "supersede",
    "retracted",
    "conditional_approval",
    "escalated",
    "delegated",
})

START_LABELS = frozenset({"start", "resumed"})
TERMINAL_END_LABELS = frozenset({"complete", "archive"})
NONTERMINAL_LABELS = frozenset({
    "premature_claim",
    "partial_complete",
    "progress",
    "blocked",
    "resumed",
    "handoff",
    "cancelled",
    "revise",
})
PROGRESS_LABELS = frozenset({
    "progress",
    "resumed",
    "partial_complete",
    "blocked",
    "handoff",
    "revise",
})

TAG_SOURCES = frozenset({"llm", "rule", "llm+rule"})
