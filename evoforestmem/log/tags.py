from __future__ import annotations

import re

from .tags_vocab import LIFECYCLE_LABELS, MARKER_LABELS


_URL = re.compile(r"https?://[^\s\)\]\},;'\"<>]+")


_PLATFORM = re.compile(
    r"\b(confluence|jira|swagger|figma|gitlab|github|sharepoint|notion|"
    r"wiki|postman|yapi)\b",
    re.IGNORECASE,
)


_IDENT = re.compile(r"\b(?:[A-Z][A-Z0-9]{1,}(?:[_-][A-Z0-9]+)+|[A-Z]{2,}-\d+|P[0-3])\b")


_QUOTED = re.compile(
    r"[\"“《「]([^\"”》」\n]{6,120})[\"”》」]"

    r"|(?<!\w)'([^'\n]{6,120})'(?!\w)"
    r"|\[([^\]\n]{6,120})\]"
)


_AT_MENTION = re.compile(r"@([A-Za-z一-鿿][\w一-鿿]*(?:\s+[A-Z][\w]*){0,2})")


_MARKERS: dict[str, re.Pattern[str]] = {
    "ratified": re.compile(
        r"\b(?:has been |is |was |now )?(?:formally |officially |mandatorily )?"
        r"(?:approved|ratified|finalized|finalised|locked|confirmed)\b",
        re.IGNORECASE,
    ),
    "decided": re.compile(r"\b(?:we(?:'ve| have)? )?decided\b|\bdecision (?:is|has been) made\b", re.IGNORECASE),
    "effective_now": re.compile(r"\beffective immediately\b|\btakes effect\b|\bwith immediate effect\b", re.IGNORECASE),
    "signoff": re.compile(r"\bsign[- ]?off\b|\bsigned off\b|\bsign off\b", re.IGNORECASE),
    "mandate": re.compile(r"\bmust\b|\brequired to\b|\bmandatory\b|\bmandatorily\b", re.IGNORECASE),
    "supersede": re.compile(r"\boverride\b|\bsupersed\w+\b|\breplaces? the\b|\bno longer (?:applies|valid)\b", re.IGNORECASE),
}


_LIFECYCLE: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "premature_claim",
        re.compile(
            r"\b(?:will|going to|expect to|plan to|aim to|should)\s+"
            r"(?:be\s+)?(?:finish|complete|deliver|submit|wrap up|close)\w*\b"
            r"|\bsubmitting\b.{0,40}\bfor review\b"
            r"|\bwill be (?:completed|delivered|finished)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "archive",
        re.compile(
            r"\barchiv\w+\b|\buploaded to\b|\bhas been released\b|\bdelivery page\b"
            r"|\bfinal version has been\b",
            re.IGNORECASE,
        ),
    ),
    (
        "complete",
        re.compile(
            r"\[task completed\]|\bhas been (?:successfully )?completed\b"
            r"|\bis now complete\b|\bhave been completed\b|\bcompleted!\b"
            r"|\bsuccessfully completed\b|\btask (?:is )?complete\b",
            re.IGNORECASE,
        ),
    ),
    (
        "revise",
        re.compile(
            r"\brevis\w+\b|\bupdated to\b|\bchanged to\b|\bre-?scheduled\b"
            r"|\brolled back\b|\bamend\w+\b",
            re.IGNORECASE,
        ),
    ),
    (
        "start",
        re.compile(
            r"\b(?:officially )?(?:launch|kick ?off|start(?:ing|ed)?|begin(?:ning)?)\b"
            r"|\bI(?:'m| am) starting\b|\btoday we\b.{0,30}\blaunch\b",
            re.IGNORECASE,
        ),
    ),
    (
        "progress",
        re.compile(
            r"\bin progress\b|\bprogress(?:ing)?\b|\bcurrently\b|\bso far\b|\bunderway\b",
            re.IGNORECASE,
        ),
    ),
)


def _dedupe(items) -> tuple[str, ...]:

    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = it.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return tuple(out)


def artifacts(text: str) -> tuple[str, ...]:
    found: list[str] = []
    found += _URL.findall(text)
    found += (m.lower() for m in _PLATFORM.findall(text))
    found += _IDENT.findall(text)
    for groups in _QUOTED.findall(text):
        for g in groups:
            if g and g.strip():
                found.append(g.strip())
    return _dedupe(found)


def mentions(text: str, known: set[str] | None = None) -> tuple[str, ...]:

    found = [m.strip() for m in _AT_MENTION.findall(text)]
    if known:
        for name in sorted(known):
            if len(name) > 3 and name in text:
                found.append(name)
    return _dedupe(found)


def markers(text: str) -> tuple[str, ...]:
    out = tuple(name for name, pat in _MARKERS.items() if pat.search(text))
    assert set(out) <= MARKER_LABELS
    return out


def lifecycle(text: str) -> str | None:
    for name, pat in _LIFECYCLE:
        if pat.search(text):
            assert name in LIFECYCLE_LABELS
            return name
    return None


def derive(text: str, known_names: set[str] | None = None):

    from .message import Tags

    return Tags(
        mentions=mentions(text, known_names),
        artifacts=artifacts(text),
        markers=markers(text),
        lifecycle=lifecycle(text),
    )
