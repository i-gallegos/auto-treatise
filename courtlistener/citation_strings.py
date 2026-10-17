"""Parse corpus citation strings and pick the one to display."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

#: Rough precedence among reporters.
_REPORTER_PRIORITY: tuple[str, ...] = (
    "U.S.",
    "S. Ct.",
    "L. Ed.",
    "F.4th",
    "F.3d",
    "F.2d",
    "F.",
    "F. Supp. 3d",
    "F. Supp. 2d",
    "F. Supp.",
    "F.R.D.",
    "B.R.",
    "Fed. Cl.",
)


def parse_citations(raw: Any) -> list[str]:
    """
    ``cases.citation`` holds a JSON list of reporter cites.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        candidates: Iterable[Any] = raw
    else:
        text = str(raw).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                candidates = json.loads(text)
            except json.JSONDecodeError:
                candidates = [text]
        else:
            candidates = re.split(r"\s*;\s*", text)

    seen, out = set(), []
    for item in candidates:
        cite = str(item).strip()
        if cite and cite not in seen:
            seen.add(cite)
            out.append(cite)
    return out


_VOLUME_CITE_RE = re.compile(
    r"^\s*(\d{1,4})\s+([A-Z][A-Za-z0-9.'&\s]{0,20}?)\s+\d{1,5}\s*$"
)


_YEAR_LIKE = range(1754, 2101)


def _cite_sort_key(cite: str) -> tuple[int, int, str]:
    match = _VOLUME_CITE_RE.match(cite)
    if not match or int(match.group(1)) in _YEAR_LIKE:
        return (2, len(_REPORTER_PRIORITY), cite)
    reporter = match.group(2).strip()
    try:
        rank = _REPORTER_PRIORITY.index(reporter)
    except ValueError:
        rank = len(_REPORTER_PRIORITY)
    return (0, rank, cite)


def choose_primary_citation(candidates: Sequence[str]) -> str | None:
    """The citation a reader would look the case up by, or ``None``."""
    cleaned = [c.strip() for c in candidates if c and c.strip()]
    if not cleaned:
        return None
    return sorted(cleaned, key=_cite_sort_key)[0]
