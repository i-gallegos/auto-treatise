"""Isolate the majority opinion from concurrences and dissents.

Using:
  1. Signature lines ("KIRSCH, Circuit Judge, dissenting.")
  2. Running headers ("Cite as: 604 U. S. 593 (2025) 631 Kavanaugh, J.,
     dissenting")
  3. Front matter ("Opinion by Judge VanDyke; Dissent by Judge Schroeder")
"""

import json
import re
import sqlite3
from typing import Any

from .utils import cache_write

MAJORITY = "majority"
CONCURRENCE = "concurrence"
DISSENT = "dissent"
FRONT_MATTER = "front_matter"

STATUS_MAJORITY_ONLY = "majority_only"
STATUS_BOUNDARY_FOUND = "boundary_found"
STATUS_NO_MAJORITY = "no_majority"
STATUS_AMBIGUOUS = "ambiguous"

MIN_BOUNDARY_PARAGRAPH_CHARS = 40
MAJORITY_START_SEARCH_LIMIT = 20
HEADER_SEARCH_CHARS = 200

NO_MAJORITY_LABEL_FRACTION = 0.7
NO_MAJORITY_MAX_LEADING_PARAGRAPHS = 5

# ── Regex components ──────────────────────────────────────────────────────────

_NAME_TOKEN = r"[A-Z][A-Za-z.'\u2019-]*"
_NAME = rf"(?:{_NAME_TOKEN}\s+){{0,3}}{_NAME_TOKEN}"
_ROLE = r"(?:(?:Circuit|District|Chief|Senior|Associate|Presiding|Acting)\s+)*(?:Judge|Justice|J\.|C\.\s*J\.)"
# Joinder clauses vary widely and can chain, as in "with whom Justice Alito
# joins, and with whom Justice Kavanaugh and Justice Barrett join as to Parts I
# and II, dissenting." Requiring the "with whom" trigger keeps the permissive
# bridge from over-matching.
_JOINDER = r"(?:with\s+whom\b.{0,250}?,\s*)?"

# Continuation words come from a closed lowercase set. This is separates
# genuine signature line from a running header: "dissenting." terminates
# cleanly, whereas "Kavanaugh, J., dissenting For example, ..." runs straight
# into capitalised body text and so never reaches the required "." or ":".
_DISP_MODIFIER = (
    r"(?:(?:specially|respectfully|partially|separately|dubitante)\s+)?"
)
_DISP_WORD = rf"{_DISP_MODIFIER}(?:dissenting|concurring)"
_DISP_CONT = (
    r"(?:\s+(?:in|and|from|of|to|the|a|part|parts|judgment|judgments|result|results"
    r"|denial|denials|grant|opinion|whole|reversal|affirmance|certification|rehearing"
    r"|review|transfer|mandamus|en|banc)\b)*"
)
# Chained dispositions may be joined by a comma, as in
# "concurring in judgment in part, dissenting in part."
_DISP_PHRASE = rf"{_DISP_WORD}{_DISP_CONT}(?:,?\s+{_DISP_WORD}{_DISP_CONT})*"

# A joinder clause can also follow the disposition:
# "FORREST, Circuit Judge, concurring, with whom BUMATAY, Circuit Judge, joins:"
_TRAILING_JOINDER = r"(?:,\s*with\s+whom\b.{0,250}?)?"

# A first-person pronoun immediately after the byline terminates a signature
# line just as reliably as a period: "CALABRESI, Circuit Judge, concurring in
# the judgment, I do not..."
_SIG_TERMINATOR = r"(?:\s*[.:]|,\s+I\s)"

# "NALBANDIAN, Circuit Judge, dissenting." / "Rawlinson, Circuit Judge,
# concurring in the result:"
_SIG_WITH_ROLE_RE = re.compile(
    rf"(?P<name>{_NAME}),\s*{_ROLE},\s*{_JOINDER}(?P<disp>{_DISP_PHRASE})"
    rf"{_TRAILING_JOINDER}{_SIG_TERMINATOR}"
)

# "JUSTICE JACKSON, concurring in the judgment." / "Justice Kagan, with whom
# Justice Sotomayor and Justice Jackson join, dissenting."
_SIG_JUSTICE_RE = re.compile(
    rf"(?P<name>(?:JUSTICE|Justice)\s+{_NAME_TOKEN}),\s*{_JOINDER}(?P<disp>{_DISP_PHRASE})"
    rf"{_TRAILING_JOINDER}{_SIG_TERMINATOR}"
)

_SIGNATURE_RES = (_SIG_WITH_ROLE_RE, _SIG_JUSTICE_RE)

# Slip-opinion running header for a separate opinion: "Kavanaugh, J.,
# dissenting" with no terminator, because reflowed body text follows on the same
# line.
_HEADER_SEPARATE_RE = re.compile(
    rf"(?P<name>{_NAME_TOKEN}),\s*(?:C\.\s*)?J(?:J)?\.,\s*(?P<disp>{_DISP_PHRASE})"
)

# Texas-style running header: "WILLIAMS DISSENT—4"
_HEADER_CAPS_RE = re.compile(
    r"\b[A-Z][A-Z'\u2019-]{2,}\s+(?P<kind>DISSENT|CONCURRENCE)\b"
)

# Slip-opinion running header for the lead opinion.
_HEADER_MAJORITY_RE = re.compile(r"Opinion\s+of\s+the\s+Court|Per\s+Curiam")

# Reporter prefix introducing a running header; the cut belongs before it.
_REPORTER_PREFIX_RE = re.compile(r"Cite\s+as:")

# Lead-author anchors marking where the majority actually begins.
_LEAD_ANCHOR_RES = (
    re.compile(r"Per\s+Curiam\s*[.:]"),
    re.compile(r"delivered\s+the\s+opinion\s+of\s+the\s+[Cc]ourt"),
    re.compile(rf"(?P<name>{_NAME}),\s*{_ROLE}\s*[.:]"),
)

# Front-matter announcements that separate opinions exist.
_EXPECTATION_RES = (
    re.compile(
        r"(?:Dissent|Concurrence)\s+by\s+(?:Judge|Justice)\s+[A-Z][A-Za-z]*"
    ),
    re.compile(
        r"(?:Judge|Justice)\s+[A-Z][A-Za-z]*\s+(?:dissents|concurs|dissented|concurred)\b"
    ),
    re.compile(
        r"delivered\s+a\s+separate\s+(?:dissenting|concurring)\s+opinion"
    ),
    re.compile(r"filed\s+a\s+(?:dissenting|concurring)\s+opinion"),
)


# ── Marker detection ──────────────────────────────────────────────────────────


def _inside_parentheses(text, position):
    """True when `position` sits inside an unclosed '(' — i.e. a parenthetical
    citation."""
    depth = 0
    for char in text[:position]:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
    return depth > 0


def _label_for_disposition(disposition):
    return DISSENT if "dissent" in disposition.lower() else CONCURRENCE


def _find_signature(text):
    """Earliest genuine signature line in `text`, or None."""
    best = None
    for pattern in _SIGNATURE_RES:
        for match in pattern.finditer(text):
            if _inside_parentheses(text, match.start()):
                continue
            if best is None or match.start() < best["offset"]:
                best = {
                    "offset": match.start(),
                    "marker": match.group(0).strip(),
                    "label": _label_for_disposition(match.group("disp")),
                    "kind": "signature",
                }
            break
    return best


def _find_separate_header(text):
    """Running header identifying a concurrence or dissent block, or None."""
    head = text[:HEADER_SEARCH_CHARS]

    caps = _HEADER_CAPS_RE.search(head)
    if caps:
        return {
            "offset": caps.start(),
            "marker": caps.group(0).strip(),
            "label": (
                DISSENT if caps.group("kind") == "DISSENT" else CONCURRENCE
            ),
            "kind": "header",
        }

    for match in _HEADER_SEPARATE_RE.finditer(head):
        if _inside_parentheses(head, match.start()):
            continue
        offset = match.start()
        prefix = _REPORTER_PREFIX_RE.search(head[:offset])
        if prefix:
            offset = prefix.start()
        return {
            "offset": offset,
            "marker": match.group(0).strip(),
            "label": _label_for_disposition(match.group("disp")),
            "kind": "header",
        }
    return None


def _find_majority_header(text):
    """Running header identifying the lead opinion, or None."""
    match = _HEADER_MAJORITY_RE.search(text[:HEADER_SEARCH_CHARS])
    if not match:
        return None
    return {
        "offset": match.start(),
        "marker": match.group(0).strip(),
        "label": MAJORITY,
        "kind": "header",
    }


def _find_majority_start(paragraphs):
    """Index of the paragraph where the lead opinion begins.

    Returns 0 when no anchor is found.
    """
    limit = min(len(paragraphs), MAJORITY_START_SEARCH_LIMIT)
    for index in range(limit):
        text = paragraphs[index]
        signature = _find_signature(text)
        for pattern in _LEAD_ANCHOR_RES:
            for match in pattern.finditer(text):
                if _inside_parentheses(text, match.start()):
                    continue
                # A separate opinion's byline is not a lead anchor.
                if signature and signature[
                    "offset"
                ] <= match.start() <= signature["offset"] + len(
                    signature["marker"]
                ):
                    continue
                return index
    return 0


def _find_expectations(paragraphs):
    """Front-matter announcements that separate opinions exist."""
    expectations = []
    for text in paragraphs:
        for pattern in _EXPECTATION_RES:
            for match in pattern.finditer(text):
                marker = match.group(0).strip()
                if marker not in expectations:
                    expectations.append(marker)
    return expectations


def _collect_markers(paragraphs, majority_start):
    """Per-paragraph block-start markers, keyed by paragraph index."""
    majority_headers = {}
    separate_markers = {}
    for index in range(majority_start, len(paragraphs)):
        text = paragraphs[index]

        majority_header = _find_majority_header(text)
        if majority_header:
            majority_headers[index] = majority_header

        candidates = [
            marker
            for marker in (_find_signature(text), _find_separate_header(text))
            if marker is not None
        ]
        if candidates:
            separate_markers[index] = min(candidates, key=lambda m: m["offset"])
    return majority_headers, separate_markers


def _label_paragraphs(
    paragraphs, majority_start, majority_headers, separate_markers
):
    """Label every paragraph by forward-filling block-start markers.
    """
    labels = [FRONT_MATTER] * majority_start + [None] * (
        len(paragraphs) - majority_start
    )
    current = MAJORITY
    for index in range(majority_start, len(paragraphs)):
        # The last marker in a paragraph wins, since text after it belongs to
        # that block.
        marker = majority_headers.get(index)
        if index in separate_markers:
            separate = separate_markers[index]
            if marker is None or separate["offset"] >= marker["offset"]:
                marker = separate
        if marker:
            current = marker["label"]
        labels[index] = current
    return labels


# ── Public API ────────────────────────────────────────────────────────────────


def find_majority_boundary(paragraphs):
    """
    Locate the end of the majority opinion in an ordered list of cleaned
    paragraphs.

    Returns a dict with:
      end_index        — number of leading paragraphs belonging to the majority
      end_char_offset  — cut offset inside paragraphs[end_index], or None for a
                         clean cut
      status           — majority_only | boundary_found | no_majority |
                         ambiguous
      marker           — the text that triggered the cut
      majority_start   — index where the lead opinion begins
    """
    if not paragraphs:
        return {
            "end_index": 0,
            "end_char_offset": None,
            "status": STATUS_MAJORITY_ONLY,
            "marker": None,
            "majority_start": 0,
        }

    majority_start = _find_majority_start(paragraphs)
    expectations = _find_expectations(paragraphs[: majority_start + 1])
    majority_headers, separate_markers = _collect_markers(
        paragraphs, majority_start
    )
    labels = _label_paragraphs(
        paragraphs, majority_start, majority_headers, separate_markers
    )

    boundary_index = min(separate_markers) if separate_markers else None

    if boundary_index is None:
        return {
            "end_index": len(paragraphs),
            "end_char_offset": None,
            "status": (
                STATUS_AMBIGUOUS if expectations else STATUS_MAJORITY_ONLY
            ),
            "marker": None,
            "majority_start": majority_start,
        }

    boundary = separate_markers[boundary_index]

    separate_count = sum(
        1 for label in labels[majority_start:] if label != MAJORITY
    )
    labeled_count = len(labels) - majority_start
    mostly_separate = (
        labeled_count > 0
        and separate_count / labeled_count >= NO_MAJORITY_LABEL_FRACTION
    )
    leading_paragraphs = boundary_index - majority_start

    if boundary_index == majority_start and boundary["offset"] == 0:
        status = STATUS_NO_MAJORITY
    elif (
        mostly_separate
        and leading_paragraphs <= NO_MAJORITY_MAX_LEADING_PARAGRAPHS
    ):
        status = STATUS_NO_MAJORITY
    else:
        status = STATUS_BOUNDARY_FOUND

    return {
        "end_index": boundary_index,
        "end_char_offset": (
            boundary["offset"] if boundary["offset"] > 0 else None
        ),
        "status": status,
        "marker": boundary["marker"],
        "majority_start": majority_start,
    }


def filter_to_majority(paragraph_map):
    """
    Drop concurrence and dissent paragraphs from a `{paragraph_index: text}`
    map.
    """
    keys = sorted(paragraph_map, key=int)
    paragraphs = [paragraph_map[key] for key in keys]

    boundary = find_majority_boundary(paragraphs)
    diagnostics = {
        "status": boundary["status"],
        "marker": boundary["marker"],
        "majority_start": boundary["majority_start"],
        "end_index": boundary["end_index"],
        "end_char_offset": boundary["end_char_offset"],
    }

    if boundary["status"] != STATUS_BOUNDARY_FOUND:
        return dict(paragraph_map), diagnostics

    end_index = boundary["end_index"]
    filtered = {keys[i]: paragraphs[i] for i in range(end_index)}

    offset = boundary["end_char_offset"]
    if offset is not None:
        head = paragraphs[end_index][:offset].strip()
        if len(head) >= MIN_BOUNDARY_PARAGRAPH_CHARS:
            filtered[keys[end_index]] = head

    return filtered, diagnostics


# ── Corpus coverage: which cases keep a majority opinion ──────────────────────


def ids_without_majority_opinion(db_path: str, ids: list[int]) -> list[int]:
    """Ids among ``ids`` whose cached majority status is ``no_majority``: rows
    that are a concurrence or dissent rather than the lead opinion."""
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT id, majority_boundary_json FROM cases WHERE id IN ({placeholders})""",
        ids,
    ).fetchall()
    conn.close()
    return [
        row_id
        for row_id, raw in rows
        if raw and json.loads(raw)["status"] == STATUS_NO_MAJORITY
    ]


def load_majority_paragraph_map(
    db_path: str, case_id: int | str, majority_only: bool = True
) -> tuple[dict[int, str], dict[str, Any] | None]:
    """Paragraph map of one case, with concurrence and dissent text removed.
    """
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT opinion_paragraphs_clean_json FROM cases WHERE id = ?",
        (case_id,),
    ).fetchone()
    conn.close()
    paragraph_map = {
        i: p.strip().replace("\n", " ")
        for i, p in enumerate(json.loads(row[0]))
        if p.strip()
    }
    if not majority_only:
        return paragraph_map, None
    return filter_to_majority(paragraph_map)


def isolate_majority_paragraphs(
    db_path: str,
    selected_cases: list[dict[str, Any]],
    majority_only: bool = True,
) -> tuple[dict[int, dict[int, str]], dict[int, str]]:
    """``(paragraph maps, majority status)``, both keyed by case id.
    """
    paragraph_maps = {}
    case_status = {}
    conn = sqlite3.connect(db_path)
    for case in selected_cases:
        paragraph_map, info = load_majority_paragraph_map(
            db_path, case["id"], majority_only
        )
        if info is not None:
            case_status[case["id"]] = info["status"]
            cache_write(conn, "majority_boundary_json", case["id"], info)
        paragraph_maps[case["id"]] = paragraph_map
    conn.close()
    return paragraph_maps, case_status


def store_boundaries(db_path: str, ids: list[int]) -> None:
    """Cache the majority-boundary diagnostics of each case in ``ids``."""
    conn = sqlite3.connect(db_path)
    for case_id in ids:
        _, info = load_majority_paragraph_map(db_path, case_id)
        cache_write(conn, "majority_boundary_json", case_id, info)
    conn.close()


def majority_exclusion(payload: dict[str, Any] | None) -> tuple[str, str]:
    """``(reason, explanation)`` for a case kept out of the tree because
    stage 0 found no lead opinion."""
    detail = "status=no_majority"
    if payload is not None and "marker" in payload and payload["marker"]:
        detail += f", marker={str(payload['marker']).strip()!r}"
    return STATUS_NO_MAJORITY, f"""No majority opinion was identified for this row ({detail}); the case was excluded from dendrogram induction."""
