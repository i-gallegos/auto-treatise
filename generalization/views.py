"""``x^[u]``: a constraint-conditioned view of a held-out case.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import api_calls  # noqa: E402

from . import data  # noqa: E402
from . import prompts  # noqa: E402
from . import schemas  # noqa: E402

CACHE_COLUMN = "constraint_fact_view_cache_json"
RELEVANCE_COLUMN = "paragraph_relevance_cache_json"
PARAGRAPHS_COLUMN = "opinion_paragraphs_clean_json"
MAJORITY_COLUMN = "majority_boundary_json"

#: Characters of opinion text sent to the extractor. Opinions run past 300k
#: characters; beyond this the middle is elided rather than the tail, because
#: the question presented is as often stated at the end of a long recitation as
#: at the start.
CHAR_BUDGET = 300_000
_ELISION = "\n\n[... middle of the opinion omitted for length ...]\n\n"

#: The ``constraint_engagement`` value that removes a case from the evaluation
#: entirely, rather than describing how it should be read.
NOT_ENGAGED = "not_engaged"


def read_cached(conn: sqlite3.Connection, case_id: str) -> dict[str, Any] | None:
    """The cached payload for one case, or ``None`` if never extracted."""
    row = conn.execute(
        f"SELECT {CACHE_COLUMN} FROM cases WHERE id = ?", (int(case_id),)
    ).fetchone()
    return json.loads(row[0]) if row[0] else None


def _majority_paragraphs(conn: sqlite3.Connection, case_id: str) -> list[str]:
    """The majority opinion's paragraphs, separate opinions dropped.
    """
    row = conn.execute(
        f"SELECT {PARAGRAPHS_COLUMN}, {MAJORITY_COLUMN} FROM cases WHERE id = ?",
        (int(case_id),),
    ).fetchone()
    paragraphs = [p for p in json.loads(row[0]) if p.strip()]
    info = json.loads(row[1])
    return paragraphs[info["majority_start"] : info["end_index"]] or paragraphs


def source_text(conn: sqlite3.Connection, case_id: str) -> str:
    """The opinion text the extractor reads, elided in the middle
    if oversized."""
    text = "\n\n".join(_majority_paragraphs(conn, case_id)).strip()
    if len(text) <= CHAR_BUDGET:
        return text
    half = (CHAR_BUDGET - len(_ELISION)) // 2
    return text[:half] + _ELISION + text[-half:]


def distilled_views(doctrine: str, case_ids: list[str]) -> dict[str, str]:
    """The distilled majority opinion for each case; empty views are dropped.
    """
    conn = data._connect(doctrine)
    views = {}
    for case_id in case_ids:
        row = conn.execute(
            f"SELECT {RELEVANCE_COLUMN} FROM cases WHERE id = ?",
            (int(case_id),),
        ).fetchone()
        kept = sorted(
            json.loads(row[0])["relevant_paragraphs"],
            key=lambda r: r["paragraph_index"],
        )
        texts = [r["text"].strip() for r in kept if r["text"].strip()]
        views[case_id] = "\n\n".join(
            texts or _majority_paragraphs(conn, case_id)
        )
    conn.close()
    return {c: v for c, v in views.items() if v.strip()}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_SECTIONS = (
    ("question_presented", "QUESTION PRESENTED)"),
    ("procedural_posture", "PROCEDURAL POSTURE"),
    ("material_facts", "FACTS"),
)


def render(view: dict[str, Any]) -> str:
    """The cached structured view as the block a routing prompt shows."""
    parts: list[str] = []
    for key, heading in _SECTIONS:
        value = str(view[key]).strip()
        if value:
            parts.append(f"## {heading}\n\n{value}")
    lines = [
        f"- **{t['party'].strip() or 'A party'}**: {t['theory'].strip()}"
        for t in view["competing_theories"]
        if t["theory"].strip()
    ]
    if lines:
        parts.append(
            """## RULES THE PARTIES ASKED FOR

""" + "\n".join(lines)
        )
    engagement = view["constraint_engagement"].strip()
    if engagement in _ENGAGEMENT_GLOSS:
        parts.append("## ENGAGEMENT\n\n" + _ENGAGEMENT_GLOSS[engagement])
    return "\n\n".join(parts)


_ENGAGEMENT_GLOSS = {
    "squarely": """The constraint under study is squarely presented: resolving this case requires the court to say what the constraint means or how it applies.""",
    "incidentally": """The constraint under study is raised but is not what decides the case. The court touches it in passing or reaches it as an alternative.""",
    "assumed": """The constraint under study is treated as settled background here. The parties do not contest its content; it frames a different dispute.""",
}


def issue_views(doctrine: str, case_ids: list[str]) -> dict[str, str]:
    """``x^[u]`` for each case, rendered from the cache; unextracted cases
    are dropped."""
    conn = data._connect(doctrine)
    payloads = {c: read_cached(conn, c) for c in case_ids}
    conn.close()
    views = {c: render(p["view"]) for c, p in payloads.items() if p}
    return {c: v for c, v in views.items() if v.strip()}


def partition_engaged(
    doctrine: str, case_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Split ``case_ids`` into ``(engaged, not_engaged)``.

    This is a **corpus filter, not a prediction**. It runs once, off the full
    opinion, and it removes the case from the run for every method at once.
    Otherwise, the tree could say``not_engaged`` while the embedding
    baseline embeds an opinion with no such marker.
    """
    conn = data._connect(doctrine)
    payloads = {c: read_cached(conn, c) for c in case_ids}
    conn.close()
    dropped = [
        c
        for c, p in payloads.items()
        if p is not None
        and p["view"]["constraint_engagement"].strip() == NOT_ENGAGED
    ]
    return [c for c in case_ids if c not in dropped], dropped


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract(
    doctrine: str, case_ids: list[str], model: str, constraint: str
) -> dict[str, int]:
    """Extract and cache ``x^[u]`` for ``case_ids``.
    """
    ids = list(dict.fromkeys(str(c) for c in case_ids))
    meta = data.case_metadata(doctrine, ids)

    conn = data._connect(doctrine)
    if CACHE_COLUMN not in data._columns(conn):
        conn.execute(f"ALTER TABLE cases ADD COLUMN {CACHE_COLUMN} TEXT")
    todo = [c for c in ids if read_cached(conn, c) is None]
    sources = {c: source_text(conn, c) for c in todo}
    print(
        f"""[extract] {len(ids)} requested, {len(ids) - len(todo)} cached, {len(todo)} to run"""
    )
    if not todo:
        conn.close()
        return {"n_written": 0, "n_cached": len(ids)}

    items = []
    for case_id in todo:
        parts = prompts.build_extraction_prompt(
            constraint, data.case_header(meta[case_id]), sources[case_id]
        )
        items.append(
            {
                "content": parts.suffix,
                "system_prompt": prompts.EXTRACTOR_SYSTEM,
                "cache_prefix": parts.prefix,
            }
        )

    responses = api_calls.batch_query_llm(
        items,
        model=model,
        response_format=schemas.fact_view_schema(),
        temperature=1,
        max_completion_tokens=8192,
    )
    for case_id, (_, view) in zip(todo, responses):
        if not view["question_presented"].strip():
            raise ValueError(f"case {case_id}: empty question_presented")
        payload = {
            "constraint": constraint,
            "model": model,
            "n_source_chars": len(sources[case_id]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "view": view,
        }
        conn.execute(
            f"UPDATE cases SET {CACHE_COLUMN} = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), int(case_id)),
        )
        conn.commit()
    conn.close()
    return {"n_written": len(todo), "n_cached": len(ids) - len(todo)}
