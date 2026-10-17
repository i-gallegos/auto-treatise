"""Shared case loading, cache access and prompt text."""

import json
import sqlite3
from typing import Any, Dict

FORMALIZATION = r"""# FORMALIZATION

## GRAMMAR
Let $\mathcal{C} = \{ c^{[1]}, \ldots, c^{[n]} \}$ be a set of court opinions, each resolving the same legal consequence $L(x)$.

Each $c^{[t]}$ contains a set of propositions $P^{[t]}$.
A \emph{legal proposition} $p \in P$ is an atomic, authoritative legal statement:
\[
p := \forall x ( \phi(x) \lawimplies L(x) )
\]
where $\phi(x)$ represents a condition or constraint that the facts $x$ must satisfy for the legal consequence $L(x)$ to follow. $\lawimplies$ denotes a \emph{legal implication}, meaning that it is non-monotonic. 

Even though courts may purport to reference the same legal constraint, its meaning may be different when we look at how it is applied in different cases. Your job is to understand how the legal constraints are applied and may differ in particular cases.
"""


# TARGET-paragraph character budget for one segmentation request.
SEGMENTATION_TARGET_CHAR_BUDGET = 50_000


def _clean(value: Any) -> str | None:
    """``value`` as stripped text; None for missing or blank."""
    return None if value is None else str(value).strip() or None


def load_cases(
    db_path: str, ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Cases as dicts with id, case_name, citation, opinion_text, opinion_id,
    cluster_id, court_id and decision_date."""
    conn = sqlite3.connect(db_path)
    select = """SELECT id, case_name, citation, opinion_text_clean, courtlistener_opinion_id, courtlistener_cluster_id, court_id, decision_date FROM cases"""
    if ids is None:
        rows = conn.execute(select).fetchall()
    else:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"{select} WHERE id IN ({placeholders})", ids
        ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "case_name": r[1],
            "citation": r[2],
            "opinion_text": r[3],
            "opinion_id": _clean(r[4]),
            "cluster_id": _clean(r[5]),
            "court_id": _clean(r[6]),
            "decision_date": _clean(r[7]),
        }
        for r in rows
    ]


def cache_read(
    conn: sqlite3.Connection,
    column: str,
    case_id: int | str,
) -> Any:
    """JSON-decoded ``column`` of one ``cases`` row, or None if unset."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cases)").fetchall()]
    if column not in cols:
        return None
    row = conn.execute(
        f"SELECT {column} FROM cases WHERE id = ?",
        (case_id,),
    ).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def cache_write(
    conn: sqlite3.Connection,
    column: str,
    case_id: int | str,
    payload: Any,
) -> None:
    """Store ``payload`` as JSON in ``column``, adding the column if missing."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cases)").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE cases ADD COLUMN {column} TEXT")
    conn.execute(
        f"UPDATE cases SET {column} = ? WHERE id = ?",
        (json.dumps(payload, ensure_ascii=False), case_id),
    )
    conn.commit()


def load_segmented_sentences(
    conn: sqlite3.Connection,
    case_id: int,
    cache_column: str = "paragraph_sentence_segmentation_cache_json",
) -> list[dict] | None:
    """
    Read the segmentation cache for one case.
    """
    return cache_read(conn, cache_column, case_id)


def format_segmented_paragraphs(
    segmented_paragraphs: list[dict] | None,
) -> str:
    """Render [{paragraph_index, sentences:[{text,attribution,role,stance}]}]
    as a compact labeled block.
    """
    if not segmented_paragraphs:
        return ""
    paragraphs = [
        para
        for para in segmented_paragraphs
        if isinstance(para, dict) and "paragraph_index" in para
    ]
    lines: list[str] = []
    for para in sorted(paragraphs, key=lambda p: int(p["paragraph_index"])):
        if "sentences" not in para or not isinstance(para["sentences"], list):
            continue
        sentences = para["sentences"]
        if not sentences:
            continue
        lines.append(f"[Paragraph {para['paragraph_index']}]")
        cur_label = (
            sentences[0]["role"],
            sentences[0]["attribution"],
            sentences[0]["stance"],
        )
        cur_texts = [sentences[0]["text"]]
        for s in sentences[1:]:
            label = (s["role"], s["attribution"], s["stance"])
            if label == cur_label:
                cur_texts.append(s["text"])
            else:
                tag = f"[{cur_label[0]} | {cur_label[1]} | {cur_label[2]}]"
                lines.append(f"  {tag} {' '.join(cur_texts)}")
                cur_label = label
                cur_texts = [s["text"]]
        tag = f"[{cur_label[0]} | {cur_label[1]} | {cur_label[2]}]"
        lines.append(f"  {tag} {' '.join(cur_texts)}")
    return "\n".join(lines)


def display_paragraph_map(numbered_paragraphs_map: Dict[int, str]) -> str:
    keys = list(numbered_paragraphs_map.keys())
    return "\n\n".join(
        [f"[{i}] {numbered_paragraphs_map[i]}" for i in sorted(keys, key=int)]
    )
