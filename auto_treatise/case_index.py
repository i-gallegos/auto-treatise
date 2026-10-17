"""The ``case_id -> opinion_id`` map of a tree run.

Tree induction addresses cases by ``cases.id``, the row id of
``citing_cases_clean.db``, which changes whenever the database is rebuilt. Each
run therefore writes ``case_index.json`` beside ``corpus_level.json``, listing
for every case the run used its CourtListener opinion id, case name, decision
date and court.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

COLUMNS = (
    "id",
    "courtlistener_opinion_id",
    "courtlistener_cluster_id",
    "case_name",
    "decision_date",
    "court_id",
    "court_name",
    "court_abbreviation",
    "court_level",
    "jurisdiction",
    "circuit",
    "precedential_status",
)
RENAME = {
    "id": "case_id",
    "courtlistener_opinion_id": "opinion_id",
    "courtlistener_cluster_id": "cluster_id",
}
ID_COLUMNS = tuple(RENAME)


def build_case_index(
    db_path: str | Path, case_ids: Iterable[Any]
) -> dict[str, list[dict[str, Any]]]:
    """``{"cases": [...]}`` for ``case_ids``, ordered by case id."""
    wanted = {str(case_id) for case_id in case_ids}
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT {', '.join(COLUMNS)} FROM cases ORDER BY CAST(id AS INTEGER)"
    ).fetchall()
    conn.close()
    cases = []
    for row in rows:
        record = {
            RENAME[c] if c in RENAME else c: (
                str(v) if c in ID_COLUMNS and v is not None else v
            )
            for c, v in zip(COLUMNS, row)
        }
        if record["case_id"] in wanted:
            cases.append(record)
    return {"cases": cases}


def write_case_index(index: dict[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "case_index.json"
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return path


def opinion_id_by_case_id(index: dict[str, Any]) -> dict[str, str]:
    """``{case_id: opinion_id}`` for the cases that have one."""
    return {
        row["case_id"]: row["opinion_id"]
        for row in index["cases"]
        if row["opinion_id"]
    }
