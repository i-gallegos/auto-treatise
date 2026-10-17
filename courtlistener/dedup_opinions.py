"""Collapse duplicate opinion rows to one row per judicial decision.

Rows are grouped by transitive closure over four keys: cluster id, hash of the
normalized opinion text, (case name, date, court, docket), and (normalized case
name, date, court). Within a group the row kept is the one with the longest
text, then the lowest opinion id.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import re
import sqlite3

MIN_TEXT_CHARS = 500


def normalize_name(name: str) -> str:
    """Lowercase ``name``, with punctuation and whitespace runs as one space."""
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    ).strip()


def load_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = []
    for r in conn.execute(
        """SELECT id, courtlistener_opinion_id, courtlistener_cluster_id, case_name, decision_date, court_id, docket_number, opinion_text_clean FROM cases ORDER BY CAST(id AS INTEGER)"""
    ):
        text = r["opinion_text_clean"] or ""
        rows.append(
            {
                "case_id": str(r["id"]),
                "opinion_id": str(r["courtlistener_opinion_id"] or ""),
                "cluster_id": str(r["courtlistener_cluster_id"] or ""),
                "case_name": r["case_name"] or "",
                "decision_date": r["decision_date"] or "",
                "court_id": r["court_id"] or "",
                "docket_number": (r["docket_number"] or "").strip(),
                "text_len": len(text),
                "text_hash": hashlib.sha1(
                    " ".join(text.split()).lower().encode("utf-8")
                ).hexdigest(),
            }
        )
    conn.close()
    return rows


def group_rows(rows: list[dict]) -> list[list[dict]]:
    parent = {r["case_id"]: r["case_id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def keys(r):
        name = normalize_name(r["case_name"])
        return [
            ("cluster", r["cluster_id"]) if r["cluster_id"] else None,
            (
                ("text", r["text_hash"])
                if r["text_len"] >= MIN_TEXT_CHARS
                else None
            ),
            (
                ("caption", r["case_name"], r["decision_date"], r["court_id"])
                + (r["docket_number"],)
                if r["docket_number"]
                else None
            ),
            (
                ("namedate", name, r["decision_date"], r["court_id"])
                if name and r["decision_date"]
                else None
            ),
        ]

    for i in range(4):
        buckets = defaultdict(list)
        for r in rows:
            key = keys(r)[i]
            if key is not None:
                buckets[key].append(r["case_id"])
        for ids in buckets.values():
            for other in ids[1:]:
                a, b = find(ids[0]), find(other)
                if a != b:
                    low, high = sorted((a, b), key=int)
                    parent[high] = low

    groups = defaultdict(list)
    for r in rows:
        groups[find(r["case_id"])].append(r)
    return list(groups.values())


def rank_key(row: dict) -> tuple:
    return (
        -row["text_len"],
        int(row["opinion_id"]) if row["opinion_id"].isdigit() else 1 << 62,
        int(row["case_id"]),
    )


def surviving_case_ids(db_path: Path) -> set[str]:
    """Case ids that remain when the corpus is collapsed to one row per
    decision."""
    keep = set()
    for members in group_rows(load_rows(db_path)):
        keep.add(min(members, key=rank_key)["case_id"])
    return keep
