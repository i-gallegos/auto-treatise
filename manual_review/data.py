"""Human annotations and tree placements shared by ``model_label`` and ``irr``."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "manual_review" / "output"
DEFAULT_DB_PATH = DEFAULT_OUT_DIR / "annotations.db"

DOCTRINE_LABELS = {
    "loper_bright": "Loper Bright Enterprises v. Raimondo (agency deference)",
    "bruen": """New York State Rifle & Pistol Association v. Bruen (Second Amendment)""",
}


def doctrine_dir(doctrine: str) -> Path:
    return REPO_ROOT / "doctrines" / doctrine


def cases_db_path(doctrine: str) -> Path:
    return doctrine_dir(doctrine) / "data" / "citing_cases_clean.db"


def runs_dir(doctrine: str) -> Path:
    return doctrine_dir(doctrine) / "analysis" / "runs"


def connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA query_only = ON")
    return conn


@dataclass(frozen=True)
class Quote:
    pages: str = ""
    quote: str = ""


@dataclass(frozen=True)
class SecondarySource:
    citation: str = ""
    quotes: tuple[Quote, ...] = ()


@dataclass(frozen=True)
class CaseAnnotation:
    """The human review of one case's placement in one tree run."""

    review_id: str
    run_id: str
    tree_agreement: str
    doctrine: str = ""
    split: str = ""
    case_name: str = ""
    primary_citation: str = ""
    manual_citation: str = ""
    cl_opinion_id: str = ""
    human_characterization: str = ""
    notes: str = ""
    tree_characterization: str = ""
    sources: tuple[SecondarySource, ...] = ()

    @property
    def effective_citation(self) -> str:
        return self.manual_citation or self.primary_citation


def read_annotations(
    run_id: str, db_path: Path | None = None
) -> list[CaseAnnotation]:
    """Every case the human judged on ``run_id``, ordered by doctrine, split
    and rank."""
    conn = connect_readonly(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cases = conn.execute(
        "SELECT * FROM case_reviews ORDER BY doctrine, split, rank, review_id"
    ).fetchall()
    sources = conn.execute(
        "SELECT * FROM secondary_sources ORDER BY review_id, position, id"
    ).fetchall()
    quotes = conn.execute(
        "SELECT * FROM source_quotes ORDER BY source_id, position, id"
    ).fetchall()
    verdicts = {
        row["review_id"]: row
        for row in conn.execute(
            "SELECT * FROM run_reviews WHERE run_id = ?", (run_id,)
        )
    }
    conn.close()

    quotes_by_source: dict[int, list[Quote]] = {}
    for row in quotes:
        quotes_by_source.setdefault(row["source_id"], []).append(
            Quote(pages=row["pages"] or "", quote=row["quote"] or "")
        )
    sources_by_case: dict[str, list[SecondarySource]] = {}
    for row in sources:
        sources_by_case.setdefault(row["review_id"], []).append(
            SecondarySource(
                citation=row["citation"] or "",
                quotes=(
                    tuple(quotes_by_source[row["id"]])
                    if row["id"] in quotes_by_source
                    else ()
                ),
            )
        )

    out = []
    for row in cases:
        if row["review_id"] not in verdicts:
            continue
        verdict = verdicts[row["review_id"]]
        out.append(
            CaseAnnotation(
                review_id=row["review_id"],
                run_id=run_id,
                tree_agreement=verdict["tree_agreement"] or "",
                doctrine=row["doctrine"] or "",
                split=row["split"] or "",
                case_name=row["case_name"] or "",
                primary_citation=row["primary_citation"] or "",
                manual_citation=row["manual_citation"] or "",
                cl_opinion_id=row["cl_opinion_id"] or "",
                human_characterization=row["human_characterization"] or "",
                notes=row["notes"] or "",
                tree_characterization=verdict["tree_characterization"] or "",
                sources=tuple(
                    sources_by_case[row["review_id"]]
                    if row["review_id"] in sources_by_case
                    else ()
                ),
            )
        )
    return out


@dataclass
class TreeRun:
    doctrine: str
    run_id: str
    nodes: dict[str, dict]
    root_id: str
    opinion_to_run_case: dict[str, str]

    @property
    def path(self) -> Path:
        return runs_dir(self.doctrine) / self.run_id


def load_run(doctrine: str, run_id: str) -> TreeRun:
    run_path = runs_dir(doctrine) / run_id
    corpus = json.loads((run_path / "corpus_level.json").read_text())
    index = json.loads((run_path / "case_index.json").read_text())
    return TreeRun(
        doctrine=doctrine,
        run_id=run_id,
        nodes={node["id"]: node for node in corpus["nodes"]},
        root_id=corpus["root_id"],
        opinion_to_run_case={
            str(case["opinion_id"]): str(case["case_id"])
            for case in index["cases"]
            if case["opinion_id"]
        },
    )


def walk_tree(run: TreeRun, run_case_id: str) -> tuple[list[dict], str | None]:
    """``(steps, leaf_id)`` from the root; ``leaf_id`` is ``None`` when the
    case is unassigned partway down."""
    steps: list[dict] = []
    node = run.nodes[run.root_id]
    while not node["is_leaf"]:
        if run_case_id not in node["assignments"]:
            return steps, None
        record = node["assignments"][run_case_id]
        side = record["side"]
        child_id = node["left_id"] if side == "left" else node["right_id"]
        child = run.nodes[child_id]
        steps.append(
            {
                "depth": len(steps),
                "node_id": node["id"],
                "side": side,
                "label": child["label"],
                "axis": node["split_criterion_short"],
                "why": record["rationale"],
            }
        )
        node = child
    return steps, node["id"]


def assignment_rows(
    doctrine: str,
    records: list[dict],
    run_id: str | None = None,
    run: TreeRun | None = None,
) -> list[dict[str, Any]]:
    """Where each selected case lands in the run: its steps and leaf.

    ``complete`` is false when the case is not in the run or is unassigned
    partway down.
    """
    run = run or load_run(doctrine, run_id)
    rows = []
    for record in records:
        steps, leaf_id = [], None
        if record["cl_opinion_id"] in run.opinion_to_run_case:
            run_case_id = run.opinion_to_run_case[record["cl_opinion_id"]]
            steps, leaf_id = walk_tree(run, run_case_id)
        rows.append(
            {
                "review_id": record["review_id"],
                "complete": leaf_id is not None,
                "leaf_id": leaf_id or "",
                "leaf_label": run.nodes[leaf_id]["label"] if leaf_id else "",
                "steps": steps,
            }
        )
    return rows
