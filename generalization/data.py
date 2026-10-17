"""Paths, the tree view, case metadata and result I/O for the generalization task.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

COURT_COLUMNS = (
    "court_id",
    "court_name",
    "court_abbreviation",
    "court_level",
    "jurisdiction",
    "circuit",
)

#: Neighbours the embedding baseline retrieves and stores.
DEFAULT_M = 5


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def doctrine_dir(doctrine: str) -> Path:
    return _ROOT / "doctrines" / doctrine


def db_path(doctrine: str) -> Path:
    return doctrine_dir(doctrine) / "data" / "citing_cases_clean.db"


def analysis_dir(doctrine: str) -> Path:
    return doctrine_dir(doctrine) / "analysis"


def runs_dir(doctrine: str) -> Path:
    return analysis_dir(doctrine) / "runs"


def run_dir(doctrine: str, run_id: str) -> Path:
    return runs_dir(doctrine) / run_id


def output_dir(doctrine: str, run_id: str) -> Path:
    """The run's ``incremental/`` artifact directory, created if missing."""
    path = run_dir(doctrine, run_id) / "incremental"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #


class Tree:
    """Thin, schema-tolerant view over a ``corpus_level.json`` artifact."""

    def __init__(
        self, payload: dict[str, Any], metadata: dict[str, Any] | None = None
    ):
        self.metadata = metadata or {}
        self.root_id: str = payload.get("root_id", "n0")
        self.nodes: dict[str, dict] = {n["id"]: n for n in payload["nodes"]}
        for node in self.nodes.values():
            node["cases"] = [str(c) for c in node.get("cases") or []]

    # -- structure ---------------------------------------------------------- #

    @property
    def root(self) -> dict:
        return self.nodes[self.root_id]

    @property
    def leaves(self) -> list[dict]:
        return [n for n in self.nodes.values() if n.get("is_leaf")]

    def children(self, node_id: str) -> tuple[dict | None, dict | None]:
        """``(left, right)`` child nodes of ``node_id``; ``None``
        where absent."""
        node = self.nodes[node_id]
        left = self.nodes.get(node.get("left_id") or "")
        right = self.nodes.get(node.get("right_id") or "")
        return left, right

    # -- cases -------------------------------------------------------------- #

    @property
    def train_case_ids(self) -> list[str]:
        """``C_<=tau``: the cases the tree was actually induced over."""
        return list(self.root["cases"])

    def case_to_leaf(self) -> dict[str, str]:
        """``{case_id: leaf_id}`` for every training case in a leaf."""
        return {c: leaf["id"] for leaf in self.leaves for c in leaf["cases"]}

    # -- rendering ---------------------------------------------------------- #

    def describe_node(self, node_id: str) -> str:
        """Render an internal node's split for a routing prompt.
        """
        node = self.nodes[node_id]
        left, right = self.children(node_id)
        parts = [
            f"Current interpretation of the constraint:\n{node.get('description', '')}"
        ]
        if node.get("axis_type"):
            parts.append(f"Axis of distinction: {node['axis_type']}")
        if node.get("split_criterion"):
            parts.append(f"Split criterion:\n{node['split_criterion']}")
        if node.get("decision_rule"):
            parts.append(f"Decision rule:\n{node['decision_rule']}")
        left, right = left or {}, right or {}
        parts.append(
            "LEFT — "
            + (left.get("label") or node.get("left_label") or "left branch")
            + f"\n{node.get('left_description') or left.get('description', '')}"
        )
        parts.append(
            "RIGHT — "
            + (right.get("label") or node.get("right_label") or "right branch")
            + f"\n{node.get('right_description') or right.get('description', '')}"
        )
        return "\n\n".join(p for p in parts if p)


def load_tree(doctrine: str, run_id: str) -> Tree:
    """Load ``corpus_level.json`` and ``tree_run_metadata.json``."""
    directory = run_dir(doctrine, run_id)
    payload = json.loads((directory / "corpus_level.json").read_text())
    meta_file = directory / "tree_run_metadata.json"
    metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return Tree(payload, metadata)


# --------------------------------------------------------------------------- #
# Case profiles and opinions
# --------------------------------------------------------------------------- #


def load_case_profiles(
    doctrine: str, path: str | Path | None = None
) -> dict[str, dict]:
    """``{case_id: profile}`` from ``case_level.json``, tolerating
    both key names."""
    path = Path(path) if path else analysis_dir(doctrine) / "case_level.json"
    if not path.exists():
        return {}
    profiles: dict[str, dict] = {}
    for row in json.loads(path.read_text()):
        case_id = row.get("case_id", row.get("id"))
        if case_id is not None:
            profiles[str(case_id)] = row.get("profile") or {}
    return profiles


def _connect(doctrine: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path(doctrine))


def _columns(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(cases)").fetchall()]


METADATA_COLUMNS = (
    "case_name",
    "citation",
    "decision_date",
    "court",
    "courtlistener_cluster_id",
    "precedential_status",
    "citation_count",
) + COURT_COLUMNS


def case_metadata(
    doctrine: str, case_ids: list[str] | None = None
) -> dict[str, dict]:
    """``{case_id: {case_name, citation, decision_date, court, court_*}}``.
    """
    conn = _connect(doctrine)
    available = _columns(conn)
    wanted = [c for c in METADATA_COLUMNS if c in available]
    rows = conn.execute(f"SELECT id, {', '.join(wanted)} FROM cases").fetchall()
    conn.close()
    out = {str(r[0]): dict(zip(wanted, r[1:])) for r in rows}
    if case_ids is None:
        return out
    keep = {str(c) for c in case_ids}
    return {k: v for k, v in out.items() if k in keep}


def _sort_ids(ids) -> list[str]:
    return sorted(
        ids, key=lambda c: (0, int(c)) if str(c).isdigit() else (1, str(c))
    )


def accepted_ids(doctrine: str, tree: Tree, case_ids: list[str]) -> list[str]:
    """Requested ids that are in the corpus and not in the training set."""
    train = set(tree.train_case_ids)
    conn = _connect(doctrine)
    known = {str(r[0]) for r in conn.execute("SELECT id FROM cases")}
    conn.close()
    requested = dict.fromkeys(str(c) for c in case_ids)
    return _sort_ids(c for c in requested if c in known and c not in train)


# --------------------------------------------------------------------------- #
# x^[u]: the fact-only view
# --------------------------------------------------------------------------- #


def fact_views(doctrine: str, case_ids: list[str]) -> dict[str, str]:
    """``{case_id: x^[u]}`` rendered from the ``views`` cache; unextracted
    cases are dropped."""
    from . import views as case_views  # deferred: views imports data

    return case_views.issue_views(doctrine, case_ids)


def majority_views(doctrine: str, case_ids: list[str]) -> dict[str, str]:
    """Batch majority-opinion text (no relevance or segmentation).
    Empty dropped."""
    from . import views as case_views  # deferred: views imports data

    conn = _connect(doctrine)
    try:
        views = {c: case_views.source_text(conn, c) for c in case_ids}
    finally:
        conn.close()
    return {c: v for c, v in views.items() if v.strip()}


def retrieval_pool(
    doctrine: str, tree: Tree
) -> tuple[list[str], dict[str, str]]:
    """Training cases the embedding baseline can retrieve over.
    """
    from . import views as case_views  # deferred: views imports data

    case_to_leaf = tree.case_to_leaf()
    views = case_views.distilled_views(doctrine, list(case_to_leaf))
    pool = [c for c in case_to_leaf if views.get(c, "").strip()]
    return pool, views


# --------------------------------------------------------------------------- #
# Reading and writing results
# --------------------------------------------------------------------------- #
#: Gold labels and scoring use the annotator's segmented profile.
LABEL_VIEW = "segmented_profile"
LABELS_FILE = f"test_labels.{LABEL_VIEW}.json"
PREDICTION_FILES = {
    "tree": "test_preds_tree.json",
    "embedding_nn": "test_preds_embed.json",
    "forced": "test_preds_forced_segment.json",
    "factor": "test_preds_factor_segment.json",
}
FORCED_CASES_FILE = "test_forced_cases_segment.json"
SUMMARY_FILE = "generalization_summary_segment_95.json"


def labels_path(doctrine: str, run_id: str) -> Path:
    return output_dir(doctrine, run_id) / LABELS_FILE


def predictions_path(doctrine: str, run_id: str, method: str) -> Path:
    return output_dir(doctrine, run_id) / PREDICTION_FILES[method]


def write_json(path: Path, payload: Any) -> Path:
    """Write strict JSON with a trailing newline.

    ``allow_nan=False`` on purpose. Python's encoder happily emits bare ``NaN``
    and ``Infinity``, which no strict parser accepts, so a non-finite value has
    to fail here, loudly, rather than land on disk.
    """
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object, or ``None`` if ``path`` does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def case_header(meta: dict) -> str:
    """One line naming the case, its cite, the deciding court, and the date.
    """
    citation = meta.get("citation")
    if isinstance(citation, str) and citation.startswith("["):
        try:
            citation = ", ".join(json.loads(citation))
        except json.JSONDecodeError:
            pass
    court = (
        meta.get("court_abbreviation")
        or meta.get("court_name")
        or meta.get("court")
    )
    bits = [meta.get("case_name"), citation, court, meta.get("decision_date")]
    return " | ".join(str(b) for b in bits if b)
