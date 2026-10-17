"""The ``citation_graph.json`` payload the UI draws the chronology from.

``u -> v`` means "u cites v". Nodes are the corpus cases, placed by decision
date (x) and court authority (y). Links are the citations between two corpus
cases.
"""

import datetime
import json
from pathlib import Path
import random
import sqlite3
from typing import Any

from courtlistener import courts

PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2",
    "#edc948", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
    "#8cd17d", "#fabfd2", "#b6992d", "#499894",
]
RANK_LABELS = {
    4: "Supreme Court",
    3: "Federal appellate court\nState supreme court",
    2: "Federal district court\nState appellate court",
    1: "State trial court",
    0: "Other",
}
LAYOUT_SEED = 42

# Bankruptcy courts share authority rank 1 with state trial courts. The
# chronology draws them in Other, since they are not state trial courts.
CHRONOLOGY_OTHER_LEVELS = frozenset({"federal_bankruptcy"})

CASE_COLUMNS = (
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


def load_cases(db_path: str | Path) -> list[dict[str, Any]]:
    """Reads the corpus cases that have an opinion id, one per opinion."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT {', '.join(CASE_COLUMNS)} FROM cases ORDER BY id"
    ).fetchall()
    conn.close()
    cases = {}
    for row in rows:
        if row["courtlistener_opinion_id"] is None:
            continue
        opinion_id = int(row["courtlistener_opinion_id"])
        if opinion_id not in cases:
            cases[opinion_id] = {
                **dict(row),
                "opinion_id": opinion_id,
                "date": (
                    datetime.date.fromisoformat(row["decision_date"][:10])
                    if row["decision_date"]
                    else None
                ),
            }
    return list(cases.values())


def _authority_rank(case: dict[str, Any]) -> int:
    level = case["court_level"]
    return courts.AUTHORITY_RANK[level] if level in courts.AUTHORITY_RANK else 0


def _display_rank(case: dict[str, Any]) -> int:
    """Rank for the chronology's y axis, not for stare decisis."""
    if case["court_level"] in CHRONOLOGY_OTHER_LEVELS:
        return 0
    return _authority_rank(case)


def _time_layout(
    cases: list[dict[str, Any]],
) -> tuple[dict[int, tuple[float, float]], dict[str, Any]]:
    """Places cases by decision date (x) and authority rank (y, banded)."""
    dates = [case["date"] for case in cases if case["date"]]
    first = min(dates) if dates else datetime.date(2000, 1, 1)
    last = max(dates) if dates else datetime.date(2001, 1, 1)
    span = max((last - first).days, 1)

    ranks = sorted({_display_rank(case) for case in cases}, reverse=True)
    band_of = {rank: i for i, rank in enumerate(ranks)}
    n_bands = max(len(ranks), 1)
    rng = random.Random(LAYOUT_SEED)

    positions = {}
    for case in cases:
        x = (case["date"] - first).days / span if case["date"] else 0.5
        band = band_of[_display_rank(case)]
        y = (band + 0.28 + 0.44 * rng.random()) / n_bands
        positions[case["opinion_id"]] = (
            round(0.02 + 0.96 * x, 5),
            round(0.04 + 0.92 * y, 5),
        )
    axis = {
        "date_min": str(first),
        "date_max": str(last),
        "bands": [
            {
                "rank": rank,
                "label": RANK_LABELS[rank],
                "y0": i / n_bands,
                "y1": (i + 1) / n_bands,
            }
            for i, rank in enumerate(ranks)
        ],
    }
    return positions, axis


def load_tree(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Reads a run's ``corpus_level.json`` as ``{node id: node}``."""
    with open(Path(run_dir) / "corpus_level.json") as f:
        return {node["id"]: node for node in json.load(f)["nodes"]}


def _case_leaves(tree: dict[str, dict[str, Any]]) -> dict[str, str]:
    """``{case id: id of the leaf holding it}``."""
    return {
        str(case_id): node["id"]
        for node in tree.values()
        if node["is_leaf"]
        for case_id in node["cases"]
    }


def _path(tree: dict[str, dict[str, Any]], node_id: str) -> list[str]:
    """Node ids from the root to ``node_id``."""
    path = [node_id]
    while tree[path[0]]["parent_id"]:
        path.insert(0, tree[path[0]]["parent_id"])
    return path


def build_payload(
    cases: list[dict[str, Any]],
    edges: dict[int, dict[int, int]],
    tree: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Builds the payload for the corpus ``cases`` and a tree run.

    Args:
        cases: Corpus cases from ``load_cases``.
        edges: Outbound citations of each case, from ``edges.fetch_edges``.
        tree: A run's nodes from ``load_tree``.

    Returns:
        The JSON-serializable payload: ``nodes``, ``links``, ``leaves``,
        ``tree``, ``time_axis`` and ``time_axis_tree``.
    """
    cases = sorted(cases, key=lambda case: case["opinion_id"])
    by_opinion = {case["opinion_id"]: case for case in cases}
    index = {case["opinion_id"]: i for i, case in enumerate(cases)}

    case_leaves = _case_leaves(tree)
    leaf_of = {
        case["opinion_id"]: (
            case_leaves[str(case["id"])] if str(case["id"]) in case_leaves
            else None
        )
        for case in cases
    }
    training = [case for case in cases if leaf_of[case["opinion_id"]]]
    positions, time_axis = _time_layout(cases)
    tree_positions, time_axis_tree = (
        _time_layout(training) if training else ({}, None)
    )

    links = []
    for case in cases:
        citing = case["opinion_id"]
        for cited in sorted(edges[citing]):
            if cited not in by_opinion or cited == citing:
                continue
            other = by_opinion[cited]
            rank_gap = _authority_rank(other) - _authority_rank(case)
            links.append(
                {
                    "s": index[citing],
                    "t": index[cited],
                    "d": edges[citing][cited],
                    # The citation crosses interpretive lines.
                    "x": bool(
                        leaf_of[citing]
                        and leaf_of[cited]
                        and leaf_of[citing] != leaf_of[cited]
                    ),
                    "v": (
                        "up" if rank_gap > 0
                        else "down" if rank_gap < 0
                        else "lateral"
                    ),
                    # The cited case is vertically binding on the citing one.
                    "b": courts.binds(other, case),
                    # The cited case was decided after the citing case.
                    "a": bool(
                        case["date"]
                        and other["date"]
                        and other["date"] > case["date"]
                    ),
                }
            )
    in_degree = [0] * len(cases)
    out_degree = [0] * len(cases)
    for link in links:
        out_degree[link["s"]] += 1
        in_degree[link["t"]] += 1

    nodes = []
    for i, case in enumerate(cases):
        opinion_id = case["opinion_id"]
        leaf = leaf_of[opinion_id]
        in_tree = opinion_id in tree_positions
        nodes.append(
            {
                "i": i,
                "opinion_id": str(opinion_id),
                "case_id": str(case["id"]),
                "cluster_id": (
                    None if case["courtlistener_cluster_id"] is None
                    else str(case["courtlistener_cluster_id"])
                ),
                "name": case["case_name"] or f"opinion {opinion_id}",
                "date": str(case["date"]) if case["date"] else None,
                "court": case["court_abbreviation"] or case["court_id"],
                "court_id": case["court_id"],
                "court_name": case["court_name"],
                "court_level": case["court_level"],
                "court_system": courts.court_system(case["court_level"]),
                "jurisdiction": case["jurisdiction"],
                "circuit": case["circuit"],
                "rank": _display_rank(case),
                "status": case["precedential_status"],
                "leaf": leaf,
                "path": _path(tree, leaf) if leaf else [],
                "out_deg": out_degree[i],
                "in_deg": in_degree[i],
                "n_cited_total": len(edges[opinion_id]),
                "tx": positions[opinion_id][0],
                "ty": positions[opinion_id][1],
                "tx_tree": tree_positions[opinion_id][0] if in_tree else None,
                "ty_tree": tree_positions[opinion_id][1] if in_tree else None,
            }
        )

    leaf_ids = sorted({leaf for leaf in leaf_of.values() if leaf})
    return {
        "time_axis": time_axis,
        "time_axis_tree": time_axis_tree,
        "leaves": [
            {"id": leaf, "color": PALETTE[i % len(PALETTE)]}
            for i, leaf in enumerate(leaf_ids)
        ],
        "tree": {
            "nodes": [
                {
                    "id": node["id"],
                    "parent": node["parent_id"],
                    "children": (
                        [] if node["is_leaf"]
                        else [node["left_id"], node["right_id"]]
                    ),
                    "depth": node["depth"],
                    "leaf": node["is_leaf"],
                    "label": node["label"],
                }
                for node in tree.values()
            ],
        },
        "nodes": nodes,
        "links": links,
    }
