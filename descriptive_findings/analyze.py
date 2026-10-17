"""Facet and decision-date tests at each interior node of a tree.

At each interior node the parent's cases are split between its two children.
Facet x split asks whether circuit or author party predicts the child
(Fisher exact for 2x2 tables, chi-square with a Monte-Carlo p-value when
sparse). Date x split compares the children's decision dates with a two-sided
Mann-Whitney test. Each family is Benjamini-Hochberg corrected on its own.

    python -m descriptive_findings --doctrine <d> [<d> ...] --run <run>

Writes ``<run_dir>/descriptive_findings/report.{json,csv}``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from descriptive_findings.stats import attach_fdr
from descriptive_findings.stats import mann_whitney
from descriptive_findings.stats import MONTE_CARLO_N
from descriptive_findings.stats import test_table

REPO_ROOT = Path(__file__).resolve().parent.parent

FACETS = ("circuit", "judge_author_party")

#: Author-party values that are not a party: no appointing president (state and
#: Article I courts) or no matched author.
NON_ANSWERS = frozenset({"non_article_iii", "unknown"})

FAMILIES = {
    "facet_x_split": "Facet x split",
    "date_split": "Date x split",
}


def load_run(
    run_dir: Path, doctrine: str
) -> tuple[list[dict], dict[str, dict]]:
    """``(nodes, index)`` from ``corpus_level.json`` and ``case_index.json``;
    ``index`` is keyed on ``case_id``. Each case also gets its
    ``judge_author_party`` from the cleaned corpus (``judge_party --write-db``),
    matched on opinion id."""
    with (run_dir / "corpus_level.json").open() as f:
        payload = json.load(f)
    with (run_dir / "case_index.json").open() as f:
        idx = json.load(f)
    nodes = payload if isinstance(payload, list) else payload["nodes"]
    cases = idx if isinstance(idx, list) else idx["cases"]
    conn = sqlite3.connect(
        REPO_ROOT / "doctrines" / doctrine / "data" / "citing_cases_clean.db"
    )
    party = {
        str(opinion_id): value
        for opinion_id, value in conn.execute(
            "SELECT courtlistener_opinion_id, judge_author_party FROM cases"
        )
    }
    conn.close()
    for case in cases:
        case["judge_author_party"] = party[case["opinion_id"]]
    return nodes, {str(c["case_id"]): c for c in cases}


def facet_of(case: dict, facet: str) -> str | None:
    """The facet's value for a case, or None if it should not be tested."""
    if facet not in case or case[facet] is None:
        return None
    text = str(case[facet]).strip()
    if not text:
        return None
    if facet == "judge_author_party" and text in NON_ANSWERS:
        return None
    return text


def parse_date(text: Any) -> date | None:
    """Leading ``YYYY-MM-DD`` of ``text``; None if missing or too short."""
    if text is None:
        return None
    raw = str(text).strip()
    if len(raw) < 10:
        return None
    return datetime.strptime(raw[:10], "%Y-%m-%d").date()


def interior_splits(
    nodes: list[dict],
) -> list[tuple[dict, dict, dict, list[str], list[str]]]:
    """``(node, left, right, left_ids, right_ids)`` for every interior node
    whose two children both hold cases."""
    by_id = {n["id"]: n for n in nodes}
    out = []
    for node in nodes:
        left_id = node["left_id"] if "left_id" in node else None
        right_id = node["right_id"] if "right_id" in node else None
        if not left_id or not right_id:
            continue
        if left_id not in by_id or right_id not in by_id:
            continue
        left, right = by_id[left_id], by_id[right_id]
        left_ids = [str(c) for c in left["cases"]] if "cases" in left else []
        right_ids = [str(c) for c in right["cases"]] if "cases" in right else []
        if not left_ids or not right_ids:
            continue
        out.append((node, left, right, left_ids, right_ids))
    return out


def _split_meta(node: dict, left: dict, right: dict) -> dict[str, Any]:
    return {
        "node": node["id"],
        "depth": node["depth"] if "depth" in node else 0,
        "label": (
            node["split_criterion_short"]
            if "split_criterion_short" in node and node["split_criterion_short"]
            else (
                node["label"]
                if "label" in node and node["label"]
                else node["id"]
            )
        ),
        "left_id": left["id"],
        "right_id": right["id"],
        "left_label": (
            left["label"] if "label" in left and left["label"] else left["id"]
        )[:40],
        "right_label": (
            right["label"]
            if "label" in right and right["label"]
            else right["id"]
        )[:40],
    }


def split_tests(
    nodes: list[dict],
    index: dict[str, dict],
    *,
    min_cell: int,
    seed: int,
    monte_carlo: int,
) -> list[dict[str, Any]]:
    """For each interior node and facet, does the facet predict the child?"""
    rows: list[dict[str, Any]] = []
    for node, left, right, left_ids, right_ids in interior_splits(nodes):
        for facet in FACETS:
            counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for side, ids in ((0, left_ids), (1, right_ids)):
                for cid in ids:
                    if cid not in index:
                        continue
                    value = facet_of(index[cid], facet)
                    if value is not None:
                        counts[value][side] += 1
            # Drop values too rare to contribute, then require a real table.
            kept = {v: c for v, c in counts.items() if sum(c) >= min_cell}
            if len(kept) < 2:
                continue
            table = [kept[v] for v in sorted(kept)]
            if sum(t[0] for t in table) == 0 or sum(t[1] for t in table) == 0:
                continue
            result = test_table(table, seed=seed, monte_carlo=monte_carlo)
            rows.append(
                {
                    **_split_meta(node, left, right),
                    "id": f"facet_x_split:{node['id']}:{facet}",
                    "facet": facet,
                    "n_values": len(kept),
                    **result,
                    "effect_name": "cramers_v",
                    "effect": result["cramers_v"],
                }
            )
    return rows


def date_tests(
    nodes: list[dict], index: dict[str, dict]
) -> list[dict[str, Any]]:
    """Mann-Whitney of decision dates, left child vs right, at each split."""
    rows: list[dict[str, Any]] = []
    for node, left, right, left_ids, right_ids in interior_splits(nodes):
        samples: list[list[int]] = []
        for ids in (left_ids, right_ids):
            ordinals = []
            for cid in ids:
                if cid not in index or "decision_date" not in index[cid]:
                    continue
                d = parse_date(index[cid]["decision_date"])
                if d is not None:
                    ordinals.append(d.toordinal())
            samples.append(ordinals)
        result = mann_whitney(samples[0], samples[1])
        if result["n1"] == 0 or result["n2"] == 0:
            continue
        medians = [
            (
                date.fromordinal(int(round(result[k]))).isoformat()
                if result[k] is not None
                else None
            )
            for k in ("median1", "median2")
        ]
        rows.append(
            {
                **_split_meta(node, left, right),
                "id": f"date_split:{node['id']}:decision_date",
                "facet": "decision_date",
                **result,
                "n_left": result["n1"],
                "n_right": result["n2"],
                "median_left": medians[0],
                "median_right": medians[1],
                "effect_name": "rank_biserial",
                "effect": result["rank_biserial"],
            }
        )
    return rows


def _pack(family: str, rows: list[dict[str, Any]], alpha: float) -> dict:
    attach_fdr(rows)
    for row in rows:
        row["family"] = family
    return {
        "id": family,
        "label": FAMILIES[family],
        "n_tests": len(rows),
        "n_sig": sum(1 for r in rows if r["q_bh"] < alpha),
        "rows": rows,
    }


def analyze_all(
    run_dir: Path,
    *,
    doctrine: str,
    min_cell: int,
    seed: int,
    monte_carlo: int,
    alpha: float,
) -> dict[str, Any]:
    """Both families on one tree run, each BH-corrected on its own."""
    nodes, index = load_run(run_dir, doctrine)
    splits = split_tests(
        nodes, index, min_cell=min_cell, seed=seed, monte_carlo=monte_carlo
    )
    return {
        "meta": {
            "doctrine": doctrine,
            "run": run_dir.name,
            "n_cases": len(index),
            "alpha": alpha,
            "seed": seed,
            "min_cell": min_cell,
            "monte_carlo": monte_carlo,
        },
        "families": [
            _pack("facet_x_split", splits, alpha),
            _pack("date_split", date_tests(nodes, index), alpha),
        ],
    }


_CSV_FIELDS = (
    "family",
    "id",
    "node",
    "depth",
    "facet",
    "label",
    "n",
    "n_values",
    "chi2",
    "dof",
    "method",
    "p",
    "q_bh",
    "effect_name",
    "effect",
    "left_id",
    "right_id",
    "left_label",
    "right_label",
    "n_tests_in_family",
)


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    """``report.json`` and ``report.csv`` (every row, sorted by p)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    rows = [r for fam in result["families"] for r in fam["rows"]]
    with (out_dir / "report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(_CSV_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["p"]):
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Facet and decision-date tests for a tree run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--doctrine", nargs="+", required=True)
    p.add_argument("--run", required=True, help="Run id.")
    p.add_argument(
        "--min-cell",
        type=int,
        default=5,
        help="Drop facet values with fewer than this many cases at a node.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--monte-carlo",
        type=int,
        default=MONTE_CARLO_N,
        help="Permutations for sparse tables larger than 2x2.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run_dir>/descriptive_findings/.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for doctrine in args.doctrine:
        run_dir = (
            REPO_ROOT / "doctrines" / doctrine / "analysis" / "runs"
        ) / args.run
        result = analyze_all(
            run_dir,
            doctrine=doctrine,
            min_cell=args.min_cell,
            seed=args.seed,
            monte_carlo=args.monte_carlo,
            alpha=args.alpha,
        )
        out_dir = (
            args.output_dir or (run_dir / "descriptive_findings")
        ).resolve()
        write_report(result, out_dir)
        counts = " ".join(
            f"{fam['id']}={fam['n_tests']}/{fam['n_sig']}"
            for fam in result["families"]
        )
        print(f"{doctrine:<14} {run_dir.name:<28} {counts}")
        print(f"    -> {out_dir}/report.{{json,csv}}")
    return 0
