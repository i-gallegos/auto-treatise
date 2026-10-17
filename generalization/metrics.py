"""Scoring for teacher-forced and pooled accuracy with bootstrap CIs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Sequence

import numpy as np

from . import data


def weighted_distribution(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Normalize ``(node, weight)`` pairs into a distribution,
    heaviest first."""
    totals: dict[str, float] = {}
    for node, weight in pairs:
        totals[node] = totals.get(node, 0.0) + max(weight, 0.0)
    mass = sum(totals.values())
    if mass <= 0:
        return {}
    return dict(
        sorted(
            ((k, v / mass) for k, v in totals.items()), key=lambda kv: -kv[1]
        )
    )


def leaf_distribution(leaf_ids: list[str]) -> dict[str, float]:
    """Empirical frequency over leaves from ``k`` independent samples."""
    return weighted_distribution([(leaf, 1.0) for leaf in leaf_ids])


EMBED_MODEL = "text-embedding-3-large"

#: Depth of the deepest leaves
MAX_DEPTH = 5

N_BOOT = 1000
SEED = 0


def summary_output_dir(run_id: str) -> Path:
    """``analysis/generalization/<run_id>/``."""
    return (
        Path(__file__).resolve().parent.parent
        / "analysis"
        / "generalization"
        / run_id
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _subtree_leaves(tree: data.Tree, node_id: str) -> set[str]:
    node = tree.nodes[node_id]
    if node.get("is_leaf"):
        return {node_id}
    return _subtree_leaves(tree, node["left_id"]) | _subtree_leaves(
        tree, node["right_id"]
    )


def _vectors(doctrine: str, views: dict[str, str]) -> dict[str, np.ndarray]:
    """Unit-normalized cached embeddings for each case's view text."""
    conn = sqlite3.connect(
        data.doctrine_dir(doctrine) / "data" / "embeddings.db"
    )
    out = {}
    for case_id, text in views.items():
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE case_id=? AND model=? AND text_hash=?",
            (
                case_id,
                EMBED_MODEL,
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ),
        ).fetchone()
        if row is None:
            raise SystemExit(
                f"[tf] no cached embedding for {doctrine}/{case_id}"
            )
        v = np.frombuffer(row[0], dtype=np.float32)
        out[case_id] = v / max(np.linalg.norm(v), 1e-12)
    conn.close()
    return out


def case_rows(doctrine: str, run_id: str) -> list[dict]:
    """One row per scored case: teacher-forced decisions for both methods."""
    tree = data.load_tree(doctrine, run_id)
    labels = {
        r["case_id"]: r
        for r in _json(data.labels_path(doctrine, run_id))["labels"]
    }
    forced_path = data.predictions_path(doctrine, run_id, "forced")
    if not forced_path.exists():
        raise SystemExit(f"[tf] no {forced_path.name}; run --stage forced")
    forced = {r["case_id"]: r for r in _json(forced_path)["predictions"]}
    embed = {
        r["case_id"]: r
        for r in _json(data.predictions_path(doctrine, run_id, "embedding_nn"))[
            "predictions"
        ]
    }
    ids = sorted(
        set(labels) & set(forced) & set(embed), key=lambda c: (len(c), c)
    )

    case_to_leaf = tree.case_to_leaf()
    pool, train_views = data.retrieval_pool(doctrine, tree)
    train = _vectors(doctrine, {c: train_views[c] for c in pool})
    held = _vectors(doctrine, data.fact_views(doctrine, ids))
    train_ids = list(train)
    train_mat = np.stack([train[c] for c in train_ids])

    # Guard: the unconstrained top-1 must reproduce the stored baseline.
    for c in ids:
        top = train_ids[int(np.argmax(train_mat @ held[c]))]
        assert top == embed[c]["neighbours"][0]["case_id"], (doctrine, c)

    # Every gold decision needs a forced answer: the oracle-prefix panel counts
    # a missing one as wrong, so a gap would read as a lower accuracy.
    gaps = [
        (c, step["node_id"])
        for c in ids
        for step in labels[c].get("path") or []
        if step.get("side")
        and not (tree.nodes.get(step["node_id"]) or {}).get("is_leaf")
        and step["node_id"]
        not in {d["node_id"] for d in forced[c]["decisions"]}
    ]
    if gaps:
        raise SystemExit(
            f"""[tf] {doctrine}: {len(gaps)} gold decision(s) have no forced prediction in {forced_path.name} (e.g. {gaps[:3]}); run --stage forced"""
        )

    rows = []
    for c in ids:
        by_node = {d["node_id"]: d for d in forced[c]["decisions"]}
        decisions = []
        for step in labels[c].get("path") or []:
            node = tree.nodes.get(step["node_id"])
            if not node or node.get("is_leaf") or not step.get("side"):
                continue
            left = _subtree_leaves(tree, node["left_id"])
            right = _subtree_leaves(tree, node["right_id"])
            cand = [
                i
                for i, t in enumerate(train_ids)
                if case_to_leaf[t] in left | right
            ]
            sims = train_mat[cand] @ held[c]
            pick = int(np.argmax(sims))
            best = train_ids[cand[pick]]
            nn_side = "left" if case_to_leaf[best] in left else "right"
            f = by_node.get(node["id"]) or {}
            decisions.append(
                {
                    "depth": int(node["depth"]),
                    "at": (
                        None if f.get("correct") is None else bool(f["correct"])
                    ),
                    "nn": nn_side == step["side"],
                    # Carried for ``write_forced_cases``; ignored by the
                    # summaries.
                    "node_id": node["id"],
                    "gold_side": step["side"],
                    "at_side": f.get("side"),
                    "nn_side": nn_side,
                    "nn_neighbour": {
                        "case_id": best,
                        "leaf": case_to_leaf[best],
                        "similarity": float(sims[pick]),
                    },
                }
            )
        rows.append({"case_id": c, "decisions": decisions})
    return rows


def write_forced_cases(doctrine: str, rows: list[dict], run_id: str) -> Path:
    """Per-case teacher-forced decisions for every method.
    """
    out_path = data.output_dir(doctrine, run_id) / data.FORCED_CASES_FILE
    payload = {
        "meta": {
            "doctrine": doctrine,
            "tree_run_id": run_id,
            "embed_model": EMBED_MODEL,
            "n_cases": len(rows),
        },
        "cases": [
            {
                "case_id": row["case_id"],
                "has_circuit": row.get("has_circuit"),
                "has_party": row.get("has_party"),
                "decisions": [
                    {
                        "depth": d["depth"],
                        "node_id": d.get("node_id"),
                        "gold_side": d.get("gold_side"),
                        "at": {
                            "side": d.get("at_side"),
                            "correct": d.get("at"),
                        },
                        "nn": {
                            "side": d.get("nn_side"),
                            "correct": d.get("nn"),
                            "neighbour": d.get("nn_neighbour"),
                        },
                        "circuit": d.get("circuit"),
                        "party": d.get("party"),
                    }
                    for d in row["decisions"]
                ],
            }
            for row in rows
        ],
    }
    data.write_json(out_path, payload)
    return out_path


def attach_factors(doctrine: str, rows: list[dict], run_id: str) -> None:
    """Copy factor correctness onto the teacher-forced decision rows."""
    path = data.predictions_path(doctrine, run_id, "factor")
    if not path.exists():
        raise SystemExit(f"[tf] no {path.name}; run --stage factor" + " first")
    payload = _json(path)
    by_id = {row["case_id"]: row for row in payload["predictions"]}
    missing = [row["case_id"] for row in rows if row["case_id"] not in by_id]
    if missing:
        raise SystemExit(
            f"[tf] {doctrine}: factor predictions missing {len(missing)} scored case(s)"
        )
    for row in rows:
        pred = by_id[row["case_id"]]
        row["has_circuit"] = (
            bool(pred["circuit"]) if "circuit" in pred else False
        )
        row["has_party"] = bool(pred["party"]) if "party" in pred else False
        by_depth = {
            int(decision["depth"]): decision for decision in pred["decisions"]
        }
        for decision in row["decisions"]:
            if decision["depth"] not in by_depth:
                raise SystemExit(
                    f"[tf] {doctrine}/{row['case_id']} has no factor decision at depth {decision['depth']}"
                )
            src = by_depth[decision["depth"]]
            for name in ("circuit", "party"):
                block = src[name] if name in src else None
                if block is None or block["correct"] is None:
                    decision[name] = None
                else:
                    decision[name] = bool(block["correct"])


#: Series of the pooled summary: Auto-Treatise, embedding NN, circuit, judge
#: appointing party.
SUMMARY_METHODS = ("at", "nn", "circuit", "party")
FEATURE_FLAG = {"circuit": "has_circuit", "party": "has_party"}


def _counts(rows: list[dict], method: str, leaf: bool = False) -> np.ndarray:
    """``[[n_correct, n_decisions], ...]`` per case for one method.

    With ``leaf=True`` each case is ``[1, 1]`` if every decision is correct and
    ``[0, 1]`` otherwise, so the pooled ratio is leaf accuracy with no oracle.
    """
    flag = FEATURE_FLAG.get(method)
    out = []
    for row in rows:
        if flag and not row[flag]:
            continue
        vals = [
            x[method] for x in row["decisions"] if x.get(method) is not None
        ]
        if leaf:
            out.append((float(all(bool(v) for v in vals)), 1.0))
        else:
            out.append((sum(bool(v) for v in vals), len(vals)))
    return np.array(out, dtype=float).reshape(-1, 2)


def pooled(
    rows_by_doctrine: dict[str, list[dict]],
    rng: np.random.Generator,
    leaf: bool = False,
) -> dict:
    """Pooled per-case (or, with ``leaf``, leaf) accuracy for each method.

    SEs and 95% CIs resample cases within each doctrine.
    """
    out = {}
    for m in SUMMARY_METHODS:
        per_doc = [_counts(rows, m, leaf) for rows in rows_by_doctrine.values()]
        total = np.sum([c.sum(axis=0) for c in per_doc], axis=0)
        reps = []
        for _ in range(N_BOOT):
            s = np.zeros(2)
            for c in per_doc:
                s += c[rng.integers(0, len(c), len(c))].sum(axis=0)
            reps.append(s[0] / s[1])
        out[m] = {
            "accuracy": float(total[0] / total[1]),
            "se": float(np.std(reps)),
            "ci95": [float(v) for v in np.percentile(reps, [2.5, 97.5])],
            "n_decisions": int(total[1]),
            "n_cases": int(sum(len(c) for c in per_doc)),
        }
    return out


def pooled_prefix(
    rows_by_doctrine: dict[str, list[dict]],
    rng: np.random.Generator,
    methods: tuple[str, ...],
) -> dict:
    """Leaf accuracy (d = 5) given ``j`` oracle decisions, pooled over
    doctrines."""
    out = {}
    for m in methods:
        flag = FEATURE_FLAG.get(m)
        per_doc = []
        for rows in rows_by_doctrine.values():
            mat = [
                [
                    float(
                        all(
                            bool(x[m])
                            for x in r["decisions"]
                            if x["depth"] >= j and m in x
                        )
                    )
                    for j in range(MAX_DEPTH)
                ]
                for r in rows
                if not flag or r[flag]
            ]
            per_doc.append(np.array(mat, dtype=float).reshape(-1, MAX_DEPTH))
        allm = np.concatenate(per_doc)
        reps = np.stack(
            [
                np.concatenate(
                    [c[rng.integers(0, len(c), len(c))] for c in per_doc]
                ).mean(axis=0)
                for _ in range(N_BOOT)
            ]
        )
        out[m] = {
            "mean": allm.mean(axis=0).tolist(),
            "se": reps.std(axis=0).tolist(),
            "ci95": np.percentile(reps, [2.5, 97.5], axis=0).tolist(),
            "n_cases": int(len(allm)),
        }
    return out


def summary_inputs_missing(doctrines: Sequence[str], run_id: str) -> list[Path]:
    """The forced and factor artifacts ``write_summaries`` needs but cannot
    find. Empty when every doctrine has both."""
    missing = []
    for doctrine in doctrines:
        for method in ("forced", "factor"):
            path = data.predictions_path(doctrine, run_id, method)
            if not path.exists():
                missing.append(path)
    return missing


def write_summaries(doctrines: Sequence[str], run_id: str) -> dict[str, Path]:
    """Write ``test_forced_cases_segment.json`` per doctrine and the pooled
    ``generalization_summary_segment_95.json`` to ``summary_output_dir(run_id)``.
    """
    out_dir = summary_output_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    rows_by_doctrine: dict[str, list[dict]] = {}
    for doctrine in doctrines:
        rows = case_rows(doctrine, run_id)
        attach_factors(doctrine, rows, run_id)
        written[f"forced_cases:{doctrine}"] = write_forced_cases(
            doctrine, rows, run_id
        )
        rows_by_doctrine[doctrine] = rows

    summary = pooled(rows_by_doctrine, np.random.default_rng(SEED + 1))
    leaf = pooled(rows_by_doctrine, np.random.default_rng(SEED + 2), leaf=True)
    payload = {
        "meta": {
            "tree_run_id": run_id,
            "n_boot": N_BOOT,
        },
        "summary": summary,
        "leaf": leaf,
        "leaf_prefix": pooled_prefix(
            rows_by_doctrine,
            np.random.default_rng(SEED + 3),
            ("at", "nn", "circuit", "party"),
        ),
    }
    path = out_dir / data.SUMMARY_FILE
    path.write_text(json.dumps(payload, indent=1))
    written["generalization_summary"] = path
    for m in SUMMARY_METHODS:
        s = summary[m]
        print(
            f"""[summary] {m:8s} {s['accuracy']:.3f} (SE {s['se']:.3f}) n_dec={s['n_decisions']} n_cases={s['n_cases']} | leaf {leaf[m]['accuracy']:.3f} (SE {leaf[m]['se']:.3f})"""
        )
    return written
