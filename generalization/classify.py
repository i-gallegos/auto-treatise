"""Classify held-out cases by descending the tree on x^[u] (§5).

* ``run_classification``: the tree-inducing model walks the tree on ``x^[u]``
  and the authority context, one greedy descent per case.
* ``run_forced``: the teacher-forced pass, one decision at every node on the
  annotated path, so an early error does not cascade.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from courtlistener import courts  # noqa: E402
from courtlistener.citation_strings import choose_primary_citation  # noqa: E402
from courtlistener.citation_strings import parse_citations  # noqa: E402

from . import data  # noqa: E402
from . import prompts  # noqa: E402
from . import schemas  # noqa: E402
from .artifacts import _meta  # noqa: E402
from .artifacts import _pass_status  # noqa: E402
from .artifacts import _prepare  # noqa: E402
from .artifacts import _route_round  # noqa: E402
from .artifacts import _sort_case_rows  # noqa: E402
from .artifacts import _write  # noqa: E402
from .metrics import leaf_distribution  # noqa: E402

# --------------------------------------------------------------------------- #
# Authority context
# --------------------------------------------------------------------------- #

MAX_PER_GROUP = 25
SUMMARY_CHARS = 240
LEAF_DESCRIPTION_CHARS = 260
RELATIONS: tuple[str, ...] = ("binding", "same_court", "same_jurisdiction")

_HEADINGS = {
    "binding": "BINDING ON THIS COURT",
    "same_court": "THIS COURT'S OWN PRIOR DECISIONS",
    "same_jurisdiction": "WITHIN THE SAME JURISDICTION (persuasive)",
}

_TRUTH_GLOSS = {
    "true": "the constraint was satisfied on these facts",
    "false": "the constraint was not satisfied on these facts",
    "unaddressed": "the court did not resolve the constraint",
}

_LEGEND = """Each entry gives the case, then the question it faced, what it did with the constraint, and the reason the tree placed it at that position. Read the placement reason as the tree's own account of itself, not as an independent authority. A case marked UNPUBLISHED is non-precedential in most circuits: it appears under its court's heading because of who decided it, but it does not bind. Cases shown without a description are ones the corpus has no summary for."""


def relation(candidate: dict[str, Any], target: dict[str, Any]) -> str | None:
    """How ``candidate`` bears on a case decided by ``target``'s court.
    """
    if not target["court_level"]:
        return None
    if courts.binds(candidate, target):
        return "binding"
    if (
        candidate["court_id"]
        and candidate["court_id"] == target["court_id"]
    ):
        return "same_court"

    system = courts.court_system(candidate["court_level"])
    if system is None or system != courts.court_system(target["court_level"]):
        return None
    if system == "federal":
        # Same circuit, but not the circuit court itself (that is ``binding``)
        # and not this court (``same_court``). Sister districts and the
        # bankruptcy courts of the circuit.
        if candidate["circuit"] and candidate["circuit"] == target["circuit"]:
            return "same_jurisdiction"
        return None
    if (
        candidate["jurisdiction"]
        and candidate["jurisdiction"] == target["jurisdiction"]
    ):
        return "same_jurisdiction"
    return None


def _decided_before(candidate: dict[str, Any], target: dict[str, Any]) -> bool:
    """Whether ``candidate`` was available to the court deciding ``target``.
    """
    target_date = str(target["decision_date"] or "")
    if not target_date:
        return True
    candidate_date = str(candidate["decision_date"] or "")
    if not candidate_date:
        return False
    return candidate_date <= target_date


# --------------------------------------------------------------------------- #
# Building the context
# --------------------------------------------------------------------------- #


def _authority_rank(meta: dict[str, Any]) -> int:
    level = meta["court_level"]
    return courts.AUTHORITY_RANK[level] if level in courts.AUTHORITY_RANK else 0


def _cite(meta: dict[str, Any]) -> str:
    """A short parenthetical: court abbreviation and year."""
    bits = [
        meta["court_abbreviation"] or meta["court_id"],
        str(meta["decision_date"] or "")[:4],
    ]
    return " ".join(str(b) for b in bits if b)


def _primary_citation(raw: Any) -> str:
    """The one cite a reader would look the case up by.
    """
    return choose_primary_citation(parse_citations(raw)) or ""


def _dedup_key(case_id: str, meta: dict[str, Any]) -> tuple:
    """Identity of the *opinion*, not of the row.
    """
    cluster = str(meta["courtlistener_cluster_id"] or "").strip()
    if cluster:
        return ("cluster", cluster)
    cite = _primary_citation(meta["citation"])
    if cite:
        return ("cite", cite, meta["court_id"])
    name = " ".join(str(meta["case_name"] or "").lower().split())
    if name and meta["decision_date"]:
        return ("namedate", name, meta["decision_date"], meta["court_id"])
    return ("row", str(case_id))


def _trim(text: Any, limit: int) -> str:
    """``text`` to at most ``limit`` characters, cut at a sentence boundary.
    """
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    window = clean[: limit + 1]
    cut = max(window.rfind(". "), window.rfind("; "), window.rfind(".” "))
    if cut > limit // 2:
        return window[: cut + 1]
    return clean[:limit].rstrip() + "…"


def _status(meta: dict[str, Any]) -> str:
    """Precedential status.
    """
    value = str(meta["precedential_status"] or "").strip().lower()
    if value == "unpublished":
        return "UNPUBLISHED — non-precedential in most circuits"
    if value in ("", "unknown"):
        return "publication status unknown"
    return ""


def _placement_rationale(tree: data.Tree, case_id: str, leaf_id: str) -> Any:
    """Why induction put this case in this leaf, from the run's own record.
    """
    assignments = tree.nodes[tree.nodes[leaf_id]["parent_id"]]["assignments"]
    return assignments[case_id]["rationale"] if case_id in assignments else None


def _case_line(case_id: str, meta: dict[str, Any]) -> str:
    """The one-line form: name, primary cite, court and year."""
    name = str(meta["case_name"] or f"case {case_id}").strip()
    parts = [name]
    citation = _primary_citation(meta["citation"])
    if citation:
        parts.append(citation)
    inner = _cite(meta)
    return f"- {', '.join(parts)}" + (f" ({inner})" if inner else "")


def _has_detail(case: dict[str, Any]) -> bool:
    return bool(
        case["issue_summary"]
        or case["application_summary"]
        or case["placement_rationale"]
    )


def _case_block(entry: dict[str, Any]) -> list[str]:
    """The detailed form: the one-line header, then what the case did and why.
    """
    head = _case_line(entry["case_id"], entry)
    tags = [t for t in (_status(entry),) if t]
    if entry["citation_count"]:
        tags.append(f"cited {entry['citation_count']}×")
    lines = [head + (f"  [{'; '.join(tags)}]" if tags else "")]

    if entry["issue_summary"]:
        lines.append(f"    Issue: {entry['issue_summary']}")
    if entry["application_summary"]:
        truth = str(entry["truth_value"] or "")
        prefix = (
            f"Analysis ({_TRUTH_GLOSS[truth]}): "
            if truth in _TRUTH_GLOSS
            else "Analysis: "
        )
        lines.append(f"    {prefix}{entry['application_summary']}")
    if entry["placement_rationale"]:
        lines.append(f"    {entry['placement_rationale']}")
    return lines


def context(
    tree: data.Tree,
    target_id: str,
    meta: dict[str, dict],
    case_to_leaf: dict[str, str],
    profiles: dict[str, dict],
) -> dict[str, Any]:
    """The structured authority context for one held-out case.
    """
    target = meta[target_id]

    grouped: dict[str, list[tuple[str, dict]]] = {r: [] for r in RELATIONS}
    for train_id in case_to_leaf:
        candidate = meta[train_id]
        if train_id == target_id:
            continue
        rel = relation(candidate, target)
        if rel is None or not _decided_before(candidate, target):
            continue
        grouped[rel].append((train_id, candidate))

    out: dict[str, Any] = {"target": target, "groups": {}, "omitted": {}}
    for rel in RELATIONS:
        rows = grouped[rel]
        rows.sort(
            key=lambda kv: str(kv[1]["decision_date"] or ""), reverse=True
        )
        rows.sort(key=lambda kv: -_authority_rank(kv[1]))

        seen: set[tuple] = set()
        deduped: list[tuple[str, dict]] = []
        for train_id, candidate in rows:
            key = _dedup_key(train_id, candidate)
            if key not in seen:
                seen.add(key)
                deduped.append((train_id, candidate))

        by_leaf: dict[str, list[dict]] = {}
        for train_id, candidate in deduped[:MAX_PER_GROUP]:
            leaf = case_to_leaf[train_id]
            profile = profiles[train_id]
            by_leaf.setdefault(leaf, []).append(
                {
                    **candidate,
                    "case_id": train_id,
                    "issue_summary": _trim(
                        profile["issue_summary"], SUMMARY_CHARS
                    ),
                    "application_summary": _trim(
                        profile["application_summary"], SUMMARY_CHARS
                    ),
                    "truth_value": profile["truth_value"],
                    "placement_rationale": _placement_rationale(
                        tree, train_id, leaf
                    ),
                }
            )
        out["groups"][rel] = [
            {
                "leaf": leaf,
                "leaf_label": tree.nodes[leaf]["label"] or "unlabeled",
                "leaf_description": _trim(
                    tree.nodes[leaf]["description"], LEAF_DESCRIPTION_CHARS
                ),
                "cases": cases,
            }

            for leaf, cases in sorted(
                by_leaf.items(), key=lambda kv: (-len(kv[1]), kv[0])
            )
        ]
        out["omitted"][rel] = max(0, len(deduped) - MAX_PER_GROUP)
    return out


def render_context(payload: dict[str, Any]) -> str:
    """The context block a routing prompt shows, or ``""`` when nothing applies.
    """
    groups = payload["groups"]
    if not any(groups[rel] for rel in RELATIONS):
        return ""

    court = payload["target"]
    name = (
        court["court_name"]
        or court["court_abbreviation"]
        or court["court_id"]
        or "an unidentified court"
    )
    where = []
    if court["circuit"]:
        where.append(f"circuit {court['circuit']}")
    if court["jurisdiction"]:
        where.append(str(court["jurisdiction"]))
    heading = f"The new case was decided by: {name}" + (
        f" ({', '.join(where)})" if where else ""
    )
    if court["decision_date"]:
        heading += f""", on {court['decision_date']}. Only cases decided on or before that date are listed."""

    parts = [heading]
    if any(
        _has_detail(case)
        for rel in RELATIONS
        for entry in groups[rel]
        for case in entry["cases"]
    ):
        parts.append(_LEGEND)

    for rel in RELATIONS:
        lines = [f"## {_HEADINGS[rel]}"]
        if not groups[rel]:
            lines.append("(no case on the tree stands in this relation)")
        else:
            for entry in groups[rel]:
                lines.append(f"[{entry['leaf']}] {entry['leaf_label']}")
                if entry["leaf_description"]:
                    lines.append(f"  {entry['leaf_description']}")
                for case in entry["cases"]:
                    if _has_detail(case):
                        lines.extend("  " + line for line in _case_block(case))
                    else:
                        lines.append("  " + _case_line(case["case_id"], case))
            if payload["omitted"][rel]:
                # Unnumbered for the same reason as the header above: the count
                # of what was withheld is itself a mass signal.
                lines.append(
                    """(Further cases in this group are not listed; the oldest were dropped first.)"""
                )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def authority_contexts(
    doctrine: str, tree: data.Tree, ids: list[str]
) -> dict[str, str]:
    """``{case_id: rendered context}``, one read of each source for the split.

    Cases with no in-jurisdiction authority on the tree map to ``""``, which
    ``prompts.build_routing_prompt`` treats as "omit the section".
    """
    case_to_leaf = tree.case_to_leaf()
    meta = data.case_metadata(doctrine, ids + list(case_to_leaf))
    profiles = data.load_case_profiles(doctrine)
    contexts = {
        c: render_context(context(tree, c, meta, case_to_leaf, profiles))
        for c in ids
    }
    n = sum(1 for v in contexts.values() if v.strip())
    print(f"[authority] {n} of {len(ids)} case(s) have authority context")
    return contexts


# --------------------------------------------------------------------------- #
# Descent
# --------------------------------------------------------------------------- #


def _routing_items(
    tree: data.Tree,
    constraint: str,
    steps: list[tuple[str, str]],
    headers: dict[str, str],
    views: dict[str, str],
    contexts: dict[str, str],
) -> list[dict[str, Any]]:
    """One routing request per ``(case_id, node_id)``."""
    items = []
    for case_id, node_id in steps:
        parts = prompts.build_routing_prompt(
            tree,
            node_id,
            constraint,
            headers[case_id],
            views[case_id],
            authority_context=contexts[case_id],
        )
        items.append(
            {
                "content": parts.suffix,
                "system_prompt": prompts.CLASSIFIER_SYSTEM,
                "cache_prefix": parts.prefix,
            }
        )
    return items


def _step_walker(
    tree: data.Tree, walker: dict[str, Any], decision: dict
) -> dict[str, Any]:
    """Advance one walker to the child on the side the model named."""
    node_id = walker["node_id"]
    side = decision["side"]
    if side not in ("left", "right"):
        raise ValueError(f"invalid side {side!r} at {node_id}")
    left, right = tree.children(node_id)
    child = left if side == "left" else right
    path = walker["path"] + [
        {
            "node_id": node_id,
            "side": side,
            "child_id": child["id"],
            "rationale": decision["rationale"],
            "confidence": decision["confidence"],
        }
    ]
    if child["is_leaf"]:
        return {
            **walker,
            "node_id": child["id"],
            "path": path,
            "leaf": child["id"],
            "done": True,
        }
    return {**walker, "node_id": child["id"], "path": path, "done": False}


def descend_many(
    tree: data.Tree,
    constraint: str,
    case_ids: list[str],
    model: str,
    headers: dict[str, str],
    views: dict[str, str],
    contexts: dict[str, str],
) -> list[dict[str, Any]]:
    """BFS-route many walkers, one batch round per depth.
    """
    schema = schemas.routing_schema()
    pending = [
        {"case_id": c, "node_id": tree.root["id"], "path": []}
        for c in case_ids
    ]
    finished: list[dict[str, Any]] = []
    round_i = 0
    while pending:
        round_i += 1
        items = _routing_items(
            tree,
            constraint,
            [(w["case_id"], w["node_id"]) for w in pending],
            headers,
            views,
            contexts,
        )
        decisions = _route_round(items, model, schema)
        stepped = [_step_walker(tree, w, d) for w, d in zip(pending, decisions)]
        newly_finished = [w for w in stepped if w["done"]]
        finished.extend(
            {"case_id": w["case_id"], "leaf": w["leaf"], "path": w["path"]}
            for w in newly_finished
        )
        pending = [w for w in stepped if not w["done"]]
        print(
            f"""[descend] round {round_i}: {len(newly_finished)} finished, {len(pending)} pending"""
        )
    return finished


# --------------------------------------------------------------------------- #
# Pass 2: Auto-Treatise classifier
# --------------------------------------------------------------------------- #


def run_classification(
    doctrine: str,
    run_id: str,
    case_ids: list[str],
    model: str,
) -> dict[str, Any]:
    """Classify each held-out case from ``x^[u]`` and the authority context.
    """
    tree, constraint, accepted, meta = _prepare(doctrine, run_id, case_ids)
    views = data.fact_views(doctrine, accepted)
    ids = [c for c in accepted if c in views]
    headers = {c: data.case_header(meta[c]) for c in ids}
    contexts = authority_contexts(doctrine, tree, ids)
    path = data.predictions_path(doctrine, run_id, "tree")
    _pass_status("classify", len(accepted), len(ids))

    walkers = descend_many(
        tree, constraint, ids, model, headers, views, contexts
    )
    predictions = [
        {
            "case_id": row["case_id"],
            "distribution": leaf_distribution([row["leaf"]]),
            "samples": [row],
        }
        for row in walkers
    ]
    payload = {
        "meta": _meta(doctrine, run_id, tree, model=model),
        "predictions": _sort_case_rows(predictions),
    }
    _write(path, payload)
    return payload


# --------------------------------------------------------------------------- #
# Teacher-forced pass
# --------------------------------------------------------------------------- #


def _forced_targets(
    tree: data.Tree, labels: dict[str, dict], ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """``{case_id: [{node_id, depth, gold_side}]}`` for every node
    on the gold path.

    The gold path is read off the annotator's label, so unlike a descent it is
    known before a single call is made.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for case_id in ids:
        steps = [
            {
                "node_id": step["node_id"],
                "depth": tree.nodes[step["node_id"]]["depth"],
                "gold_side": step["side"],
                "gold_confidence": step["confidence"],
            }
            for step in labels[case_id]["path"]
        ]
        if steps:
            out[case_id] = steps
    return out


def _forced_rounds(
    targets: dict[str, list[dict[str, Any]]],
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Split the work into a cache-warming round and a bulk round.
    """
    warm: list[tuple[str, dict[str, Any]]] = []
    bulk: list[tuple[str, dict[str, Any]]] = []
    for case_id, steps in targets.items():
        ordered = sorted(steps, key=lambda s: s["depth"])
        if ordered:
            warm.append((case_id, ordered[0]))
        bulk.extend((case_id, s) for s in ordered[1:])
    bulk.sort(key=lambda item: (item[0], item[1]["depth"]))
    return [r for r in (warm, bulk) if r]


def run_forced(
    doctrine: str, run_id: str, case_ids: list[str], model: str
) -> dict[str, Any]:
    """Ask the classifier to decide at **every node on the gold path**.
    """
    tree, constraint, accepted, meta = _prepare(doctrine, run_id, case_ids)
    labels = {
        row["case_id"]: row
        for row in json.loads(data.labels_path(doctrine, run_id).read_text())[
            "labels"
        ]
    }
    views = data.fact_views(doctrine, accepted)
    ids = [c for c in accepted if c in views and c in labels]
    headers = {c: data.case_header(meta[c]) for c in ids}
    contexts = authority_contexts(doctrine, tree, ids)
    targets = _forced_targets(tree, labels, ids)

    path = data.predictions_path(doctrine, run_id, "forced")
    results: dict[str, list[dict[str, Any]]] = {c: [] for c in targets}
    n_calls = sum(len(s) for s in targets.values())
    _pass_status(
        "forced", len(accepted), len(targets), extra=f"  ({n_calls} decisions)"
    )

    schema = schemas.routing_schema()
    for round_i, batch in enumerate(_forced_rounds(targets), start=1):
        items = _routing_items(
            tree,
            constraint,
            [(case_id, step["node_id"]) for case_id, step in batch],
            headers,
            views,
            contexts,
        )
        kind = "cache-warming" if round_i == 1 else "bulk (prefix cached)"
        print(f"[forced] round {round_i} ({kind}): {len(items)} call(s)")
        decisions = _route_round(items, model, schema)
        for (case_id, step), decision in zip(batch, decisions):
            if decision["side"] not in ("left", "right"):
                raise ValueError(
                    f"invalid side {decision['side']!r} at {step['node_id']}"
                )
            results[case_id].append(
                {
                    "node_id": step["node_id"],
                    "depth": step["depth"],
                    "gold_side": step["gold_side"],
                    "gold_confidence": step["gold_confidence"],
                    "side": decision["side"],
                    "correct": decision["side"] == step["gold_side"],
                    "rationale": decision["rationale"],
                    "confidence": decision["confidence"],
                    "axis_type": tree.nodes[step["node_id"]]["axis_type"],
                }
            )

    rows = [
        {"case_id": c, "decisions": sorted(d, key=lambda x: x["depth"])}
        for c, d in results.items()
        if d
    ]
    scored = [d for row in rows for d in row["decisions"]]
    payload = {
        "meta": _meta(
            doctrine,
            run_id,
            tree,
            model=model,
            n_cases=len(rows),
            n_decisions=len(scored),
            accuracy=sum(d["correct"] for d in scored) / len(scored),
        ),
        "predictions": _sort_case_rows(rows),
    }
    _write(path, payload)
    print(
        f"""[forced] {len(scored)} decision(s) over {len(rows)} case(s): accuracy {payload['meta']['accuracy']:.3f}"""
    )
    return payload
