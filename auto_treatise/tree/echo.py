"""Detect, with one LLM call, a proposed split that would leave cases an earlier path sibling describes better."""

from __future__ import annotations

import json
from typing import Any

import api_calls

from .prompts import _complement_judge_schema
from .prompts import _COMPLEMENT_JUDGE_SYSTEM


def _path_not_labels(lineage_steps: list[dict[str, str]]) -> list[str]:
    labels = []
    for step in lineage_steps:
        alt = step["alternate_label"].strip()
        if alt and alt not in labels:
            labels.append(alt)
    return labels


def _tree_outline(
    out_nodes: list[dict[str, Any]], current_node_id: str
) -> list[dict[str, Any]]:
    """Every node written so far.
    """
    by_id = {node["id"]: node for node in out_nodes}

    def child(child_id: str) -> dict[str, str]:
        # Children of the level being split are not written yet, so they have
        # no label.
        if child_id in by_id and by_id[child_id]["label"]:
            return {"node": child_id, "label": by_id[child_id]["label"]}
        return {"node": child_id}

    outline: list[dict[str, Any]] = []
    for node in out_nodes:
        entry: dict[str, Any] = {
            "node": node["id"],
            "label": node["label"],
            "n_cases": len(node["cases"]),
        }
        if node["id"] == current_node_id:
            entry["is_current_cluster"] = True
        if not node["is_leaf"] and node["split_criterion_short"]:
            entry["splits_on"] = node["split_criterion_short"]
            entry["decision_rule"] = node["decision_rule"]
            entry["left_child"] = child(node["left_id"])
            entry["right_child"] = child(node["right_id"])
        outline.append(entry)
    return outline


def _step_for_better_fit_node(
    node_id: str, lineage_steps: list[dict[str, str]]
) -> dict[str, str] | None:
    """The path step that diverged from the named node, or None.
    """
    for step in lineage_steps:
        alternate = step["alternate_id"]
        if alternate and (
            node_id == alternate or node_id.startswith(alternate + ".")
        ):
            return step
    return None


def _build_complement_judge_prompt(
    proposal: dict[str, Any],
    node_label: str,
    node_id: str,
    out_nodes: list[dict[str, Any]],
) -> str:
    """The whole tree, with every decision rule, plus the proposed split."""
    payload = {
        "tree_so_far": _tree_outline(out_nodes, node_id),
        "current_cluster": {"node": node_id, "label": node_label},
        # Same key names and nesting as a tree_so_far entry, so the judge
        # compares like with like. The children have no node id or case count
        # yet because this split has not been committed.
        "proposed_split": {
            "splits_on": proposal["split_criterion_short"],
            "decision_rule": proposal["decision_rule"],
            "left_child": {
                "label": proposal["left_label"],
                "description": proposal["left_description"],
            },
            "right_child": {
                "label": proposal["right_label"],
                "description": proposal["right_description"],
            },
        },
    }
    return f"""`tree_so_far` is every cluster written so far, with the rule that splits it. A node id shows its position: n0.L.L sits under n0.L, which sits under n0. The cluster being split now is marked `is_current_cluster`.

Would either proposed child hold cases that a different existing cluster describes better than this one does?

Explain in `misplacement_analysis`. If so, set `better_fit_node` to that cluster's node id. If every case belongs where it is, set it to null.

Naming a cluster moves those cases there and stops this one from splitting, so name one only when you can say what kind of case is in the wrong place. A split that draws a finer line among cases that all belong here is a good split: answer null.

Clusters that discuss the same subject are not the same cluster. A finer cut inside this cluster will often reuse the vocabulary of the cluster it was separated from, because both describe the same area of law. Shared wording alone does not mean any case belongs elsewhere.

{json.dumps(payload, indent=2, ensure_ascii=False)}"""


def _judge_complement_echo(
    proposal: dict[str, Any],
    lineage_steps: list[dict[str, str]],
    model: str,
    node_label: str,
    node_id: str,
    out_nodes: list[dict[str, Any]],
) -> str | None:
    """Node id of a cluster that fits the proposed children's cases better, or
    None when the split keeps every case where it is."""
    if not _path_not_labels(lineage_steps):
        return None
    parsed = api_calls.query_llm(
        _build_complement_judge_prompt(
            proposal, node_label, node_id, out_nodes
        ),
        system_prompt=_COMPLEMENT_JUDGE_SYSTEM,
        model=model,
        response_format=_complement_judge_schema(),
    )
    return parsed["better_fit_node"] or None


def _detect_complement_echo(
    proposal: dict[str, Any],
    lineage_steps: list[dict[str, str]],
    model: str,
    node_label: str,
    node_id: str,
    out_nodes: list[dict[str, Any]],
) -> dict[str, str] | None:
    """The path step this split would leave misrouted cases behind, or None.
    """
    if not lineage_steps:
        return None
    better_fit = _judge_complement_echo(
        proposal, lineage_steps, model, node_label, node_id, out_nodes
    )
    if better_fit is None:
        return None
    step = _step_for_better_fit_node(better_fit, lineage_steps)
    if step is None:
        print(
            f"""[tree_inducer] {node_id} judge named {better_fit!r} as a better fit, but it is not a cluster these cases passed over; keeping the split"""
        )
    return step
