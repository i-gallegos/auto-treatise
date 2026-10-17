"""Recursive splitting: finalize a split and expand the tree."""

from __future__ import annotations

from typing import Any

from .config import MAX_AXIS_RETRIES
from .config import MAX_RECURSION_DEPTH
from .config import MIN_SPLIT_SIZE
from .prompts import _failed_axis_notes
from .prompts import _failed_axis_record
from .stages import _emit_leaf
from .stages import _inbound_confidence_fields
from .stages import _is_degenerate_split
from .stages import _normalize_case_keys
from .stages import _propose_axes_for_items
from .stages import _stage2_assign_level


def _finalize_split(
    item: dict[str, Any],
    proposal: dict[str, Any],
    left_ids: list[str],
    right_ids: list[str],
    assignments: dict[str, dict[str, str]],
    dropped_ids: list[str],
    notes: list[str],
    out_nodes: list[dict[str, Any]],
    created_labels: list[str],
    failed_axes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write one internal node and return its two child work items."""
    node_id = item["node_id"]
    node_label = item["node_label"]
    ids = [row["id"] for row in item["cases_with_profiles"]]
    notes = list(notes)
    if failed_axes:
        notes.extend(_failed_axis_notes(failed_axes))

    # Cases the model never assigned after retries are excluded from the split.
    if dropped_ids:
        notes.append(
            f"""{len(dropped_ids)} case(s) never assigned by the model and excluded from the split: {', '.join(str(i) for i in dropped_ids)}."""
        )

    if _is_degenerate_split(len(ids), len(left_ids), len(right_ids)):
        _emit_leaf(
            item,
            out_nodes,
            split_criterion=proposal["split_criterion"],
            axis_type=proposal["axis_type"],
            split_criterion_short=proposal["split_criterion_short"],
            decision_rule=proposal["decision_rule"],
            leaf_reason="degenerate_split",
            unassigned_cases=list(dropped_ids),
            failed_axes=failed_axes,
            split_notes=notes
            + [
                f"""Proposed split was {len(left_ids)}/{len(right_ids)}; kept as a leaf after {len(failed_axes)} degenerate partition(s)."""
            ],
        )
        return []

    left_label = proposal["left_label"].strip() or f"{node_label} — cluster A"
    right_label = proposal["right_label"].strip() or f"{node_label} — cluster B"
    left_description = proposal["left_description"].strip()
    right_description = proposal["right_description"].strip()

    split_description = item["inherited_description"]
    if notes:
        split_description = f"{split_description} {' '.join(notes)}"

    left_id = f"{node_id}.L"
    right_id = f"{node_id}.R"
    # Outgoing split audit stays on this node; arrival confidence lives on
    # children.
    split_node: dict[str, Any] = {
        "id": node_id,
        "parent_id": item["parent_id"],
        "depth": item["depth"],
        "label": node_label,
        "description": split_description,
        "split_criterion": proposal["split_criterion"],
        "split_criterion_short": proposal["split_criterion_short"],
        "axis_type": proposal["axis_type"],
        "decision_rule": proposal["decision_rule"],
        "left_description": left_description,
        "right_description": right_description,
        "cases": ids,
        "is_leaf": False,
        "left_id": left_id,
        "right_id": right_id,
        "assignments": assignments,
        "unassigned_cases": dropped_ids,
        "split_notes": notes,
    }
    if failed_axes:
        split_node["failed_axes"] = failed_axes
    split_node.update(_inbound_confidence_fields(item["inbound_assignments"]))
    out_nodes.append(split_node)

    if left_label not in created_labels:
        created_labels.append(left_label)
    if right_label not in created_labels:
        created_labels.append(right_label)

    left_set = set(left_ids)
    right_set = set(right_ids)
    left_rows = [
        row for row in item["cases_with_profiles"] if row["id"] in left_set
    ]
    right_rows = [
        row for row in item["cases_with_profiles"] if row["id"] in right_set
    ]
    left_inbound = {
        case_id: assignments[case_id]
        for case_id in left_ids
        if case_id in assignments
    }
    right_inbound = {
        case_id: assignments[case_id]
        for case_id in right_ids
        if case_id in assignments
    }

    base_step = {
        "split_criterion": proposal["split_criterion"],
        "split_criterion_short": proposal["split_criterion_short"],
        "axis_type": proposal["axis_type"],
    }
    return [
        {
            "node_id": left_id,
            "parent_id": node_id,
            "depth": item["depth"] + 1,
            "node_label": left_label,
            "inherited_description": left_description,
            "cases_with_profiles": left_rows,
            "inbound_assignments": left_inbound,
            "lineage_steps": item["lineage_steps"]
            + [
                {
                    **base_step,
                    "selected_label": left_label,
                    "selected_description": left_description,
                    "alternate_label": right_label,
                    "alternate_description": right_description,
                    "alternate_id": right_id,
                    "ancestor_id": node_id,
                    "selected_side": "left",
                }
            ],
        },
        {
            "node_id": right_id,
            "parent_id": node_id,
            "depth": item["depth"] + 1,
            "node_label": right_label,
            "inherited_description": right_description,
            "cases_with_profiles": right_rows,
            "inbound_assignments": right_inbound,
            "lineage_steps": item["lineage_steps"]
            + [
                {
                    **base_step,
                    "selected_label": right_label,
                    "selected_description": right_description,
                    "alternate_label": left_label,
                    "alternate_description": left_description,
                    "alternate_id": left_id,
                    "ancestor_id": node_id,
                    "selected_side": "right",
                }
            ],
        },
    ]


def _expand_tree(
    constraint: str,
    constraint_id: str,
    root_cases: list[dict[str, Any]],
    out_nodes: list[dict[str, Any]],
    model: str,
    max_recursion_depth: int,
) -> None:
    """Level-synchronous tree induction via BFS.

    The frontier is processed one whole level at a time, and stage 2 for the
    entire level goes out as a single batch. Degenerate partitions are
    re-proposed up to MAX_AXIS_RETRIES more times before the node is kept as a
    leaf. Retry path-complement echoes the same way.
    """
    frontier: list[dict[str, Any]] = [
        {
            "node_id": "n0",
            "parent_id": None,
            "depth": 0,
            "node_label": "Corpus root",
            "inherited_description": "Root node.",
            "cases_with_profiles": _normalize_case_keys(root_cases),
            "inbound_assignments": {},
            "lineage_steps": [],
        }
    ]
    created_labels: list[str] = []
    created_axes: list[dict[str, str]] = []

    while frontier:
        for item in frontier:
            item.setdefault("failed_axes", [])

        # 1. Nodes that terminate by rule never reach the model.
        splittable: list[dict[str, Any]] = []
        for item in frontier:
            n_cases = len(item["cases_with_profiles"])
            if n_cases <= 1:
                _emit_leaf(item, out_nodes, leaf_reason="singleton")
            elif n_cases < MIN_SPLIT_SIZE:
                _emit_leaf(item, out_nodes, leaf_reason="too_small")
            elif item["depth"] >= max_recursion_depth:
                _emit_leaf(item, out_nodes, leaf_reason="max_depth")
            else:
                splittable.append(item)

        if not splittable:
            break

        depth = min(item["depth"] for item in splittable)
        print(
            f"""[tree_inducer] depth {depth}: {len(splittable)} node(s) to split, {sum(len(i['cases_with_profiles']) for i in splittable)} case(s)"""
        )

        next_frontier: list[dict[str, Any]] = []
        # 2. Stage 1, sequential within the level so peers see each other's
        # labels.
        work = _propose_axes_for_items(
            splittable,
            constraint,
            constraint_id,
            created_labels,
            model,
            out_nodes,
            created_axes,
        )

        # 3. Stage 2 for the whole level; degenerate nodes retry stage 1+2 here.
        while work:
            assigned = _stage2_assign_level(work, constraint, model)
            retry_items: list[dict[str, Any]] = []
            for item, proposal in work:
                outcome = assigned[item["node_id"]]
                n_left = len(outcome["left_ids"])
                n_right = len(outcome["right_ids"])
                if _is_degenerate_split(
                    len(item["cases_with_profiles"]), n_left, n_right
                ):
                    failed_axes = item["failed_axes"] + [
                        _failed_axis_record(proposal, n_left, n_right)
                    ]
                    item["failed_axes"] = failed_axes
                    if len(failed_axes) <= MAX_AXIS_RETRIES:
                        print(
                            f"""[tree_inducer] {item['node_id']} degenerate {n_left}/{n_right}; retrying axis ({len(failed_axes)}/{MAX_AXIS_RETRIES})"""
                        )
                        retry_items.append(item)
                        continue
                    print(
                        f"""[tree_inducer] {item['node_id']} still degenerate {n_left}/{n_right} after {len(failed_axes)} attempt(s); keeping as a leaf."""
                    )
                next_frontier.extend(
                    _finalize_split(
                        item=item,
                        proposal=proposal,
                        left_ids=outcome["left_ids"],
                        right_ids=outcome["right_ids"],
                        assignments=outcome["assignments"],
                        dropped_ids=outcome["dropped_ids"],
                        notes=[],
                        out_nodes=out_nodes,
                        created_labels=created_labels,
                        failed_axes=item["failed_axes"],
                    )
                )
            work = _propose_axes_for_items(
                retry_items,
                constraint,
                constraint_id,
                created_labels,
                model,
                out_nodes,
                created_axes,
            )

        frontier = next_frontier


def _lineage_steps_for(
    node_id: str, by_id: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current = by_id[node_id]
    while current["parent_id"]:
        parent = by_id[current["parent_id"]]
        if parent["left_id"] == current["id"]:
            selected_side = "left"
            alternate_id = parent["right_id"]
        else:
            selected_side = "right"
            alternate_id = parent["left_id"]
        sibling = by_id[alternate_id]
        chain.append(
            {
                "split_criterion": parent["split_criterion"],
                "split_criterion_short": parent["split_criterion_short"],
                "axis_type": parent["axis_type"],
                "selected_label": current["label"],
                "selected_description": current["description"],
                "alternate_label": sibling["label"],
                "alternate_description": sibling["description"],
                "alternate_id": alternate_id,
                "ancestor_id": parent["id"],
                "selected_side": selected_side,
            }
        )
        current = parent
    chain.reverse()
    return chain


def induce_grammar(
    constraint: str,
    case_profiles: list[dict[str, Any]],
    model: str,
    max_recursion_depth: int = MAX_RECURSION_DEPTH,
) -> dict[str, Any]:
    """Recursively induce a binary case-split dendrogram.

    Each node is split in two stages. First, choose the doctrinal axis from
    issue, facts, and reasoning summaries and state a decision rule. Second,
    use batched calls to sort the full case profiles against that rule.

    Returns:
      {
        "schema": "binary_dendrogram_v1",
        "constraint_id": "...",
        "root_id": "n0",
        "nodes": [...]
      }
    """
    if not case_profiles:
        return {
            "schema": "binary_dendrogram_v1",
            "constraint_id": "r",
            "root_id": None,
            "nodes": [],
        }

    nodes: list[dict[str, Any]] = []
    _expand_tree(
        constraint=constraint,
        constraint_id="r",
        root_cases=case_profiles,
        out_nodes=nodes,
        model=model,
        max_recursion_depth=max_recursion_depth,
    )
    return {
        "schema": "binary_dendrogram_v1",
        "constraint_id": "r",
        "root_id": "n0",
        "nodes": nodes,
    }
