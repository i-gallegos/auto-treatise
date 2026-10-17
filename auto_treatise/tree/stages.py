"""Stage 1 (axis proposal) and stage 2 (case assignment) LLM calls."""

from __future__ import annotations

from typing import Any

import api_calls

from .config import MAX_ASSIGNMENT_ROUNDS
from .config import MAX_AXIS_RETRIES
from .config import MIN_BALANCE_CHECK_SIZE
from .config import MIN_CHILD_FRACTION
from .echo import _detect_complement_echo
from .prompts import _assignment_chunks
from .prompts import _assignment_schema
from .prompts import _ASSIGNMENT_SYSTEM
from .prompts import _axis_proposal_schema
from .prompts import _AXIS_SYSTEM
from .prompts import _AXIS_SYSTEM_RAW
from .prompts import _build_assignment_prompt
from .prompts import _build_assignment_retry_prompt
from .prompts import _build_axis_prompt
from .prompts import _failed_axis_notes
from .prompts import _failed_axis_record
from .prompts import _rows_are_raw


def _remember_proposed_axis(
    created_axes: list[dict[str, str]],
    created_labels: list[str],
    item: dict[str, Any],
    proposal: dict[str, Any],
) -> None:
    created_axes.append(
        {
            "node_id": item["node_id"],
            "split_criterion_short": proposal["split_criterion_short"].strip(),
            "left_label": proposal["left_label"].strip(),
            "right_label": proposal["right_label"].strip(),
        }
    )
    for label in (proposal["left_label"], proposal["right_label"]):
        text = label.strip()
        if text and text not in created_labels:
            created_labels.append(text)


def _is_degenerate_split(n_cases: int, n_left: int, n_right: int) -> bool:
    smaller = min(n_left, n_right)
    return smaller == 0 or (
        n_cases >= MIN_BALANCE_CHECK_SIZE
        and smaller < MIN_CHILD_FRACTION * n_cases
    )


def _retry_fields(failed_axes: list[dict[str, Any]]) -> dict[str, Any]:
    if not failed_axes:
        return {}
    return {
        "failed_axes": failed_axes,
        "split_notes": _failed_axis_notes(failed_axes),
    }


def _propose_axes_for_items(
    items: list[dict[str, Any]],
    constraint: str,
    constraint_id: str,
    created_labels: list[str],
    model: str,
    out_nodes: list[dict[str, Any]],
    created_axes: list[dict[str, str]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Stage 1 for every item."""
    work: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in items:
        while True:
            failed_axes = item["failed_axes"]
            proposal = _stage1_propose_axis(
                item,
                constraint,
                constraint_id,
                created_labels,
                model,
                failed_axes,
                created_axes,
            )
            extra = _retry_fields(failed_axes)
            if proposal["is_leaf"]:
                extra.update(
                    {
                        key: proposal[key]
                        for key in (
                            "split_criterion",
                            "axis_type",
                            "split_criterion_short",
                            "decision_rule",
                        )
                    }
                )
                _emit_leaf(
                    item, out_nodes, leaf_reason="model_declared_leaf", **extra
                )
                break
            step = _detect_complement_echo(
                proposal,
                item["lineage_steps"],
                model,
                item["node_label"],
                item["node_id"],
                out_nodes,
            )
            if step is not None:
                failed_axes = failed_axes + [
                    _failed_axis_record(
                        proposal,
                        0,
                        0,
                        reason="path_complement",
                        echoed_label=step["alternate_label"],
                    )
                ]
                item["failed_axes"] = failed_axes
                if len(failed_axes) <= MAX_AXIS_RETRIES:
                    print(
                        f"""[tree_inducer] {item['node_id']} path-complement echo of {step['alternate_label']!r}; retrying axis ({len(failed_axes)}/{MAX_AXIS_RETRIES})"""
                    )
                    continue
                _emit_leaf(
                    item,
                    out_nodes,
                    leaf_reason="path_complement",
                    **_retry_fields(failed_axes),
                )
                break
            _remember_proposed_axis(
                created_axes, created_labels, item, proposal
            )
            work.append((item, proposal))
            break
    return work


def _normalize_case_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "id": str(row["id"])} for row in rows]


def _ancestor_labels(
    node_label: str, lineage_steps: list[dict[str, str]]
) -> set[str]:
    """Labels already visible to this node via the lineage block."""
    labels = {node_label}
    for step in lineage_steps:
        labels.add(step["selected_label"])
        labels.add(step["alternate_label"])
    return {label for label in labels if label}


def _inbound_confidence_fields(
    inbound_assignments: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Per-case confidence for how cases arrived at this node from its parent
    split. Stored on the child."""
    if not inbound_assignments:
        return {}
    low_ids = sorted(
        case_id
        for case_id, entry in inbound_assignments.items()
        if entry["confidence"] == "low"
    )
    return {
        "inbound_assignments": dict(inbound_assignments),
        "low_confidence_cases": low_ids,
        "low_confidence_fraction": round(
            len(low_ids) / len(inbound_assignments), 3
        ),
    }


def _emit_leaf(
    item: dict[str, Any], out_nodes: list[dict[str, Any]], **extra: Any
) -> None:
    leaf = {
        "id": item["node_id"],
        "parent_id": item["parent_id"],
        "depth": item["depth"],
        "label": item["node_label"],
        "description": item["inherited_description"],
        "split_criterion": extra.pop("split_criterion", ""),
        "cases": [row["id"] for row in item["cases_with_profiles"]],
        "is_leaf": True,
        "left_id": None,
        "right_id": None,
    }
    leaf.update(_inbound_confidence_fields(item["inbound_assignments"]))
    leaf.update(extra)
    out_nodes.append(leaf)


def _stage1_propose_axis(
    item: dict[str, Any],
    constraint: str,
    constraint_id: str,
    created_labels: list[str],
    model: str,
    failed_axes: list[dict[str, Any]],
    created_axes: list[dict[str, str]],
) -> dict[str, Any]:
    """Stage 1 for one node. Sequential so that a node sees what its peers have
    already named."""
    excluded = _ancestor_labels(item["node_label"], item["lineage_steps"])
    prompt = _build_axis_prompt(
        constraint=constraint,
        constraint_id=constraint_id,
        node_id=item["node_id"],
        node_label=item["node_label"],
        inherited_description=item["inherited_description"],
        cases_with_profiles=item["cases_with_profiles"],
        depth=item["depth"],
        lineage_steps=item["lineage_steps"],
        used_labels=[
            label for label in created_labels if label not in excluded
        ],
        created_axes=created_axes,
        failed_axes=failed_axes,
    )
    return api_calls.query_llm(
        prompt,
        system_prompt=(
            _AXIS_SYSTEM_RAW
            if _rows_are_raw(item["cases_with_profiles"])
            else _AXIS_SYSTEM
        ),
        model=model,
        response_format=_axis_proposal_schema(),
    )


def _rows_for_ids(
    chunk: list[dict[str, Any]], ids: list[str]
) -> list[dict[str, Any]]:
    """Rows in *chunk* whose ids are in *ids*, in *ids* order."""
    by_id = {row["id"]: row for row in chunk}
    return [by_id[i] for i in ids if i in by_id]


def _stage2_assign_level(
    work: list[tuple[dict[str, Any], dict[str, Any]]],
    constraint: str,
    model: str,
) -> dict[str, dict[str, Any]]:
    """Assign every case at this depth to LEFT or RIGHT.
    """
    results: dict[str, dict[str, Any]] = {
        item["node_id"]: {
            "left_ids": [],
            "right_ids": [],
            "assignments": {},
            "dropped_ids": [],
        }
        for item, _ in work
    }

    # (node_id, prompt, ids, chunk rows)
    pending: list[tuple] = []
    for item, proposal in work:
        for chunk in _assignment_chunks(item["cases_with_profiles"]):
            prompt = _build_assignment_prompt(
                constraint=constraint,
                node_label=item["node_label"],
                decision_rule=proposal["decision_rule"],
                split_criterion_short=proposal["split_criterion_short"],
                left_label=proposal["left_label"],
                left_description=proposal["left_description"],
                right_label=proposal["right_label"],
                right_description=proposal["right_description"],
                chunk=chunk,
            )
            pending.append(
                (
                    item["node_id"],
                    prompt,
                    [row["id"] for row in chunk],
                    chunk,
                )
            )

    round_idx = 0
    while pending:
        responses = [
            response
            for _, response in api_calls.batch_query_llm(
                [
                    {"content": prompt, "system_prompt": _ASSIGNMENT_SYSTEM}
                    for _, prompt, *_rest in pending
                ],
                model=model,
                response_format=_assignment_schema(),
            )
        ]
        retry: list[tuple] = []
        for (node_id, prompt, expected_ids, chunk), response in zip(
            pending, responses
        ):
            store = results[node_id]["assignments"]
            for entry in response["assignments"]:
                # Ignore ids this chunk was not asked about, and never let
                # a later chunk overwrite an assignment already made.
                if entry["id"] in expected_ids and entry["id"] not in store:
                    store[entry["id"]] = {
                        "side": entry["side"],
                        "rationale": entry["rationale"],
                        "confidence": entry["confidence"],
                    }
            missing = [id for id in expected_ids if id not in store]
            if not missing:
                continue
            if round_idx < MAX_ASSIGNMENT_ROUNDS - 1:
                retry.append(
                    (
                        node_id,
                        _build_assignment_retry_prompt(prompt, missing),
                        missing,
                        _rows_for_ids(chunk, missing),
                    )
                )
            else:
                results[node_id]["dropped_ids"].extend(missing)
        pending = retry
        if retry:
            round_idx += 1

    for item, _ in work:
        outcome = results[item["node_id"]]
        for row in item["cases_with_profiles"]:
            if row["id"] in outcome["assignments"]:
                side = outcome["assignments"][row["id"]]["side"]
                outcome[f"{side}_ids"].append(row["id"])
    return results
