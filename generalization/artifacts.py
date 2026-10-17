"""Setup and output helpers shared by every pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import api_calls

from . import data  # noqa: E402


MAX_COMPLETION_TOKENS = 16384

def _sort_case_rows(rows: list[dict], key: str = "case_id") -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            (0, int(r[key])) if str(r[key]).isdigit() else (1, str(r[key]))
        ),
    )


def _prepare(
    doctrine: str, run_id: str, case_ids: list[str]
) -> tuple[data.Tree, str, list[str], dict[str, dict]]:
    """The tree, constraint, accepted held-out ids and case metadata."""
    tree = data.load_tree(doctrine, run_id)
    constraint = tree.metadata["constraint"]
    ids = data.accepted_ids(doctrine, tree, case_ids)
    meta = data.case_metadata(doctrine, ids)
    return tree, constraint, ids, meta


def _route_round(
    items: list[dict[str, Any]], model: str, schema: dict
) -> list[dict[str, Any]]:
    """One batched call; the parsed decision for each item."""
    responses = api_calls.batch_query_llm(
        items,
        model=model,
        response_format=schema,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    return [decision for _, decision in responses]


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _pass_status(
    name: str, n_accepted: int, n_usable: int, extra: str = ""
) -> None:
    n_empty = n_accepted - n_usable
    print(
        f"""[{name}] {n_accepted} accepted, {n_empty} without an extracted x^[u], {n_usable} to run{extra}"""
    )


def _meta(
    doctrine: str, run_id: str, tree: data.Tree, **extra
) -> dict[str, Any]:
    return {
        "doctrine": doctrine,
        "tree_run_id": run_id,
        "tree_model": tree.metadata["model_name"],
        "constraint": tree.metadata["constraint"],
        **extra,
    }
