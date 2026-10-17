"""Gold leaf labels for held-out cases.

``run_segmentation`` has the annotator model run the same preprocessing that
Auto-Treatise applies to training cases: majority isolation, the relevance
filter, sentence attribution and the case profile.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from auto_treatise.doctrine import load_config  # noqa: E402
from . import data  # noqa: E402
from . import prompts  # noqa: E402
from . import schemas  # noqa: E402
from .artifacts import _meta  # noqa: E402
from .artifacts import _pass_status  # noqa: E402
from .artifacts import _prepare  # noqa: E402
from .artifacts import _route_round  # noqa: E402
from .artifacts import _sort_case_rows  # noqa: E402
from .artifacts import _write  # noqa: E402

# --------------------------------------------------------------------------- #
# The segmented_profile view
# --------------------------------------------------------------------------- #


def _safe(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def cache_columns(model: str) -> dict[str, str]:
    """The three annotator-owned cache columns on ``cases`` for ``model``."""
    safe = _safe(model)
    return {
        "relevance": f"paragraph_relevance_cache_annotator_{safe}_json",
        "segmentation": f"paragraph_sentence_segmentation_cache_annotator_{safe}_json",
        "profile": f"annotator_constraint_profile_{safe}_json",
    }


def run_segmentation(
    doctrine: str,
    case_ids: list[str],
    constraint: str,
    model: str,
) -> None:
    """Segment and profile ``case_ids`` with ``model``.
    Cached per column; re-runs are free.

    ``constraint`` should be the tree's (``tree.metadata["constraint"]``), so
    the profile is extracted against the same rule the annotator routes on.
    """
    from auto_treatise.cases import extract_case_profiles
    from auto_treatise.paragraph_pipeline import prepare_segmentations

    seg = load_config(doctrine)["segmentation"]
    cols = cache_columns(model)
    db = str(data.db_path(doctrine))
    ids = [int(c) for c in case_ids]

    print(
        f"""[segment] {doctrine}  {len(ids)} case(s)  model={model}  columns={cols['relevance']}, {cols['segmentation']}, {cols['profile']}"""
    )
    prepare_segmentations(
        query=seg["query"],
        db_path=db,
        ids=ids,
        relevance_cache_column=cols["relevance"],
        segmentation_cache_column=cols["segmentation"],
        majority_only=seg["majority_only"],
        model=model,
    )
    extract_case_profiles(
        constraint=constraint,
        db_path=db,
        cache_column=cols["profile"],
        ids=ids,
        segmentation_cache_column=cols["segmentation"],
        relevance_cache_column=cols["relevance"],
        model=model,
    )

    _, status = segmented_views(doctrine, case_ids, model)
    print(f"[segment] {doctrine}  status={dict(Counter(status.values()))}")


def _read(conn: sqlite3.Connection, column: str, case_id: str) -> Any:
    """The cached JSON in ``column`` for one case, or ``None`` if not run."""
    row = conn.execute(
        f"SELECT {column} FROM cases WHERE id = ?", (int(case_id),)
    ).fetchone()
    return json.loads(row[0]) if row[0] else None


def render(profile: dict[str, Any], segmented: list[dict]) -> str:
    """Profile first (the tree's own summary of a case), then the
    labelled evidence."""
    from auto_treatise.utils import format_segmented_paragraphs

    def field(key: str) -> str:
        value = profile[key]
        return str(value).strip() if value is not None else ""

    lines = [
        "## CONSTRAINT PROFILE",
        f"Issue: {field('issue_summary')}",
        f"Material facts: {field('facts_summary')}",
        f"Application: {field('application_summary')}",
        f"Reasoning chain: {field('reasoning_chain')}",
        f"Truth value: {field('truth_value')}    Applies constraint: {field('applies_constraint')}    Disposition: {field('output_type')}",
    ]
    quotes = profile["relevant_texts"]
    if quotes:
        lines.append("Key passages:")
        lines.extend(
            f'  - "{str(q).strip()}"' for q in quotes if str(q).strip()
        )
    lines += [
        "",
        "## LABELED RELEVANT PARAGRAPHS",
        format_segmented_paragraphs(segmented),
    ]
    return "\n".join(lines)


def segmented_views(
    doctrine: str, case_ids: list[str], model: str
) -> tuple[dict[str, str], dict[str, str]]:
    """``({case_id: rendered view}, {case_id: status})`` from the
    annotator columns.
    """
    cols = cache_columns(model)
    views: dict[str, str] = {}
    status: dict[str, str] = {}
    conn = sqlite3.connect(f"file:{data.db_path(doctrine)}?mode=ro", uri=True)
    for case_id in case_ids:
        segmented = _read(conn, cols["segmentation"], case_id)
        if segmented is None:
            status[case_id] = "unsegmented"
            continue
        if not segmented:
            status[case_id] = "no_relevant_paragraphs"
            continue
        profile = _read(conn, cols["profile"], case_id)
        if profile is None:
            status[case_id] = "no_profile"
            continue
        views[case_id] = render(profile, segmented)
        status[case_id] = "ok"
    conn.close()
    return views, status


# --------------------------------------------------------------------------- #
# Annotation: forked multi-role descent
# --------------------------------------------------------------------------- #


def _ordered_roles(roles) -> tuple[str, ...]:
    return tuple(r for r in schemas.ANNOTATION_ROLES if r in roles)


def _role_terminal(
    leaf: str | None,
    path: list[dict[str, Any]],
    not_pressed: bool = False,
    role: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"leaf": leaf, "path": path, "role": role}
    if not_pressed:
        entry["not_pressed"] = True
    return entry


def _build_annotation_round_items(
    tree: data.Tree,
    constraint: str,
    walkers: list[dict[str, Any]],
    headers: dict[str, str],
    views: dict[str, str],
    view_kinds: dict[str, str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for walker in walkers:
        case_id = walker["case_id"]
        parts = prompts.build_annotation_prompt(
            tree,
            walker["node_id"],
            constraint,
            headers[case_id],
            views[case_id],
            roles=walker["roles"],
            party_names=walker["party_names"] or None,
            view_kind=view_kinds[case_id],
        )
        items.append(
            {
                "content": parts.suffix,
                "system_prompt": prompts.ANNOTATOR_SYSTEM,
                "cache_prefix": parts.prefix,
            }
        )
    return items


def _fork_annotation(
    tree: data.Tree, walker: dict[str, Any], decision: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Advance one multi-role walker.
    Returns continuing walkers, new terminals, names."""
    roles = walker["roles"]
    party_names = dict(walker["party_names"])
    paths = {r: list(walker["paths"][r]) for r in roles}

    for role in ("party_a", "party_b"):
        # Only the first decision naming a party carries a ``role`` field.
        if role in decision and "role" in decision[role]:
            name = str(decision[role]["role"]).strip()
            if name:
                party_names[role] = name

    left, right = tree.children(walker["node_id"])
    terminals: dict[str, Any] = {}
    continuing: dict[str, list[str]] = {}

    for role in roles:
        block = decision[role]
        allowed = ("left", "right")
        if role != "court":
            allowed += ("not_pressed",)
        if block["side"] not in allowed:
            raise ValueError(
                f"""invalid side {block['side']!r} for {role} at {walker['node_id']}"""
            )
        step = {
            "node_id": walker["node_id"],
            "side": block["side"],
            "rationale": block["rationale"],
            "confidence": block["confidence"],
        }
        party_role = party_names[role] if role in party_names else None
        if block["side"] == "not_pressed":
            step["child_id"] = None
            paths[role].append(step)
            terminals[role] = _role_terminal(
                None, paths[role], not_pressed=True, role=party_role
            )
            continue
        nxt = left if block["side"] == "left" else right
        step["child_id"] = nxt["id"]
        paths[role].append(step)
        if nxt["is_leaf"]:
            terminals[role] = _role_terminal(
                nxt["id"], paths[role], role=party_role
            )
        else:
            continuing.setdefault(nxt["id"], []).append(role)

    next_walkers = []
    for child_id, child_roles in continuing.items():
        ordered = _ordered_roles(child_roles)
        next_walkers.append(
            {
                "case_id": walker["case_id"],
                "node_id": child_id,
                "roles": ordered,
                "party_names": party_names,
                "paths": {r: paths[r] for r in ordered},
            }
        )
    return next_walkers, terminals, party_names


def _annotation_result(
    case_id: str,
    terminals: dict[str, Any],
    party_names: dict[str, str],
) -> dict[str, Any]:
    """One labels-file row: court in ``leaf``/``path``, parties alongside."""
    court = terminals["court"]
    row: dict[str, Any] = {
        "case_id": case_id,
        "leaf": court["leaf"],
        "path": court["path"],
        "parties": {},
    }
    for key in ("party_a", "party_b"):
        t = terminals[key]
        entry: dict[str, Any] = {
            "role": party_names[key] if key in party_names else t["role"],
            "leaf": t["leaf"],
            "path": t["path"],
        }
        if "not_pressed" in t:
            entry["not_pressed"] = True
        row["parties"][key] = entry
    return row


def descend_annotation_many(
    tree: data.Tree,
    constraint: str,
    case_ids: list[str],
    model: str,
    headers: dict[str, str],
    views: dict[str, str],
    view_kinds: dict[str, str],
) -> list[dict[str, Any]]:
    """BFS-route annotation walkers, forking when court and parties diverge.
    """
    pending: list[dict[str, Any]] = []
    finished_rows: list[dict[str, Any]] = []
    case_state: dict[str, dict[str, Any]] = {}

    for case_id in case_ids:
        case_state[case_id] = {"party_names": {}, "terminals": {}}
        node = tree.root
        if node["is_leaf"]:
            terminals = {
                r: _role_terminal(node["id"], [])
                for r in schemas.ANNOTATION_ROLES
            }
            finished_rows.append(_annotation_result(case_id, terminals, {}))
            continue
        pending.append(
            {
                "case_id": case_id,
                "node_id": node["id"],
                "roles": schemas.ANNOTATION_ROLES,
                "party_names": {},
                "paths": {r: [] for r in schemas.ANNOTATION_ROLES},
            }
        )

    round_i = 0
    while pending:
        round_i += 1
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for walker in pending:
            roles = walker["roles"]
            named = tuple(
                r for r in roles if r != "court" and r in walker["party_names"]
            )
            groups.setdefault((roles, named), []).append(walker)

        next_pending: list[dict[str, Any]] = []
        newly_complete: list[dict[str, Any]] = []
        n_this_round = 0

        for (roles, named), group in groups.items():
            schema = schemas.annotation_routing_schema(
                roles, named_parties=named
            )
            items = _build_annotation_round_items(
                tree, constraint, group, headers, views, view_kinds
            )
            decisions = _route_round(items, model, schema)

            n_this_round += len(group)
            for walker, decision in zip(group, decisions):
                continuing, terminals, party_names = _fork_annotation(
                    tree, walker, decision
                )
                next_pending.extend(continuing)
                state = case_state[walker["case_id"]]
                state["party_names"].update(party_names)
                state["terminals"].update(terminals)
                if set(state["terminals"]) == set(schemas.ANNOTATION_ROLES):
                    newly_complete.append(
                        _annotation_result(
                            walker["case_id"],
                            state["terminals"],
                            state["party_names"],
                        )
                    )

        finished_rows.extend(newly_complete)
        pending = next_pending
        print(
            f"""[descend] round {round_i}: {len(newly_complete)} case(s) finished, {len(pending)} walkers pending ({n_this_round} calls)"""
        )
    return finished_rows


# --------------------------------------------------------------------------- #
# Pass 1: ground-truth labels
# --------------------------------------------------------------------------- #


def run_annotation(
    doctrine: str,
    run_id: str,
    case_ids: list[str],
    model: str,
) -> dict[str, Any]:
    """Label each held-out case that has an extracted ``x^[u]``.
    """
    tree, constraint, accepted, meta = _prepare(doctrine, run_id, case_ids)
    extracted = data.fact_views(doctrine, accepted)
    majority = data.majority_views(doctrine, accepted)
    ids = [c for c in accepted if c in extracted and c in majority]
    views = {c: majority[c] for c in ids}
    view_kinds = {c: "majority" for c in ids}
    seg_views, view_status = segmented_views(doctrine, ids, model)
    unsegmented = [c for c in ids if view_status[c] == "unsegmented"]
    if unsegmented:
        raise SystemExit(
            f"""[annotate] {len(unsegmented)} of {len(ids)} case(s) have no {model} segmentation yet; run --stage segment first (e.g. {', '.join(unsegmented[:5])})"""
        )
    for c in seg_views:
        views[c] = seg_views[c]
        view_kinds[c] = "segmented_profile"
    n_seg = len(seg_views)
    print(
        f"""[annotate] {n_seg} segmented, {len(ids) - n_seg} on the majority-opinion fallback"""
    )
    headers = {c: data.case_header(meta[c]) for c in ids}
    path = data.labels_path(doctrine, run_id)
    _pass_status("annotate", len(accepted), len(ids))

    rows = descend_annotation_many(
        tree,
        constraint,
        ids,
        model,
        headers,
        views,
        view_kinds,
    )
    for row in rows:
        case_id = row["case_id"]
        row["view_used"] = (
            "segmented_profile"
            if view_kinds[case_id] == "segmented_profile"
            else "majority_fallback"
        )
        if view_status[case_id] != "ok":
            row["fallback_reason"] = view_status[case_id]
    payload = {
        "meta": _meta(
            doctrine,
            run_id,
            tree,
            annotator_model=model,
            n_labelled=len(rows),
            n_view_segmented=n_seg,
            n_view_majority_fallback=len(ids) - n_seg,
        ),
        "labels": _sort_case_rows(rows),
    }
    _write(path, payload)
    return payload
