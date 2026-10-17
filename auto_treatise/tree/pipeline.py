"""End-to-end tree build: profiles, induction and run artifacts."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from ..case_index import build_case_index
from ..case_index import opinion_id_by_case_id
from ..case_index import write_case_index
from ..cases import extract_case_profiles
from ..cases import INPUT_MODES
from ..cases import load_induction_texts
from ..cases import make_run_id
from ..cases import profile_applies_constraint
from ..cases import profile_cache_column
from ..cases import raw_case_row
from ..majority import ids_without_majority_opinion
from ..majority import majority_exclusion
from ..paragraph_prompts import payload_engagement
from ..paragraph_prompts import relevance_exclusion
from ..paragraph_pipeline import ids_with_relevant_paragraphs
from ..paragraph_pipeline import ids_with_relevant_segmentation
from ..utils import cache_read
from .config import MAX_RECURSION_DEPTH
from .expand import induce_grammar


def merge_missing_case_profiles(
    case_path: Path,
    rows: list[dict[str, Any]],
) -> list[str]:
    """Append profiles whose ids are absent from ``case_path``.
    """
    existing = json.loads(case_path.read_text())
    have = {str(row["id"]) for row in existing}
    added = []
    for row in rows:
        if str(row["id"]) not in have:
            have.add(str(row["id"]))
            added.append(row)
    if added:
        case_path.write_text(
            json.dumps(existing + added, indent=2, ensure_ascii=False)
        )
    return [str(row["id"]) for row in added]


def partition_for_induction(
    case_profiles: list[dict[str, Any]],
    *,
    input_mode: str,
    no_majority_ids: list,
    eligible_ids: set,
) -> dict[str, list]:
    """Choose which profiled cases enter dendrogram induction.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
        )
    if input_mode == "full":
        no_majority_ids = []
        eligible_ids = {row["id"] for row in case_profiles}
    no_majority_set = set(no_majority_ids)
    off_constraint_ids = [
        row["id"]
        for row in case_profiles
        if row["id"] in eligible_ids
        and row["id"] not in no_majority_set
        and not profile_applies_constraint(row["profile"])
    ]
    off_constraint_set = set(off_constraint_ids)
    tree_profiles = [
        row
        for row in case_profiles
        if row["id"] in eligible_ids
        and row["id"] not in off_constraint_set
        and row["id"] not in no_majority_set
    ]
    excluded_ids = [
        row["id"]
        for row in case_profiles
        if row["id"] not in eligible_ids and row["id"] not in no_majority_set
    ]
    return {
        "no_majority_ids": list(no_majority_ids),
        "excluded_ids": excluded_ids,
        "off_constraint_ids": off_constraint_ids,
        "tree_profiles": tree_profiles,
    }


def run_pipeline(
    doctrine: str,
    constraint: str,
    model: str,
    analysis_root: Path,
    subset_ids: list[int] | None = None,
    segmentation_cache_column: str = "paragraph_sentence_segmentation_cache_json",
    run_type: str | None = None,
    max_recursion_depth: int = MAX_RECURSION_DEPTH,
    run_id: str | None = None,
    input_mode: str = "segmented",
    use_profile: bool = True,
    relevance_cache_column: str = "paragraph_relevance_cache_json",
    profile_column: str | None = None,
    share_case_level: bool = True,
) -> dict[str, Any]:
    """
    Full two-pass pipeline:
      1. Extract per-case constraint profiles.
      2. Induce a recursive binary case-split dendrogram.
    """
    # File paths
    doctrine_root = Path(f"doctrines/{doctrine}")
    db_path = doctrine_root / "data" / "citing_cases_clean.db"

    analysis_dir = Path(analysis_root)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
        )

    # Run metadata
    if run_type is None:
        run_type = "test" if subset_ids is not None else "full"
    run_date = date.today().isoformat()
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    if not model_name:
        model_name = "model"
    run_id = run_id or make_run_id(
        run_date, model_name, run_type, input_mode, use_profile
    )
    run_dir = analysis_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if share_case_level and input_mode == "segmented" and use_profile:
        case_path = analysis_dir / "case_level.json"
    else:
        case_path = run_dir / "case_level.json"
    scope = (
        f"{len(subset_ids)}-case subset"
        if subset_ids is not None
        else "full corpus"
    )

    # Run pipeline
    case_profile_col = None
    if use_profile:
        case_profile_col = profile_column or profile_cache_column(
            model, input_mode
        )
        print(
            f"""[tree_inducer] Pass 1: extracting case profiles ({scope})  input_mode={input_mode}  cache={case_profile_col!r} ..."""
        )
        case_profiles = extract_case_profiles(
            constraint=constraint,
            db_path=db_path,
            cache_column=case_profile_col,
            ids=subset_ids,
            segmentation_cache_column=segmentation_cache_column,
            relevance_cache_column=relevance_cache_column,
            model=model,
            input_mode=input_mode,
        )
    else:
        print(
            f"""[tree_inducer] Pass 1 skipped ({scope}): splitter reads {input_mode} text, no constraint profile"""
        )
        case_profiles = [
            raw_case_row(row["id"], row["text"])
            for row in load_induction_texts(
                str(db_path),
                subset_ids,
                input_mode,
                relevance_cache_column=relevance_cache_column,
                segmentation_cache_column=segmentation_cache_column,
            )
        ]
    print(
        f"[tree_inducer] Pass 1 complete: {len(case_profiles)} cases profiled."
    )

    case_index = build_case_index(db_path, [row["id"] for row in case_profiles])
    opinion_ids = opinion_id_by_case_id(case_index)

    case_profiles_for_disk = [
        {
            "id": str(row["id"]),
            "opinion_id": opinion_ids[str(row["id"])],
            "profile": row["profile"],
        }
        for row in case_profiles
    ]
    should_write_case_level = run_type == "full" or not case_path.exists()
    if should_write_case_level:
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.write_text(
            json.dumps(case_profiles_for_disk, indent=2, ensure_ascii=False)
        )
        print(f"[tree_inducer] Updated case-level cache: {case_path}")
    else:
        added_ids = merge_missing_case_profiles(
            case_path, case_profiles_for_disk
        )
        if added_ids:
            print(
                f"""[tree_inducer] Merged {len(added_ids)} profile(s) missing from case-level cache: {case_path}"""
            )

    profile_ids = [row["id"] for row in case_profiles]

    gate_mode = "full" if not use_profile else input_mode
    if gate_mode == "full":
        no_majority_ids: list = []
        eligible_ids: set = set()
    else:
        no_majority_ids = ids_without_majority_opinion(
            db_path=str(db_path),
            ids=profile_ids,
        )
        if gate_mode == "relevant":
            eligible_ids = set(
                ids_with_relevant_paragraphs(
                    db_path=str(db_path),
                    ids=profile_ids,
                    relevance_cache_column=relevance_cache_column,
                )
            )
        else:
            eligible_ids = set(
                ids_with_relevant_segmentation(
                    db_path=str(db_path),
                    ids=profile_ids,
                    segmentation_cache_column=segmentation_cache_column,
                )
            )
    parts = partition_for_induction(
        case_profiles,
        input_mode=gate_mode,
        no_majority_ids=no_majority_ids,
        eligible_ids=eligible_ids,
    )
    no_majority_ids = parts["no_majority_ids"]
    excluded_ids = parts["excluded_ids"]
    off_constraint_ids = parts["off_constraint_ids"]
    tree_profiles = parts["tree_profiles"]
    if no_majority_ids:
        print(
            f"""[tree_inducer] excluding {len(no_majority_ids)} case(s) with no majority opinion from dendrogram induction"""
        )
    if excluded_ids:
        print(
            f"""[tree_inducer] excluding {len(excluded_ids)} case(s) with no relevant paragraphs from dendrogram induction"""
        )
    if off_constraint_ids:
        print(
            f"""[tree_inducer] excluding {len(off_constraint_ids)} case(s) the profiler flagged as off-query from dendrogram induction"""
        )

    print(
        "[tree_inducer] Pass 2: "
        + "inducing"
        + " binary dendrogram (not cached; written to the run directory) ..."
    )
    grammar = induce_grammar(
        constraint=constraint,
        case_profiles=tree_profiles,
        model=model,
        max_recursion_depth=max_recursion_depth,
    )
    print(
        f"[tree_inducer] Pass 2 complete: {len(grammar['nodes'])} dendrogram nodes."
    )

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_json = json.dumps(grammar, indent=2, ensure_ascii=False)
    corpus_path = run_dir / "corpus_level.json"
    metadata_path = run_dir / "tree_run_metadata.json"
    unassigned_path = run_dir / "unassigned_case_ids.json"
    corpus_path.write_text(corpus_json)

    case_index_path = write_case_index(case_index, run_dir)

    unassigned_records: list[dict[str, Any]] = []
    majority_by_id: dict[Any, Any] = {}
    if no_majority_ids:
        conn = sqlite3.connect(db_path)
        for case_id in no_majority_ids:
            majority_by_id[case_id] = cache_read(
                conn, "majority_boundary_json", case_id
            )
        conn.close()
    for case_id in no_majority_ids:
        reason, explanation = majority_exclusion(majority_by_id[case_id])
        unassigned_records.append(
            {
                "id": str(case_id),
                "reason": reason,
                "explanation": explanation,
                "node_id": None,
                "is_leaf": False,
            }
        )
    relevance_by_id: dict[Any, Any] = {}
    if excluded_ids:
        conn = sqlite3.connect(db_path)
        for case_id in excluded_ids:
            relevance_by_id[case_id] = cache_read(
                conn, "paragraph_relevance_cache_json", case_id
            )
        conn.close()
    for case_id in excluded_ids:
        reason, explanation = relevance_exclusion(relevance_by_id[case_id])
        unassigned_records.append(
            {
                "id": str(case_id),
                "reason": reason,
                "explanation": explanation,
                "stage": "relevance",
                "query_engagement": payload_engagement(
                    relevance_by_id[case_id]
                ),
                "node_id": None,
                "is_leaf": False,
            }
        )
    profile_by_id = {row["id"]: row["profile"] for row in case_profiles}
    for case_id in off_constraint_ids:
        profile = profile_by_id[case_id]
        explanation = (
            profile["nonapplication_reason"].strip()
            if "nonapplication_reason" in profile
            else ""
        )
        if not explanation:
            explanation = """Constraint profile marked the opinion as being about a different legal question; the case was excluded from dendrogram induction."""
        unassigned_records.append(
            {
                "id": str(case_id),
                "reason": "off_query",
                "explanation": explanation,
                "stage": "profile",
                "query_engagement": "off_query",
                "node_id": None,
                "is_leaf": False,
            }
        )
    for node in grammar["nodes"]:
        unassigned = (
            node["unassigned_cases"] if "unassigned_cases" in node else []
        )
        if not unassigned:
            continue
        leaf_reason = (
            node["leaf_reason"]
            if "leaf_reason" in node
            else "unassigned_at_node"
        )
        for case_id in unassigned:
            unassigned_records.append(
                {
                    "id": str(case_id),
                    "reason": leaf_reason,
                    "explanation": f"""Case could not be placed on either child of node {node['id']} during tree induction ({leaf_reason}).""",
                    "node_id": node["id"],
                    "is_leaf": node["is_leaf"],
                }
            )
    unassigned_path.write_text(
        json.dumps(unassigned_records, indent=2, ensure_ascii=False)
    )

    metadata = {
        "run_id": run_id,
        "run_date": run_date,
        "model_name": model_name,
        "run_type": run_type,
        "max_recursion_depth": max_recursion_depth,
        "doctrine": doctrine,
        "constraint": constraint,
        "db_path": str(Path(db_path).resolve()),
        "scope": scope,
        "corpus_level_path": str(corpus_path),
        "case_level_path": str(case_path),
        "unassigned_case_ids_path": str(unassigned_path),
        "unassigned_case_count": len(unassigned_records),
        "case_index_path": str(case_index_path),
        "input_mode": input_mode,
        "use_profile": use_profile,
        "profile_cache_column": case_profile_col,
        "relevance_cache_column": relevance_cache_column,
        "segmentation_cache_column": segmentation_cache_column,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    level_where = (
        "shared"
        if case_path == analysis_dir / "case_level.json"
        else "run-local"
    )
    print(
        f"""[tree_inducer] Wrote run artifacts: {corpus_path}, {metadata_path}, {unassigned_path}, {case_index_path}; {level_where} case-level={case_path}"""
    )
    if input_mode != "segmented":
        print(
            f"""[tree_inducer] ablation input_mode={input_mode} run_id={run_id}
  Gemini 2.5 Pro labels this tree, then score against those labels:
  python -m generalization --stage all --doctrine {doctrine} --run-id {run_id}"""
        )
