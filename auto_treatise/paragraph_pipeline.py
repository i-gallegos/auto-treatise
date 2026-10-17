"""Relevance filtering and sentence segmentation of majority opinions, cached per row."""

import json
import sqlite3
from typing import Any

import api_calls

from . import majority
from . import utils
from .majority import isolate_majority_paragraphs
from .paragraph_prompts import _build_relevance_payload
from .paragraph_prompts import _chunk_relevant_for_segmentation
from .paragraph_prompts import _merge_segmented_paragraphs
from .paragraph_prompts import _no_majority_relevance_payload
from .paragraph_prompts import _relevance_prompt_batch
from .paragraph_prompts import _seal_without_segmentation
from .paragraph_prompts import _segmentation_blocks
from .paragraph_prompts import _segmentation_prompt_batch
from .paragraph_prompts import _target_char_count
from .paragraph_prompts import paragraphs_for_segmentation
from .paragraph_prompts import RELEVANCE_SCHEMA_BATCH
from .paragraph_prompts import RELEVANCE_SYSTEM
from .paragraph_prompts import SEGMENTATION_SCHEMA_BATCH
from .paragraph_prompts import SEGMENTATION_SYSTEM
from .utils import cache_read
from .utils import cache_write
from .utils import SEGMENTATION_TARGET_CHAR_BUDGET


def ids_with_relevant_segmentation(
    db_path: str,
    ids: list[int] | None = None,
    segmentation_cache_column: str = "paragraph_sentence_segmentation_cache_json",
) -> list[int]:
    """
    Case ids whose segmentation cache is present and non-empty.

    Empty `[]` means the segmenter ran and found no paragraphs relevant to the
    query — those rows are processed for caching, but they must not enter
    dendrogram induction (no labeled text to split on).
    """
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cases)").fetchall()]
    if segmentation_cache_column not in cols:
        conn.close()
        return []

    if ids is None:
        rows = conn.execute(
            f"SELECT id, {segmentation_cache_column} FROM cases"
        ).fetchall()
    else:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""SELECT id, {segmentation_cache_column} FROM cases WHERE id IN ({placeholders})""",
            ids,
        ).fetchall()
    conn.close()

    kept: list[int] = []
    for row_id, raw in rows:
        if not raw:
            continue
        payload = json.loads(raw)
        if payload:
            kept.append(row_id)
    return kept


def ids_with_relevant_paragraphs(
    db_path: str,
    ids: list[int] | None = None,
    relevance_cache_column: str = "paragraph_relevance_cache_json",
) -> list[int]:
    """Case ids whose relevance cache kept at least one paragraph.
    """
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cases)").fetchall()]
    if relevance_cache_column not in cols:
        conn.close()
        return []

    if ids is None:
        rows = conn.execute(
            f"SELECT id, {relevance_cache_column} FROM cases"
        ).fetchall()
    else:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""SELECT id, {relevance_cache_column} FROM cases WHERE id IN ({placeholders})""",
            ids,
        ).fetchall()
    conn.close()

    kept: list[int] = []
    for row_id, raw in rows:
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict) and paragraphs_for_segmentation(payload):
            kept.append(row_id)
    return kept


def _select_cases(
    db_path: str,
    ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """``{id, case_name, citation}`` rows, in ``ids`` order or sorted by id.
    """
    if ids is not None:
        # Preserve user-provided order while removing duplicates.
        requested_ids = list(dict.fromkeys(ids))
        loaded = utils.load_cases(db_path, ids=requested_ids)
        by_id = {int(c["id"]): c for c in loaded}

        missing_ids = [n for n in requested_ids if int(n) not in by_id]
        for n in missing_ids:
            print(f"[MISSING id] {n}")

        selected_cases = [
            {
                "id": by_id[int(n)]["id"],
                "case_name": by_id[int(n)]["case_name"],
                "citation": by_id[int(n)]["citation"],
            }
            for n in requested_ids
            if int(n) in by_id
        ]
        print(
            f"""[segmenter] id subset: {len(selected_cases)} cases ({len(requested_ids)} requested)"""
        )
        return selected_cases

    loaded = utils.load_cases(db_path, ids=ids)
    selected_cases = [
        {
            "id": c["id"],
            "case_name": c["case_name"],
            "citation": c["citation"],
        }
        for c in loaded
    ]
    return sorted(selected_cases, key=lambda x: int(x["id"]))


def prepare_segmentations(
    query: str,
    db_path: str,
    model: str,
    ids: list[int] | None = None,
    context_instructions: str = "",
    relevance_cache_column: str = "paragraph_relevance_cache_json",
    segmentation_cache_column: str = "paragraph_sentence_segmentation_cache_json",
    majority_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Run relevance selection + segmentation over many cases.

    With majority_only=True, concurrence and dissent paragraphs are dropped
    before prompting so only the court's own reasoning is labeled. Rows whose
    majority isolation status is ``no_majority`` never reach the relevance or
    segmentation batches; synthetic caches are written instead. After relevance,
    rows with no paragraphs to segment (off-query or none relevant) are sealed
    the same way, overwriting a stale non-empty segmentation cache on rerun.
    """
    selected_cases = _select_cases(db_path=db_path, ids=ids)

    if not selected_cases:
        return []

    results_index: dict[int, dict[str, Any]] = {}
    relevance_index: dict[int, list[dict[str, Any]]] = {}

    relevance_pending: list[dict[str, Any]] = []
    segmentation_pending: list[dict[str, Any]] = []

    # Stage 0: majority isolation — must precede relevance so that concurrence
    # and dissent paragraphs are never classified or labeled.
    paragraph_maps, case_status = isolate_majority_paragraphs(
        db_path, selected_cases, majority_only
    )
    no_majority_ids = {
        case["id"]
        for case in selected_cases
        if case_status[case["id"]] == majority.STATUS_NO_MAJORITY
    }
    if no_majority_ids:
        print(
            f"""[segmenter] excluding {len(no_majority_ids)} case(s) with no majority opinion from relevance and segmentation"""
        )
        conn = sqlite3.connect(db_path)
        for case in selected_cases:
            if case["id"] not in no_majority_ids:
                continue
            cache_write(
                conn,
                relevance_cache_column,
                case["id"],
                _no_majority_relevance_payload(query),
            )
        conn.close()
        for case in selected_cases:
            if case["id"] not in no_majority_ids:
                continue
            relevance_index[case["id"]] = []
            _seal_without_segmentation(
                db_path,
                case,
                segmentation_cache_column,
                relevance_index,
                results_index,
                "no_majority",
            )

    for case in selected_cases:
        id = case["id"]
        if id in no_majority_ids:
            continue
        paragraph_map = paragraph_maps[id]

        conn = sqlite3.connect(db_path)
        relevance_cached = cache_read(conn, relevance_cache_column, id)
        segmentation_cached = cache_read(conn, segmentation_cache_column, id)
        conn.close()

        if relevance_cached is not None:
            relevance_index[id] = paragraphs_for_segmentation(relevance_cached)
        else:
            relevance_pending.append(
                {
                    "id": id,
                    "case_name": case["case_name"],
                    "citation": case["citation"],
                    "paragraph_map": paragraph_map,
                }
            )

        # `[]` is a real cache hit (no relevant paragraphs) — do not re-queue.
        if segmentation_cached is not None:
            results_index[id] = {
                "id": id,
                "case_name": case["case_name"],
                "citation": case["citation"],
                "relevant_paragraphs": [],  # filled from relevance_index below
                "segmented_paragraphs": segmentation_cached,
            }
        else:
            segmentation_pending.append(
                {
                    "id": id,
                    "case_name": case["case_name"],
                    "citation": case["citation"],
                    "paragraph_map": paragraph_map,
                }
            )

    # Stage 1: relevance
    if relevance_pending:
        relevance_items = [
            {
                "content": _relevance_prompt_batch(
                    query=query,
                    paragraph_map=p["paragraph_map"],
                    context_instructions=context_instructions,
                ),
                "system_prompt": RELEVANCE_SYSTEM,
            }
            for p in relevance_pending
        ]
        relevance_responses = api_calls.batch_query_llm(
            relevance_items,
            model=model,
            response_format=RELEVANCE_SCHEMA_BATCH,
        )
        for pending_case, (_, response) in zip(
            relevance_pending, relevance_responses
        ):
            relevant, payload = _build_relevance_payload(
                query=query,
                paragraph_map=pending_case["paragraph_map"],
                response=response,
            )
            conn = sqlite3.connect(db_path)
            cache_write(
                conn, relevance_cache_column, pending_case["id"], payload
            )
            conn.close()
            relevance_index[pending_case["id"]] = relevant

    # Backfill relevance for rows where segmentation was already cached.
    for id in results_index:
        if id in relevance_index:
            results_index[id]["relevant_paragraphs"] = relevance_index[id]

    # After relevance: rows with nothing to segment do not proceed, even when a
    # prior run left a non-empty segmentation cache.
    empty_relevance_cases = [
        case
        for case in selected_cases
        if case["id"] not in no_majority_ids
        and case["id"] in relevance_index
        and not relevance_index[case["id"]]
    ]
    if empty_relevance_cases:
        empty_relevance_ids = {case["id"] for case in empty_relevance_cases}
        segmentation_pending = [
            pending
            for pending in segmentation_pending
            if pending["id"] not in empty_relevance_ids
        ]
        print(
            f"""[segmenter] excluding {len(empty_relevance_cases)} case(s) with no relevant paragraphs from segmentation"""
        )
        for case in empty_relevance_cases:
            _seal_without_segmentation(
                db_path,
                case,
                segmentation_cache_column,
                relevance_index,
                results_index,
                "no_relevant_paragraphs",
            )

    # Stage 2: segmentation
    if segmentation_pending:
        segmentation_items: list[dict[str, str]] = []
        item_case_ids: list[int] = []
        n_chunked_cases = 0
        n_chunked_requests = 0

        for pending_case in segmentation_pending:
            id = pending_case["id"]
            relevant = relevance_index[id]
            paragraph_map = pending_case["paragraph_map"]

            chunks = _chunk_relevant_for_segmentation(relevant, paragraph_map)
            if len(chunks) > 1:
                n_chunked_cases += 1
                n_chunked_requests += len(chunks)
                print(
                    f"""[segmenter] id={id} ({pending_case['case_name']}): {_target_char_count(relevant, paragraph_map)} TARGET chars; splitting into {len(chunks)} segmentation requests"""
                )

            for chunk in chunks:
                segmentation_items.append(
                    {
                        "content": _segmentation_prompt_batch(
                            _segmentation_blocks(chunk, paragraph_map)
                        ),
                        "system_prompt": SEGMENTATION_SYSTEM,
                    }
                )
                item_case_ids.append(id)

        if n_chunked_cases:
            print(
                f"""[segmenter] split {n_chunked_cases} long case(s) into {n_chunked_requests} segmentation requests (TARGET budget={SEGMENTATION_TARGET_CHAR_BUDGET} chars)"""
            )

        if segmentation_items:
            segmentation_responses = api_calls.batch_query_llm(
                segmentation_items,
                model=model,
                response_format=SEGMENTATION_SCHEMA_BATCH,
            )

            parts: dict[int, list] = {}
            for case_id, (_, response) in zip(
                item_case_ids, segmentation_responses
            ):
                parts.setdefault(case_id, []).append(response["paragraphs"])
            for pending_case in segmentation_pending:
                case_id = pending_case["id"]
                segmented = _merge_segmented_paragraphs(parts[case_id])
                conn = sqlite3.connect(db_path)
                cache_write(conn, segmentation_cache_column, case_id, segmented)
                conn.close()
                results_index[case_id] = {
                    "id": case_id,
                    "case_name": pending_case["case_name"],
                    "citation": pending_case["citation"],
                    "relevant_paragraphs": relevance_index[case_id],
                    "segmented_paragraphs": segmented,
                }

    return [results_index[c["id"]] for c in selected_cases]
