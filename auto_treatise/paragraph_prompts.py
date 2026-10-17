"""Relevance and segmentation prompts."""

import sqlite3
from typing import Any


from . import utils
from .utils import cache_write
from .utils import SEGMENTATION_TARGET_CHAR_BUDGET

# ── Schemas ───────────────────────────────────────────────────────────────────
# Field descriptions. The schema and the relevance rules both use these.
QUERY_ENGAGEMENT_DESC = """Rate how this court engages the query. 'presented': the court frames the query as a question presented in THIS case and either resolves it or expressly declines to resolve it. 'adjacent': the query's subject matter is live in this case and the court engages it, but the court resolves the dispute on another ground (precedent, procedural default, jurisdiction, a threshold or scope holding, or a different statute or clause), or never frames the query as the question it is deciding. 'off_query': this opinion is about a different legal question. The leading case associated with the query appears only as background or is analogized to a different question. The query's subject matter is not itself at issue."""
EXCLUSION_REASON_DESC = """If query_engagement is 'off_query', use one sentence stating which legal question the opinion is about. If it is 'adjacent', use one sentence stating the ground the court resolved the case on instead. If it is 'presented', use an empty string."""


SENTENCE_ITEM = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "attribution": {
            "type": "string",
            "enum": [
                "Appellant",
                "Appellee",
                "Amicus",
                "LowerCourt",
                "CurrentCourt",
            ],
        },
        "role": {
            "type": "string",
            "enum": [
                "Facts",
                "Procedure",
                "Issue",
                "Argument",
                "Holding",
                "Reasoning",
            ],
        },
        "stance": {"type": "string", "enum": ["Disputed", "Undisputed"]},
    },
    "required": ["text", "attribution", "role", "stance"],
    "additionalProperties": False,
}


RELEVANCE_SCHEMA_BATCH = {
    "name": "paragraph_relevance",
    "schema": {
        "type": "object",
        "properties": {
            "query_engagement": {
                "type": "string",
                "enum": ["presented", "adjacent", "off_query"],
                "description": QUERY_ENGAGEMENT_DESC,
            },
            "exclusion_reason": {
                "type": "string",
                "description": EXCLUSION_REASON_DESC,
            },
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paragraph_index": {"type": "integer"},
                        "is_relevant": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["paragraph_index", "is_relevant", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["query_engagement", "exclusion_reason", "paragraphs"],
        "additionalProperties": False,
    },
    "strict": True,
}


SEGMENTATION_SCHEMA_BATCH = {
    "name": "paragraph_sentence_labels",
    "schema": {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paragraph_index": {"type": "integer"},
                        "sentences": {"type": "array", "items": SENTENCE_ITEM},
                    },
                    "required": ["paragraph_index", "sentences"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["paragraphs"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ── System prompts ────────────────────────────────────────────────────────────

RELEVANCE_SYSTEM = """You are a legal research assistant. Identify which paragraphs of a court opinion are relevant to a given query. A paragraph is relevant only when this court is treating the query as a question presented in THIS case, not merely because the opinion cites related authority."""


SEGMENTATION_SYSTEM = """You are a legal research assistant. Your job is to segment paragraphs of a court opinion into sentences and label each sentence with its attribution, rhetorical role, and whether the underlying proposition is disputed."""


# ── Prompts ───────────────────────────────────────────────────────────────────

_RELEVANCE_RULES = """A paragraph is relevant if it is part of this court's treatment of the query's subject matter in THIS case (facts, procedure, issue, argument, holding, or reasoning bearing on that subject matter).
Do not mark a paragraph relevant merely because it cites a leading case associated with the query, recites that case's holding as background, or analogizes from that holding to a different legal question.

Set query_engagement to exactly one of presented, adjacent, or off_query. """ + QUERY_ENGAGEMENT_DESC + """

Mark every paragraph not relevant only when query_engagement is "off_query". For "adjacent", still mark the paragraphs where this court engages the query's subject matter, including the paragraphs that explain why it is resolving the case on another ground.
Set exclusion_reason. """ + EXCLUSION_REASON_DESC


def _relevance_prompt_batch(
    query: str, paragraph_map: dict[int, str], context_instructions: str
) -> str:
    context_block = (
        f"\nAdditional instructions: {context_instructions}\n"
        if context_instructions
        else ""
    )
    return f"""Classify each paragraph as relevant or not to the query below.
A paragraph is relevant only if it materially helps answer the query.
{_RELEVANCE_RULES}
Return one entry per paragraph index; keep each reason to one sentence.
{context_block}
Query: {query}

Paragraphs:
{utils.display_paragraph_map(paragraph_map)}"""


def _segmentation_prompt_batch(blocks: list[str]) -> str:
    return f"""Segment each TARGET paragraph into individual sentences and label each with:
- attribution: who advances the proposition — Appellant, Appellee, Amicus, LowerCourt, or CurrentCourt
- role: Facts, Procedure, Issue, Argument, Holding, or Reasoning
- stance: whether the underlying proposition is Disputed or Undisputed

Rules: Holding sentences must be attributed to CurrentCourt. Party arguments must not be attributed to CurrentCourt.
Do not label CONTEXT paragraphs.

Paragraphs (TARGET = label these; CONTEXT = reference only):
{chr(10).join(blocks)}"""


def _target_char_count(
    relevant: list[dict[str, Any]],
    paragraph_map: dict[int, str],
) -> int:
    return sum(
        len(paragraph_map[int(row["paragraph_index"])]) for row in relevant
    )


def _segmentation_blocks(
    relevant: list[dict[str, Any]],
    paragraph_map: dict[int, str],
) -> list[str]:
    """CONTEXT / TARGET / CONTEXT blocks for one set of TARGET paragraphs.

    Neighbor CONTEXT is included even when that neighbor is a TARGET in another
    chunk, so a split case still sees the same local window as a one-shot
    prompt.
    """
    blocks: list[str] = []
    for row in relevant:
        i = int(row["paragraph_index"])
        if i - 1 in paragraph_map:
            blocks.append(f"[{i - 1} CONTEXT] {paragraph_map[i - 1]}")
        blocks.append(f"[{i} TARGET] {paragraph_map[i]}")
        if i + 1 in paragraph_map:
            blocks.append(f"[{i + 1} CONTEXT] {paragraph_map[i + 1]}")
        blocks.append("---")
    return blocks


def _chunk_relevant_for_segmentation(
    relevant: list[dict[str, Any]],
    paragraph_map: dict[int, str],
    budget: int = SEGMENTATION_TARGET_CHAR_BUDGET,
) -> list[list[dict[str, Any]]]:
    """Pack TARGET paragraphs into chunks whose TARGET text fits ``budget``.
    """
    sorted_relevant = sorted(relevant, key=lambda x: int(x["paragraph_index"]))
    if _target_char_count(sorted_relevant, paragraph_map) <= budget:
        return [sorted_relevant]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for row in sorted_relevant:
        n = len(paragraph_map[int(row["paragraph_index"])])
        if current and current_chars + n > budget:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += n
    if current:
        chunks.append(current)
    return chunks


def _merge_segmented_paragraphs(
    chunk_results: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Concatenate chunk outputs; first write wins on a repeated index."""
    by_index: dict[int, dict[str, Any]] = {}
    for paragraphs in chunk_results:
        for entry in paragraphs:
            idx = int(entry["paragraph_index"])
            if idx not in by_index:
                by_index[idx] = entry
    return [by_index[i] for i in sorted(by_index)]


# ── Main functions ────────────────────────────────────────────────────────────


def _build_relevance_payload(
    query: str,
    paragraph_map: dict[int, str],
    response: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the relevance cache payload.
    """
    by_index = {int(r["paragraph_index"]): r for r in response["paragraphs"]}
    all_decisions = [
        {
            "paragraph_index": i,
            "text": paragraph_map[i],
            "is_relevant": (
                by_index[i]["is_relevant"] if i in by_index else False
            ),
            "reason": (
                by_index[i]["reason"].strip()
                if i in by_index
                else "No classification returned."
            ),
        }
        for i in sorted(paragraph_map)
    ]
    engagement = response["query_engagement"]
    exclusion_reason = response["exclusion_reason"].strip()
    addresses_query = engagement in ("presented", "adjacent")
    if addresses_query:
        relevant = [r for r in all_decisions if r["is_relevant"]]
        if engagement == "presented":
            exclusion_reason = ""
    else:
        relevant = []
        if not exclusion_reason:
            exclusion_reason = "Opinion does not present or decide the query."
    payload = {
        "query": query,
        "query_engagement": engagement,
        "opinion_addresses_query": addresses_query,
        "exclusion_reason": exclusion_reason,
        "all_paragraph_decisions": all_decisions,
        "relevant_paragraphs": relevant,
    }
    return relevant, payload


def _no_majority_relevance_payload(query: str) -> dict[str, Any]:
    """Synthetic relevance cache for a row that has no lead opinion. No LLM."""
    return {
        "query": query,
        "query_engagement": "off_query",
        "opinion_addresses_query": False,
        "exclusion_reason": (
            "No majority opinion was identified for this row (status=no_majority)."
        ),
        "all_paragraph_decisions": [],
        "relevant_paragraphs": [],
    }


def _seal_without_segmentation(
    db_path: str,
    case: dict[str, Any],
    segmentation_cache_column: str,
    relevance_index: dict[int, list[dict[str, Any]]],
    results_index: dict[int, dict[str, Any]],
    reason: str,
) -> None:
    """Write an empty segmentation cache and record the case as done. No LLM."""
    case_id = case["id"]
    conn = sqlite3.connect(db_path)
    cache_write(conn, segmentation_cache_column, case_id, [])
    conn.close()
    relevant = relevance_index[case_id]
    results_index[case_id] = {
        "id": case_id,
        "case_name": case["case_name"],
        "citation": case["citation"],
        "relevant_paragraphs": relevant,
        "segmented_paragraphs": [],
    }
    print(
        f"""[segmenter] {reason} id={case_id} ({case['case_name']}); wrote empty segmentation cache"""
    )


def paragraphs_for_segmentation(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Relevant paragraphs that may proceed to segmentation.
    """
    if (
        "opinion_addresses_query" in payload
        and payload["opinion_addresses_query"] is False
    ):
        return []
    return payload["relevant_paragraphs"]


def payload_engagement(payload: dict[str, Any] | None) -> str:
    """``query_engagement`` of a relevance cache payload, or ``"unknown"``."""
    if payload is None:
        return "unknown"
    return payload["query_engagement"]


def relevance_exclusion(payload: dict[str, Any] | None) -> tuple[str, str]:
    """``(reason, explanation)`` for a case kept out of the tree at relevance.
    """
    if (
        payload is not None
        and "opinion_addresses_query" in payload
        and payload["opinion_addresses_query"] is False
    ):
        explanation = (
            payload["exclusion_reason"].strip()
            if "exclusion_reason" in payload
            else ""
        )
        if not explanation:
            explanation = """Opinion does not present or decide the query; excluded from dendrogram induction."""
        return "off_query", explanation
    return (
        "no_relevant_paragraphs",
        """Segmenter found no opinion paragraphs relevant to the constraint; the case was excluded from dendrogram induction.""",
    )
