import re
import sqlite3
from typing import Any, List, Optional

import api_calls

from . import majority
from . import paragraph_pipeline
from . import utils
from .paragraph_prompts import paragraphs_for_segmentation

# What the profiler reads. ``segmented`` is the labeled relevant paragraphs the
# published trees use. ``relevant`` is those same paragraphs as plain text.
# ``full`` is the stored opinion, with no relevance filter and no segmentation.
INPUT_MODES = ("segmented", "relevant", "full")

_INPUT_MODE_CACHE_TAG = {
    "segmented": "",
    "relevant": "_relevant",
    "full": "_opinion",
}

# Appended to ``{date}_{model}_{run_type}`` so an ablation cannot land in the
# segmented run directory.
_INPUT_MODE_RUN_SUFFIX = {
    "segmented": "",
    "relevant": "relevant",
    "full": "opinion",
}


def profile_cache_column(model: str, input_mode: str = "segmented") -> str:
    """Return the SQLite column name for *model*'s constraint profile cache.
    """
    if input_mode not in _INPUT_MODE_CACHE_TAG:
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
        )
    safe = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    tag = _INPUT_MODE_CACHE_TAG[input_mode]
    return f"case_constraint_profile_{safe}{tag}_json"


def make_run_id(
    run_date: str,
    model_name: str,
    run_type: str,
    input_mode: str = "segmented",
    use_profile: bool = True,
) -> str:
    """Run directory name.
    """
    if input_mode not in _INPUT_MODE_RUN_SUFFIX:
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
        )
    parts = [f"{run_date}_{model_name}_{run_type}"]
    suffix = _INPUT_MODE_RUN_SUFFIX[input_mode]
    if suffix:
        parts.append(suffix)
    if not use_profile:
        parts.append("noprofile")
    return "_".join(parts)


def raw_case_row(case_id: int | str, text: str) -> dict[str, Any]:
    """Use the opinion instead of a constraint profile.
    """
    excerpt = text[:12_000]
    return {
        "id": case_id,
        "raw_text": text,
        "profile": {
            "id": str(case_id),
            "relevant_texts": [excerpt] if excerpt else [],
            "issue_summary": "",
            "facts_summary": "",
            "application_summary": "",
            "reasoning_chain": "",
            "truth_value": "unaddressed",
            "applies_constraint": True,
            "nonapplication_reason": "",
            "output_type": "",
        },
    }


def load_induction_texts(
    db_path: str,
    ids: Optional[List[int]],
    input_mode: str,
    relevance_cache_column: str = "paragraph_relevance_cache_json",
    segmentation_cache_column: str = "paragraph_sentence_segmentation_cache_json",
) -> list[dict[str, Any]]:
    """Opinion text for a no-profile build.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
        )
    wanted = [int(i) for i in ids] if ids is not None else None
    if input_mode == "full":
        rows = []
        for case in utils.load_cases(db_path, ids=wanted):
            text = (case["opinion_text"] or "").strip()
            if text:
                rows.append({"id": case["id"], "text": text})
        return rows

    no_majority = set(
        majority.ids_without_majority_opinion(db_path, ids=wanted)
    )
    if input_mode == "segmented":
        eligible = paragraph_pipeline.ids_with_relevant_segmentation(
            db_path,
            ids=wanted,
            segmentation_cache_column=segmentation_cache_column,
        )
    else:
        eligible = paragraph_pipeline.ids_with_relevant_paragraphs(
            db_path, ids=wanted, relevance_cache_column=relevance_cache_column
        )
    conn = sqlite3.connect(db_path)
    rows = []
    for case_id in eligible:
        if case_id in no_majority:
            continue
        if input_mode == "segmented":
            segmented = utils.load_segmented_sentences(
                conn, case_id, segmentation_cache_column
            )
            text = utils.format_segmented_paragraphs(segmented)
        else:
            relevance = utils.cache_read(conn, relevance_cache_column, case_id)
            text = _format_relevant_paragraphs(
                paragraphs_for_segmentation(relevance)
            )
        if text.strip():
            rows.append({"id": case_id, "text": text})
    conn.close()
    return rows


# ── Schemas ───────────────────────────────────────────────────────────────────

# Field descriptions. The schema and both profile prompts use these.
RELEVANT_TEXTS_DESC = "List every verbatim text span from the opinion that supports your conclusions for this constraint."
ISSUE_SUMMARY_DESC = "Use 1-2 sentences. Summarize the main legal issue the constraint raises in this case."
FACTS_SUMMARY_DESC = "Use 2-4 sentences. State the facts the court treated as material to this constraint: who the parties are and what happened. Use the opinion's own vocabulary. Report a disputed fact as disputed. Do not include the holding, the disposition, or the interpretive tools. Those belong in application_summary and reasoning_chain."
APPLICATION_SUMMARY_DESC = "Use 4 sentences. Summarize how the court applied or ignored this constraint. Quote the opinion on the legal tools of interpretation invoked. Refer to facts_summary. Do not restate the facts. Start with: `In response to the argument ..., the court ... .`"
REASONING_CHAIN_DESC = "Use 3-5 steps, each a short sentence, tracing how the court's reasoning leads to the assigned truth value. Make each step a distinct inferential move: name the legal rule or standard the court invokes, apply it to the facts in facts_summary without restating them, and state what follows. Connect the steps with explicit logical connectives (e.g. 'First ... Therefore ... Finally ...')."
TRUTH_VALUE_DESC = "'true': the court applies or accepts the constraint and finds it satisfied. 'false': the court applies or accepts the constraint and finds it not satisfied. 'unaddressed': the court presents the constraint but does not decide it."
APPLIES_CONSTRAINT_DESC = "Set false ONLY if the opinion is about a different legal question than the constraint. The constraint's leading case appears merely as background or is analogized to a different question, and the constraint's subject matter is not itself at issue. Set true in every other case, including when the court presents the constraint but does not decide it, resolves the dispute on another ground (precedent, procedural default, jurisdiction, a threshold or scope holding), or engages the constraint's subject matter without framing the constraint as the question it decides. Use truth_value 'unaddressed' for those cases."
NONAPPLICATION_REASON_DESC = "If applies_constraint is false, use one sentence naming the different legal question this opinion is about. Otherwise use an empty string."
OUTPUT_TYPE_DESC = "The overall case outcome, as a succinct disposition term (e.g. 'affirmed', 'reversed', 'remanded')."


def _case_profile_schema() -> dict:
    properties = {
        "id": {
            "type": "string",
            "description": "The case ID, copied verbatim from the prompt.",
        },
        "relevant_texts": {
            "type": "array",
            "items": {"type": "string"},
            "description": RELEVANT_TEXTS_DESC,
        },
        "issue_summary": {
            "type": "string",
            "description": ISSUE_SUMMARY_DESC,
        },
        "facts_summary": {
            "type": "string",
            "description": FACTS_SUMMARY_DESC,
        },
        "application_summary": {
            "type": "string",
            "description": APPLICATION_SUMMARY_DESC,
        },
        "reasoning_chain": {
            "type": "string",
            "description": REASONING_CHAIN_DESC,
        },
        "truth_value": {
            "type": "string",
            "enum": ["true", "false", "unaddressed"],
            "description": TRUTH_VALUE_DESC,
        },
        "applies_constraint": {
            "type": "boolean",
            "description": APPLIES_CONSTRAINT_DESC,
        },
        "nonapplication_reason": {
            "type": "string",
            "description": NONAPPLICATION_REASON_DESC,
        },
        "output_type": {
            "type": "string",
            "description": OUTPUT_TYPE_DESC,
        },
    }

    case_profile_schema = {
        "type": "object",
        "properties": properties,
        "required": [
            "id",
            "relevant_texts",
            "issue_summary",
            "facts_summary",
            "application_summary",
            "reasoning_chain",
            "truth_value",
            "applies_constraint",
            "nonapplication_reason",
            "output_type",
        ],
        "additionalProperties": False,
    }

    return {
        "name": "case_profile",
        "schema": case_profile_schema,
        "strict": True,
    }


# ── System prompts ────────────────────────────────────────────────────────────

_CASE_LEVEL_SYSTEM = "You are a legal research assistant. Your job is to extract the constraint profile from a single court opinion: which doctrinal constraints the court engages, with truth values and supporting text. Analogical citation of a leading case is not engagement of the constraint."

# ── Prompts ───────────────────────────────────────────────────────────────────


def _case_profile_prompt_segmented(
    constraint: str,
    id: str,
    segmented_text: str,
) -> str:
    """Receives pre-labeled sentences from paragraph_pipeline.py.

    Each paragraph block uses the format:
      [Paragraph N]
        [Role | Attribution | Stance] sentence text ...

    This lets the model reason about who said what and focus on CurrentCourt
    reasoning.
    """
    return f"""{utils.FORMALIZATION}

# CONSTRAINT (PHI)
{constraint}

# YOUR TASK
Read the labeled opinion below and extract its constraint profile.

The opinion has been pre-segmented into sentences. Each sentence is annotated with:
  [Role | Attribution | Stance]
where:
  - Role:        Facts | Procedure | Issue | Argument | Holding | Reasoning
  - Attribution: CurrentCourt | Appellant | Appellee | Amicus | LowerCourt
  - Stance:      Disputed | Undisputed

Focus on sentences attributed to **CurrentCourt** with Role **Holding** or **Reasoning** when
assessing truth values. Use sentences from other attributions as supporting context only.

Fill these fields:
  - "relevant_texts": {RELEVANT_TEXTS_DESC} Prefer sentences attributed to **CurrentCourt** with Role **Holding** or **Reasoning**. Use other attributions or roles as supporting context.
  - "issue_summary": {ISSUE_SUMMARY_DESC}
  - "facts_summary": {FACTS_SUMMARY_DESC} Draw from sentences with Role Facts. Use Procedure only for posture that bears on the constraint.
  - "application_summary": {APPLICATION_SUMMARY_DESC}
  - "reasoning_chain": {REASONING_CHAIN_DESC}
  - "truth_value": {TRUTH_VALUE_DESC}
  - "applies_constraint": {APPLIES_CONSTRAINT_DESC}
  - "nonapplication_reason": {NONAPPLICATION_REASON_DESC}
  - "output_type": {OUTPUT_TYPE_DESC}

Case ID: {id}

# LABELED RELEVANT PARAGRAPHS FROM THE FULL COURT OPINION:
{segmented_text}"""


def _format_relevant_paragraphs(paragraphs: list[dict]) -> str:
    """Plain text of relevance-kept paragraphs, in opinion order.

    No sentence labels are added.
    """
    ordered = sorted(paragraphs, key=lambda p: int(p["paragraph_index"]))
    blocks = [
        f"[Paragraph {para['paragraph_index']}]\n{para['text']}"
        for para in ordered
    ]
    return "\n\n".join(blocks)


_PLAIN_SOURCE = {
    "relevant": (
        "# RELEVANT PARAGRAPHS FROM THE MAJORITY OPINION",
        """The text below is the paragraphs a relevance pass kept from the majority opinion. They are not sentence-segmented and carry no role, attribution, or stance labels. Read the prose itself to tell the court's holding and reasoning from the parties' arguments, the facts, and the procedural history.""",
    ),
    "full": (
        "# FULL COURT OPINION",
        """The text below is the full stored opinion. No relevance filter has been applied and the text is not sentence-segmented. Read the prose itself to tell the court's holding and reasoning from the parties' arguments, the facts, and the procedural history.""",
    ),
}


def _case_profile_prompt_plain(
    constraint: str,
    id: str,
    opinion_text: str,
    *,
    source: str,
) -> str:
    """Profile prompt for unlabeled text.

    ``source="relevant"`` is the relevance-kept paragraphs.
    ``source="full"`` is the stored opinion.
    """
    if source not in _PLAIN_SOURCE:
        raise ValueError(
            f"source must be one of {tuple(_PLAIN_SOURCE)}, got {source!r}"
        )
    heading, intro = _PLAIN_SOURCE[source]
    return f"""{utils.FORMALIZATION}

# CONSTRAINT (PHI)
{constraint}

# YOUR TASK
Read the opinion text below and extract its constraint profile.

{intro}

Focus on the court's own holding and reasoning when assessing truth values.
Use the parties' arguments and the procedural history as supporting context only.

Fill these fields:
  - "relevant_texts": {RELEVANT_TEXTS_DESC} Prefer the court's holding and reasoning. Use other passages as supporting context.
  - "issue_summary": {ISSUE_SUMMARY_DESC}
  - "facts_summary": {FACTS_SUMMARY_DESC}
  - "application_summary": {APPLICATION_SUMMARY_DESC}
  - "reasoning_chain": {REASONING_CHAIN_DESC}
  - "truth_value": {TRUTH_VALUE_DESC}
  - "applies_constraint": {APPLIES_CONSTRAINT_DESC}
  - "nonapplication_reason": {NONAPPLICATION_REASON_DESC}
  - "output_type": {OUTPUT_TYPE_DESC}

Case ID: {id}

{heading}:
{opinion_text}"""


# ── Pass 1: case-level profiling ──────────────────────────────────────────────


def _unaddressed_profile(
    case_id: int | str, *, no_majority: bool = False
) -> dict:
    """Deterministic profile when the case cannot move forward. No LLM."""
    if no_majority:
        reason = """No majority opinion was identified for this row (status=no_majority)."""
        return {
            "id": str(case_id),
            "relevant_texts": [],
            "issue_summary": """No majority opinion was identified, so the constraint was not scored from this row.""",
            "facts_summary": """No majority opinion was identified, so no constraint-relevant facts were extracted.""",
            "application_summary": """In response to the argument under this constraint, no majority opinion was available to engage it.""",
            "reasoning_chain": """First, majority isolation found no lead opinion on this row. Therefore there is no CurrentCourt Holding or Reasoning text to score. Finally, the constraint is treated as unaddressed.""",
            "truth_value": "unaddressed",
            "applies_constraint": False,
            "nonapplication_reason": reason,
            "output_type": "unaddressed",
        }
    return {
        "id": str(case_id),
        "relevant_texts": [],
        "issue_summary": (
            "No opinion paragraphs were found relevant to this constraint."
        ),
        "facts_summary": """No opinion paragraphs were found relevant to this constraint, so no constraint-relevant facts were extracted.""",
        "application_summary": """In response to the argument under this constraint, the court did not engage it in any retained paragraph after relevance filtering.""",
        "reasoning_chain": """First, relevance filtering retained no paragraphs for this constraint. Therefore there is no CurrentCourt Holding or Reasoning text to score. Finally, the constraint is treated as unaddressed.""",
        "truth_value": "unaddressed",
        "applies_constraint": False,
        "nonapplication_reason": (
            "No opinion paragraphs were found relevant to this constraint."
        ),
        "output_type": "unaddressed",
    }


def profile_applies_constraint(profile: dict[str, Any]) -> bool:
    """False when the profiler marked the opinion as off-constraint."""
    return profile["applies_constraint"]


def extract_case_profiles(
    constraint: str,
    db_path: str,
    model: str,
    cache_column: str | None = None,
    ids: Optional[List[int]] = None,
    segmentation_cache_column: str = "paragraph_sentence_segmentation_cache_json",
    relevance_cache_column: str = "paragraph_relevance_cache_json",
    input_mode: str = "segmented",
) -> List[dict]:
    """Pass 1: call the model to extract each case's constraint profile.

    Returns a list of dicts: {id, profile}.
    Profiles are cached per-row in ``cache_column``, which defaults to
    ``profile_cache_column(model, input_mode)`` so that each model and each
    input representation has its own cache. When ``ids`` is set, only those
    rows are processed.

    Cases that cannot move forward (no majority opinion, no relevant
    paragraphs, or an empty segmentation cache) get a deterministic
    ``unaddressed`` profile without an LLM call. Cases with no segmentation
    cache at all are skipped with a warning. Profiles with
    ``applies_constraint`` false, and cases whose majority isolation status is
    ``no_majority``, are still returned; ``run_pipeline`` excludes them from
    dendrogram induction. All cache-missing cases go out in one batch.

    ``input_mode``: ``segmented`` reads labeled relevant paragraphs;
    ``relevant`` reads those paragraphs as plain text and does not consult the
    segmentation cache; ``full`` reads ``opinion_text_clean`` and does not
    consult relevance, segmentation, or majority isolation.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
        )
    if cache_column is None:
        cache_column = profile_cache_column(model, input_mode)

    cases = utils.load_cases(db_path, ids=ids)
    profiles: dict[Any, dict] = {}
    pending: List[dict] = []
    skipped_unsegmented: List[int] = []
    skipped_blank_segmentation: List[int] = []
    skipped_no_majority: List[int] = []
    skipped_empty_relevance: List[int] = []
    skipped_missing_relevance: List[int] = []
    skipped_empty_opinion: List[int] = []

    conn = sqlite3.connect(db_path)
    for case in cases:
        id = case["id"]

        if input_mode != "full":
            majority_info = utils.cache_read(conn, "majority_boundary_json", id)
            if (
                majority_info is not None
                and majority_info["status"] == majority.STATUS_NO_MAJORITY
            ):
                profiles[id] = _unaddressed_profile(id, no_majority=True)
                utils.cache_write(conn, cache_column, id, profiles[id])
                skipped_no_majority.append(id)
                continue

            relevance = utils.cache_read(conn, relevance_cache_column, id)
            if input_mode == "relevant" and relevance is None:
                skipped_missing_relevance.append(id)
                continue
            if relevance is not None and not paragraphs_for_segmentation(
                relevance
            ):
                profiles[id] = _unaddressed_profile(id)
                utils.cache_write(conn, cache_column, id, profiles[id])
                skipped_empty_relevance.append(id)
                continue

        cached = utils.cache_read(conn, cache_column, id)
        if cached is not None:
            profiles[id] = cached
            continue

        if input_mode == "segmented":
            segmented = utils.load_segmented_sentences(
                conn, id, segmentation_cache_column
            )
            if segmented is None:
                skipped_unsegmented.append(id)
                continue
            if not segmented:
                profiles[id] = _unaddressed_profile(id)
                utils.cache_write(conn, cache_column, id, profiles[id])
                continue
            source_text = utils.format_segmented_paragraphs(segmented)
            if not source_text.strip():
                skipped_blank_segmentation.append(id)
                continue
            prompt = _case_profile_prompt_segmented(constraint, id, source_text)
        elif input_mode == "relevant":
            source_text = _format_relevant_paragraphs(
                paragraphs_for_segmentation(relevance)
            )
            prompt = _case_profile_prompt_plain(
                constraint, id, source_text, source="relevant"
            )
        else:
            source_text = (case["opinion_text"] or "").strip()
            if not source_text:
                skipped_empty_opinion.append(id)
                continue
            prompt = _case_profile_prompt_plain(
                constraint, id, source_text, source="full"
            )
        pending.append({"id": id, "prompt": prompt})

    if pending:
        responses = api_calls.batch_query_llm(
            [
                {"content": row["prompt"], "system_prompt": _CASE_LEVEL_SYSTEM}
                for row in pending
            ],
            model=model,
            response_format=_case_profile_schema(),
        )
        for row, (_, profile) in zip(pending, responses):
            profiles[row["id"]] = profile
            utils.cache_write(conn, cache_column, row["id"], profile)
    conn.close()

    for label, skipped in (
        ("with no majority opinion (no LLM)", skipped_no_majority),
        ("with no relevant paragraphs (no LLM)", skipped_empty_relevance),
        (
            "with no segmentation cache (run prepare_segmentations first)",
            skipped_unsegmented,
        ),
        (
            "whose segmentation had no complete sentence",
            skipped_blank_segmentation,
        ),
        ("with no relevance cache", skipped_missing_relevance),
        ("with no stored opinion text", skipped_empty_opinion),
    ):
        if skipped:
            print(
                f"""[case_profiles] skipped {len(skipped)} case(s) {label}: {skipped[:10]}{'...' if len(skipped) > 10 else ''}"""
            )

    return [
        {"id": case["id"], "profile": profiles[case["id"]]}
        for case in cases
        if case["id"] in profiles
    ]
