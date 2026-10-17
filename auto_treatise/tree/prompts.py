"""System prompts, response schemas and prompt builders for stages 1 and 2.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from .. import utils
from .config import ASSIGNMENT_BATCH_SIZE
from .config import AXIS_SHUFFLE_SEED
from .config import AXIS_TYPES
from .config import MAX_USED_LABELS_SHOWN
from .config import RAW_ASSIGNMENT_CHUNK_CHARS
from .config import RAW_ASSIGNMENT_TEXT_CHARS
from .config import RAW_STAGE1_CHAR_BUDGET

FORMALIZATION_CORPUS_LEVEL = f"""{utils.FORMALIZATION}
## CORPUS ANALYSIS
Given some constraint $phi_i$ and a corpus of cases that purportedly apply it, your job is to assess the differences in how the constraint has been applied and to identify the most signficant divergence.

## OUTPUT
Identify TWO clusters of cases that represent the most significant divergences in how the constraint has been applied. Describe the differences in how the constraint has been applied in each cluster.
"""


# ── System prompts ────────────────────────────────────────────────────────────

_AXIS_SYSTEM = "You are a legal research assistant. Given summaries of the issues, material facts, and reasoning of a set of courts about a single constraint, name the doctrinal axis along which they most significantly diverge, and state a test that another reader could apply to sort any one of those cases without seeing the others."


_AXIS_SYSTEM_RAW = "You are a legal research assistant. Given opinion text from a set of courts about a single constraint, with no extracted constraint profile, name the doctrinal axis along which they most significantly diverge, and state a test that another reader could apply to sort any one of those cases without seeing the others."


_ASSIGNMENT_SYSTEM = """You are a legal research assistant. Apply a fixed doctrinal decision rule to each court opinion profile you are given, one at a time, and record which side of the distinction it falls on and how confident that reading is. Do not revise the rule and do not balance the two sides. If both sides' signals are present in the same opinion, classify by which signal is the engine of disposition: the construction that, if removed, would change the result. Recited boilerplate does not override the analysis that actually decides the case, and vice versa. When the signals are mixed, still pick a side by that engine rule and mark confidence as low."""


_COMPLEMENT_JUDGE_SYSTEM = """You are a legal research assistant auditing a doctrinal classification tree. You see the whole tree built so far and one proposed way of splitting a cluster. Decide whether that split would leave cases in the cluster that another cluster describes better, which would mean they were routed there by mistake. Judge by where the cases belong, not by whether the wording resembles another label. Most splits are legitimate finer cuts among cases that all belong where they are."""


# ── Field descriptions ────────────────────────────────────────────────────────
# The schemas and the prompt builders both use these, so the wording stays the
# same in both places.

AXIS_TYPE_REPEAT = """An axis_type may repeat one already used only if the split draws a strictly finer distinction that would not sort the same cases the same way."""
AXIS_REPEAT_LEAF = """If the most significant remaining divergence is an axis already used, set is_leaf=true."""
CHILD_LABEL_RULES = """Name only what distinguishes this child from its sibling, not what it inherits from its ancestors. Do not reuse distinctive content words from the current cluster label or any path 'took' label. Do not repeat or paraphrase the current cluster label, any ancestor label, any path 'not' label, or any label already in use elsewhere in the tree."""

SPLIT_CRITERION_DESC = """Explain in 3-5 steps how the case summaries support this node's status (leaf or split) and, if split, the doctrinal axis that divides them. Make each step a distinct inferential move: identify the doctrinal signal used for grouping (issue framing, material facts, reasoning structure, truth value, disposition), explain how it applies, and state the conclusion. Connect the steps with explicit logical connectives (e.g. 'First ... Therefore ... Finally ...', 'Because ... Therefore ... In Contrast ...'). Include one step stating how this axis differs from every axis already used on this path. Name each path 'not' label and state why the new axis is not that rejected sibling under a new name. If is_leaf=true, justify doctrinal homogeneity and explain why no reliable discriminating axis remains."""
AXIS_TYPE_DESC = (
    """Choose the category that best fits the doctrinal axis of split_criterion.
- interpretive_method: which tools the court uses to read the governing source: plain text, structure, canons, history, precedent, purpose.
- authority_allocation: who gets to decide: deference posture, institutional role, allocation between court and agency or between court levels.
- constraint_predicate: WHAT MUST BE SHOWN to satisfy phi: the elements, preconditions, triggers, or justifications a party must establish.
- constraint_scope: WHAT PHI REACHES: which persons, items, conduct, sources, or legal contexts fall inside or outside the constraint's coverage at all.
- evidentiary_threshold: HOW MUCH or what quality of showing phi demands once the predicate is fixed: burden, sufficiency, quantum, representativeness of proof.
- procedural_posture: standard of review, stage of litigation, preservation, waiver, jurisdiction, reviewability.
- engagement_depth: how substantively the court engages phi at all, as opposed to reciting it and deciding on another ground.
- other: none of the above fits.
Do not conflate constraint_predicate and constraint_scope. Scope asks whether phi applies to this case's subject at all. Predicate asks what must be established once it does. Choose constraint_scope if the split turns on coverage. Choose constraint_predicate if it turns on what the proponent must prove.
"""
    + AXIS_TYPE_REPEAT
    + """ An axis_type that already appears twice on this path signals a homogeneous cluster. Set is_leaf=true."""
)
SPLIT_SHORT_DESC = """Compress the split axis to at most 12 words, phrased as the question the split answers, in terms a lawyer understands (e.g. 'whether the court consults legislative history or stops at plain text'). Be specific to THIS split. Do not restate any axis listed under AXES ALREADY USED ON THIS PATH."""
DECISION_RULE_DESC = """State one test, in 1-2 sentences, that sorts ANY case in this cluster to left or right, in terms a lawyer understands. Make it applicable to a case read in isolation: state the observable feature of the court's reasoning to look for, and say which side its presence indicates. Make it apply when an opinion recites both sides' language: the grader classifies by the engine of disposition, not by boilerplate. Do not name or count specific cases. A grader given only this rule and one case profile must reach the same answer you would."""
IS_LEAF_DESC = """Set true when the cluster is doctrinally homogeneous and cannot be split further. Also set true when the only distinction you can find would restate an axis already used higher on this path, would recreate a path 'not' label, or would reproduce a label already in use elsewhere in the tree. Do not manufacture a split to avoid one."""


def _child_label_desc(side: str) -> str:
    return (
        f"Use a 3-8 word label for the side the decision_rule sorts {side}. "
        "Use legal terminology a lawyer understands. " + CHILD_LABEL_RULES
    )


def _child_description_desc(side: str) -> str:
    return (
        "Use 4 sentences. Explain the reading of the constraint shared by "
        f"cases on the {side} side of the axis, and the legal tools those "
        "courts invoke. Start the first sentence with: "
        "Phi is interpreted to mean `...`."
    )


# ── Schemas ───────────────────────────────────────────────────────────────────


def _axis_proposal_schema() -> dict[str, Any]:
    """Stage 1: Choose the doctrinal axis."""
    return {
        "name": "axis_proposal",
        "schema": {
            "type": "object",
            "properties": {
                "split_criterion": {
                    "type": "string",
                    "description": SPLIT_CRITERION_DESC,
                },
                "axis_type": {
                    "type": "string",
                    "enum": AXIS_TYPES,
                    "description": AXIS_TYPE_DESC,
                },
                "split_criterion_short": {
                    "type": "string",
                    "description": SPLIT_SHORT_DESC,
                },
                "decision_rule": {
                    "type": "string",
                    "description": DECISION_RULE_DESC,
                },
                "is_leaf": {
                    "type": "boolean",
                    "description": IS_LEAF_DESC,
                },
                "left_label": {
                    "type": "string",
                    "description": _child_label_desc("LEFT"),
                },
                "left_description": {
                    "type": "string",
                    "description": _child_description_desc("left"),
                },
                "right_label": {
                    "type": "string",
                    "description": _child_label_desc("RIGHT"),
                },
                "right_description": {
                    "type": "string",
                    "description": _child_description_desc("right"),
                },
            },
            "required": [
                "split_criterion",
                "axis_type",
                "split_criterion_short",
                "decision_rule",
                "is_leaf",
                "left_label",
                "left_description",
                "right_label",
                "right_description",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _assignment_schema() -> dict[str, Any]:
    """Stage 2: Sort one chunk of full case profiles against a fixed decision
    rule."""
    return {
        "name": "case_assignments",
        "schema": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "description": "Return exactly one entry per id shown, in the order shown.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The id being assigned, copied verbatim from the input.",
                            },
                            "rationale": {
                                "type": "string",
                                "description": """Use one sentence. Name the feature of THIS court's reasoning that the decision rule keys on. Do not restate the rule.""",
                            },
                            "side": {
                                "type": "string",
                                "enum": ["left", "right"],
                                "description": "The side the decision rule assigns this case to.",
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "low"],
                                "description": """Rate how cleanly the decision rule applies. Use 'low' when the case sits on the boundary or the rule's key feature is absent from the profile.""",
                            },
                            "confidence_explanation": {
                                "type": "string",
                                "description": """Use one sentence explaining the confidence. Do not restate the rule.""",
                            },
                        },
                        "required": [
                            "id",
                            "rationale",
                            "side",
                            "confidence",
                            "confidence_explanation",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["assignments"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _complement_judge_schema() -> dict[str, Any]:
    """Would this split leave cases here that belong on an existing branch?"""
    return {
        "name": "complement_echo_judge",
        "schema": {
            "type": "object",
            "properties": {
                "misplacement_analysis": {
                    "type": "string",
                    "description": """Use 2-3 sentences. Say whether either proposed child would hold cases that a different existing cluster describes better. If so, name the child, the kind of case, and the cluster. If the split draws a finer line among cases that all belong here, say that.""",
                },
                "better_fit_node": {
                    "type": ["string", "null"],
                    "description": """Copy exactly the node id from tree_so_far that describes those cases better. Use null when every case belongs in the current cluster, including when the split reuses another cluster's vocabulary.""",
                },
            },
            "required": ["misplacement_analysis", "better_fit_node"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _axis_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Stage-1 view of a case: issue, facts, and reasoning structure; no
    identifier."""
    return {
        "issue_summary": profile["issue_summary"],
        "facts_summary": profile["facts_summary"],
        "reasoning_chain": profile["reasoning_chain"],
        "truth_value": profile["truth_value"],
        "output_type": profile["output_type"],
    }


def _chunked(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def _assignment_chunks(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Profiled rows stay in fixed batches. Raw text is packed by characters."""
    if not _rows_are_raw(rows):
        return _chunked(rows, ASSIGNMENT_BATCH_SIZE)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for row in rows:
        n = min(len(row["raw_text"]), RAW_ASSIGNMENT_TEXT_CHARS)
        if current and size + n > RAW_ASSIGNMENT_CHUNK_CHARS:
            chunks.append(current)
            current = []
            size = 0
        current.append(row)
        size += n
    if current:
        chunks.append(current)
    return chunks


def _truncate_chars(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    if cap <= 0:
        return ""
    return text[: cap - 1].rstrip() + "…"


def _rows_are_raw(cases: list[dict[str, Any]]) -> bool:
    """True when every row is opinion text rather than a constraint profile."""
    return bool(cases) and all("raw_text" in row for row in cases)


def _raw_stage1_cap(n_cases: int) -> int:
    """Per-case character cap for a no-profile stage-1 prompt."""
    if n_cases <= 0:
        return 0
    return max(RAW_STAGE1_CHAR_BUDGET // n_cases, 1)


# ── Prompts ───────────────────────────────────────────────────────────────────


def _build_axis_prompt(
    constraint: str,
    constraint_id: str,
    node_id: str,
    node_label: str,
    inherited_description: str,
    cases_with_profiles: list[dict[str, Any]],
    depth: int,
    lineage_steps: list[dict[str, str]],
    used_labels: list[str],
    created_axes: list[dict[str, str]] | None = None,
    failed_axes: list[dict[str, Any]] | None = None,
) -> str:
    """Stage 1: Propose the axis from unlabeled, shuffled case views.

    Profiled rows show issue, facts, and reasoning summaries. 
    Rows with ``raw_text`` show that opinion text, capped so the cluster fits a
    context window.
    """
    lineage_text = _format_tree_lineage(lineage_steps)
    used_labels_text = _format_used_labels(used_labels)
    created_axes_text = _format_created_axes(created_axes or [])
    forbidden_text = _format_forbidden_label_terms(node_label, lineage_steps)
    failed_axes = failed_axes or []
    rejected_text = _format_failed_axes(failed_axes)

    raw = _rows_are_raw(cases_with_profiles)
    if raw:
        cap = _raw_stage1_cap(len(cases_with_profiles))
        summaries = [
            {"text": _truncate_chars(row["raw_text"], cap)}
            for row in cases_with_profiles
        ]
        case_intro = f"""Below are the {len(summaries)} cases in this cluster, in arbitrary order. Each entry is the opinion text for that case. There is no constraint profile."""
        prefer = """Prefer axes that separate courts that reason differently in the text. Do not split merely on different outcomes, or on subject matter that is not material to the constraint. A difference in the facts the courts treated as the input to phi counts only when it changes what phi requires or reaches — classify such a split as constraint_predicate or constraint_scope, whichever it turns on."""
    else:
        summaries = [
            _axis_profile(row["profile"]) for row in cases_with_profiles
        ]
        case_intro = f"""Below are the {len(summaries)} cases in this cluster, in arbitrary order, summarized by the issue each court faced, the facts it treated as material, and how it reasoned."""
        prefer = """Prefer axes that separate courts that reason differently (pay attention to reasoning_chain). Do not split merely on different outcomes, or on subject matter that is not material to the constraint. A difference in the facts the courts treated as the input to phi counts only when it changes what phi requires or reaches — classify such a split as constraint_predicate or constraint_scope, whichever it turns on."""

    rng = random.Random(f"{AXIS_SHUFFLE_SEED}:{node_id}:{len(failed_axes)}")
    rng.shuffle(summaries)
    summaries_json = json.dumps(summaries, indent=2, ensure_ascii=False)

    return f"""{FORMALIZATION_CORPUS_LEVEL}

# CONSTRAINT (PHI)
{constraint}

# AXES ALREADY USED ON THIS PATH
{lineage_text}

# AXES ALREADY USED ELSEWHERE IN THE TREE
{created_axes_text}

# LABELS ALREADY IN USE ELSEWHERE IN THE TREE
{used_labels_text}

# TERMS NOT TO REUSE IN CHILD LABELS
{forbidden_text}
{rejected_text}
# CURRENT TASK: CHOOSE THE AXIS
Constraint: {constraint_id}
Depth: {depth}
Current cluster label: {node_label}
This cluster's working interpretation: {inherited_description}

{case_intro}

{summaries_json}

Instructions:
1) Every case above already satisfies ALL of the distinctions listed under AXES ALREADY USED ON THIS PATH. Do not re-derive them.
2) Decide whether this cluster is homogeneous under the constraint GIVEN those settled distinctions.
3) If not homogeneous, identify a NEW axis that divides it. {prefer}
4) Do not restate any axis listed under AXES ALREADY USED ELSEWHERE IN THE TREE. {AXIS_TYPE_REPEAT} {AXIS_REPEAT_LEAF}
5) {CHILD_LABEL_RULES}
"""


def _build_assignment_prompt(
    constraint: str,
    node_label: str,
    decision_rule: str,
    split_criterion_short: str,
    left_label: str,
    left_description: str,
    right_label: str,
    right_description: str,
    chunk: list[dict[str, Any]],
) -> str:
    """Stage 2: sort one chunk of cases against a fixed, already-chosen rule."""
    raw = _rows_are_raw(chunk)
    if raw:
        case_payload = [
            {
                "id": row["id"],
                "text": _truncate_chars(
                    row["raw_text"], RAW_ASSIGNMENT_TEXT_CHARS
                ),
            }
            for row in chunk
        ]
        judge_on = "text. There is no constraint profile"
    else:
        case_payload = [
            {
                "id": row["id"],
                "profile": {
                    "issue_summary": row["profile"]["issue_summary"],
                    "facts_summary": row["profile"]["facts_summary"],
                    "application_summary": row["profile"][
                        "application_summary"
                    ],
                    "relevant_texts": row["profile"]["relevant_texts"],
                    "reasoning_chain": row["profile"]["reasoning_chain"],
                    "truth_value": row["profile"]["truth_value"],
                    "output_type": row["profile"]["output_type"],
                },
            }
            for row in chunk
        ]
        judge_on = "profile"
    case_payload_json = json.dumps(case_payload, indent=2, ensure_ascii=False)
    ids = [row["id"] for row in chunk]
    return f"""{utils.FORMALIZATION}

# CONSTRAINT (PHI)
{constraint}

# THE DISTINCTION TO APPLY
These cases all belong to the cluster "{node_label}". Now, the goal is to divide them based on the distinction of {split_criterion_short}, as described in detail below.

DECISION RULE (apply exactly as written):
{decision_rule}

LEFT -- {left_label}
{left_description}

RIGHT -- {right_label}
{right_description}

# YOUR TASK
Sort each case below to LEFT or RIGHT by applying the decision rule to that case alone. Judge each case on its own {judge_on}; do not balance the two sides, and do not let the number of cases already sorted one way influence the next.
If both sides' signals appear in the same opinion, classify by the engine of disposition: the construction that, if removed, would change the result. Recited boilerplate does not override the analysis that actually decides the case, and vice versa. Mixed signals still get a side; mark confidence low.

Return exactly one assignment per id, in the order given, for all {len(chunk)} cases:
{json.dumps(ids)}

Cases:
{case_payload_json}
"""


def _build_assignment_retry_prompt(
    base_prompt: str, missing_ids: list[str]
) -> str:
    return f"""{base_prompt}\n\n
        RETRY: Your previous response omitted the following ids. Apply the same decision rule and return an assignment for each of them. Do not change your earlier assignments.

        Missing ids: {json.dumps(missing_ids)}
    """


def _format_tree_lineage(lineage_steps: list[dict[str, str]]) -> str:
    """Compact one-line-per-ancestor rendering of the path to this node."""
    if not lineage_steps:
        return "This is the corpus root. No axes have been used yet, so any doctrinal axis is available."

    lines = [
        f"The current cluster is the conjunction of the {len(lineage_steps)} distinction(s) below; all of them already hold for every case in it. Each line gives the axis that split the cluster, the side this branch took, and the side it did not.",
    ]
    for idx, step in enumerate(lineage_steps, start=1):
        axis_type = step["axis_type"].strip() or "unspecified"
        axis_short = step["split_criterion_short"].strip()
        lines.append(f"{idx}. [{axis_type}] {axis_short}")
        lines.append(
            f"   -> took: {step['selected_label']}  |  not: {step['alternate_label']}"
        )
    lines.append("")
    lines.append(
        "Your split_criterion must not restate any axis above. If it would, set is_leaf=true."
    )
    lines.append(
        """If the only remaining split is that some cases look like a 'not' label above, set is_leaf=true. Do not propose a child that recreates a rejected sibling."""
    )
    return "\n".join(lines)


def _format_used_labels(used_labels: list[str]) -> str:
    """Labels already created anywhere in the tree, so cousins don't repeat."""
    if not used_labels:
        return "None yet."
    shown = used_labels[-MAX_USED_LABELS_SHOWN:]
    body = "\n".join(f"- {label}" for label in shown)
    return f"These cluster labels already exist in other branches of this tree. Do not produce a child label that duplicates or paraphrases any of them; a label should identify exactly one region of the tree.\n{body}"


def _format_created_axes(created_axes: list[dict[str, str]]) -> str:
    """Cousin and ancestor axes already proposed or accepted anywhere in the
    tree."""
    if not created_axes:
        return "None yet. Any doctrinal axis is available."
    lines = [
        "Do not restate these questions. " + AXIS_TYPE_REPEAT + " " + AXIS_REPEAT_LEAF,
    ]
    for axis in created_axes:
        short = axis["split_criterion_short"] or "(unnamed)"
        left = axis["left_label"]
        right = axis["right_label"]
        prefix = f"{axis['node_id']}: "
        lines.append(f"- {prefix}{short}")
        if left or right:
            lines.append(f"    left: {left}  |  right: {right}")
    return "\n".join(lines)


# ── Label tokens (shared with the echo check) ─────────────────────────────────

_LABEL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "over",
        "the",
        "to",
        "under",
        "versus",
        "vs",
        "with",
        "without",
    }
)


def _label_tokens(label: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", label.lower())
    return {w for w in words if w not in _LABEL_STOPWORDS and len(w) > 2}


def _distinctive_terms_from_path(
    node_label: str, lineage_steps: list[dict[str, str]]
) -> set[str]:
    terms = set(_label_tokens(node_label))
    for step in lineage_steps:
        terms.update(_label_tokens(step["selected_label"]))
    return terms


def _format_forbidden_label_terms(
    node_label: str, lineage_steps: list[dict[str, str]]
) -> str:
    terms = _distinctive_terms_from_path(node_label, lineage_steps)
    if not terms:
        return "None yet."
    listed = ", ".join(sorted(terms))
    return f"""Do not reuse these distinctive words from the current cluster label or any path 'took' label: {listed}."""


def _format_failed_axes(failed_axes: list[dict[str, Any]]) -> str:
    """Retry block listing axes that already produced a lopsided or complement
    split."""
    if not failed_axes:
        return "\n"
    lines = [
        "# REJECTED AXES FOR THIS CLUSTER",
        """A previous proposal was discarded. Propose a DIFFERENT axis that does not recreate a rejected sibling on this path and that carves the cluster more evenly, or set is_leaf=true if no such axis remains. If some cases look like they belong on a path 'not' label, do not split them off here.""",
    ]
    for idx, axis in enumerate(failed_axes, start=1):
        short = axis["split_criterion_short"] or "(unnamed)"
        reason = axis["reason"]
        if reason == "path_complement":
            echoed = axis["echoed_label"]
            lines.append(
                f"{idx}. [{reason}] {short}  restates not-label {echoed!r}"
            )
        else:
            lines.append(
                f"{idx}. [{reason}] {short}  ->  {axis['left_n']}/{axis['right_n']}"
            )
    return "\n" + "\n".join(lines) + "\n"


def _failed_axis_record(
    proposal: dict[str, Any],
    left_n: int,
    right_n: int,
    reason: str = "degenerate",
    echoed_label: str = "",
) -> dict[str, Any]:
    record = {
        "split_criterion_short": proposal["split_criterion_short"],
        "left_n": left_n,
        "right_n": right_n,
        "reason": reason,
    }
    if echoed_label:
        record["echoed_label"] = echoed_label
    if reason == "path_complement":
        record["proposal"] = {
            key: proposal[key]
            for key in (
                "axis_type",
                "decision_rule",
                "left_label",
                "left_description",
                "right_label",
                "right_description",
            )
        }
    return record


def _failed_axis_notes(failed_axes: list[dict[str, Any]]) -> list[str]:
    notes = []
    for idx, axis in enumerate(failed_axes, start=1):
        reason = axis["reason"]
        short = axis["split_criterion_short"] or "(unnamed)"
        if reason == "path_complement":
            echoed = axis["echoed_label"]
            notes.append(
                f"Rejected axis {idx}: {short} restates path not-label {echoed!r}."
            )
        else:
            notes.append(
                f"""Rejected axis {idx}: {short} split {axis['left_n']}/{axis['right_n']}."""
            )
    return notes
