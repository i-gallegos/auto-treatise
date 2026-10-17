"""Relabeling pass over a finished tree.

Reads ``corpus_level.json`` and writes ``labels_gpt-5.6-sol.json`` beside it. For
each sibling pair, top-down, GPT-5.6-Sol rewrites the ``label``, ``summary``,
``description`` and ``split_criterion_short`` so they use legal terms of art.
It may call tools to read more of the assigned opinions and to search the web.
Case assignments, ``decision_rule`` and tree structure are never written.

    python -m auto_treatise.labels --doctrine <d> --run-id <run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Callable

import api_calls

from . import utils
from .tree import config as tree_config
from .tree import echo as tree_echo
from .tree import expand as tree_expand

ROOT_ID = tree_config.ROOT_ID
LABEL_MODEL = "gpt-5.6-sol"
OVERLAY_NAME = "labels_gpt-5.6-sol.json"
TERMINAL_TOOL = "submit_revision"

# Cases shown per side, and the seed that picks them.
SAMPLE_SIZE = 50
SAMPLE_SEED = 42

# Size limits on what the prompt and the tools return.
TOOL_RESULT_CHAR_CAP = 12_000
APPLICATION_SUMMARY_CHARS = 800
RELEVANT_TEXTS_PER_CASE = 2
RELEVANT_TEXT_CHARS = 400

# Field descriptions. The schema and the revision prompt both use these, so
# the wording stays the same in both places.
REGISTER_DESC = """Use the correct legal term of art, in an analytical expert register. A lawyer must recognize it immediately. Do not use colorful, literary, or lay language. Do not invent cluster jargon."""
SHORT_NAME_DESC = (
    """Compress the frozen cut to at most 12 words, phrased as the question the decision rule answers. Name this same cut, not a new one. """
    + REGISTER_DESC
)
LABEL_DESC = (
    """Use a short heading of 3-5 words. The summary carries the holding; do not pack it into the title. """
    + REGISTER_DESC
)
SUMMARY_DESC = """Use 1-2 sentences stating what this cluster means. Write it as the topic sentence of a legal brief describing this distinction for litigation: a lawyer and a judge would accept it. Use the same legal register as the label. This field carries the holding. Do not compress it into shorthand or hyphenated neologisms."""
PHI_DESC = """Write 4 sentences, as a hornbook explanation. Start the first sentence with: Phi is interpreted to mean `...`. Make the gloss legally precise, not a paraphrase for the public. Use an analytical register with no colorful language."""


_LABEL_SYSTEM = """You are an expert legal analyst, academic, and litigator. You rename doctrinal clusters so a legal expert can read them as black-letter law headings. Use correct terms of art as lawyers and courts use them. Do not translate them into plain English. Write in an analytical register: name the legal operation (who decides, what must be shown, which source controls). Avoid any metaphor, color, narrative flourish, and invented cluster jargon. The audience is a legal expert, not a lay person. Do not change which cases belong on which side of the frozen decision rule."""


REVISION_USER_TEMPLATE = """{formalization}

# CONSTRAINT (PHI)
{constraint}

# FROZEN CUT (do not revise)
Axis type: {axis_type}
Decision rule:
{decision_rule}

# CURRENT SHORT NAME (revise)
{split_criterion_short}

# NODE DESCRIPTIONS
Prior headings are withheld. Name each node from these descriptions and
the assigned opinions; do not reuse or lightly edit a previous title.

LEFT  [{left_id}]
{left_description}

RIGHT [{right_id}]
{right_description}

# PATH 'NOT' LABELS (rejected siblings; do not recreate)
{not_labels}

# LABELS ALREADY IN USE ELSEWHERE
{used_labels}

# ASSIGNED CASES
Up to {sample_size} high-confidence cases per side, drawn from the completed
assignment. These are the opinions the labels must name.

## LEFT ({n_left} shown)
{left_cases}

## RIGHT ({n_right} shown)
{right_cases}

# REGISTER
The audience is a legal expert, not a lay person. """ + REGISTER_DESC + """
Prefer established doctrine names over hyphenated neologisms.

# TASK
Write a new short split name and new LEFT and RIGHT labels, summaries, and
descriptions. Name this frozen cut in the terms of art these opinions
actually use. Always submit a revision.
Do not duplicate or paraphrase any label listed as already in use.

# FIELDS
split_criterion_short: """ + SHORT_NAME_DESC + """
left_label, right_label: """ + LABEL_DESC + """
left_summary, right_summary: """ + SUMMARY_DESC + """
left_description, right_description: """ + PHI_DESC + """
"""


TOOLS_ADDENDUM = """
# TOOLS
You may call tools before submitting. Use node_cases to fetch more assigned
profiles; case_quotes for one opinion's quotations; search_quotes to grep
assigned relevant_texts for a candidate term of art. Use web_search to look
up legal usage: doctrine names, treatise/hornbook headings, how courts
phrase this distinction, or whether a candidate label is a recognized term
of art. If the opinions' operative doctrine name and a web result diverge,
prefer the name a lawyer would cite from these opinions. Do not let a web
result rewrite the frozen decision rule, and do not substitute a lay gloss
for a term of art. When finished, call submit_revision with the overlay
fields.
"""


def load_overlay(run_dir: Path) -> dict[str, Any] | None:
    """The run's overlay, or None when it has not been written."""
    path = Path(run_dir) / OVERLAY_NAME
    return json.loads(path.read_text()) if path.exists() else None


def apply_overlay(
    corpus: dict[str, Any], overlay: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a shallow-copied corpus with overlay wording applied.

    Does not mutate ``corpus``. Never patches ``n0`` ``label``,
    ``summary``, or ``description``. Parent ``left_description`` /
    ``right_description`` are synced from children. ``split_criterion_short``
    is applied on the node that owns the cut, including ``n0``.
    """
    if overlay is None:
        return corpus
    nodes = [{**node} for node in corpus["nodes"]]
    by_id = {node["id"]: node for node in nodes}
    for nid, patch in overlay["nodes"].items():
        node = by_id[nid]
        if nid != ROOT_ID:
            if "label" in patch:
                node["label"] = patch["label"]
            if "summary" in patch:
                node["summary"] = patch["summary"]
            if "description" in patch:
                node["description"] = patch["description"]
                parent_id = node["parent_id"]
                if parent_id:
                    parent = by_id[parent_id]
                    if node["id"] == parent["left_id"]:
                        parent["left_description"] = patch["description"]
                    elif node["id"] == parent["right_id"]:
                        parent["right_description"] = patch["description"]
        if "split_criterion_short" in patch:
            node["split_criterion_short"] = patch["split_criterion_short"]
    return {**corpus, "nodes": nodes}


# ── Revision tools ──────────────────────────────────────────────────────────


def _revision_properties() -> dict[str, Any]:
    """JSON-schema properties shared by the revision schema and submit tool."""
    return {
        "split_criterion_short": {
            "type": "string",
            "description": SHORT_NAME_DESC,
        },
        "left_label": {
            "type": "string",
            "description": "LEFT child label. " + LABEL_DESC,
        },
        "left_summary": {
            "type": "string",
            "description": "LEFT child summary. " + SUMMARY_DESC,
        },
        "left_description": {
            "type": "string",
            "description": "LEFT child description. " + PHI_DESC,
        },
        "right_label": {
            "type": "string",
            "description": "RIGHT child label. " + LABEL_DESC,
        },
        "right_summary": {
            "type": "string",
            "description": "RIGHT child summary. " + SUMMARY_DESC,
        },
        "right_description": {
            "type": "string",
            "description": "RIGHT child description. " + PHI_DESC,
        },
        "rationale": {
            "type": "string",
            "description": """Use 1-2 sentences. Name the terms of art chosen and why they fit the assigned cases. Use an analytical register.""",
        },
    }


def client_tools() -> list[dict[str, Any]]:
    """Function tools for the revision loop. Web search is the provider's
    hosted tool, registered by ``web_search=True``."""
    submit_properties = _revision_properties()
    return [
        {
            "name": "node_cases",
            "description": """Fetch additional assigned case profiles from one child node beyond those already shown in the prompt.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Child node id (the LEFT or RIGHT id).",
                    },
                    "n": {
                        "type": "integer",
                        "description": "How many additional cases to return (default 20).",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "low", "any"],
                        "description": "Filter by assignment confidence (default high).",
                    },
                },
                "required": ["node_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "case_quotes",
            "description": """Return one assigned opinion's application_summary and relevant_texts (verbatim quotations).""",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {
                        "type": "string",
                        "description": "Case id as shown in the prompt.",
                    },
                },
                "required": ["case_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_quotes",
            "description": """Grep assigned relevant_texts in a child node for a candidate term of art.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Child node id to search.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive substring to find.",
                    },
                },
                "required": ["node_id", "query"],
                "additionalProperties": False,
            },
        },
        {
            "name": TERMINAL_TOOL,
            "description": "Submit the revised short split name and LEFT/RIGHT labels, summaries, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": submit_properties,
                "required": list(submit_properties),
                "additionalProperties": False,
            },
        },
    ]


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _compact_case(
    case_id: str,
    assignment: dict[str, Any],
    profile: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    quotes = [
        _truncate(str(t), RELEVANT_TEXT_CHARS)
        for t in profile["relevant_texts"][:RELEVANT_TEXTS_PER_CASE]
    ]
    return {
        "id": case_id,
        "case_name": meta["case_name"],
        "confidence": assignment["confidence"],
        "rationale": assignment["rationale"],
        "application_summary": _truncate(
            str(profile["application_summary"]), APPLICATION_SUMMARY_CHARS
        ),
        "relevant_texts": quotes,
    }


def sample_side(
    assignments: dict[str, dict[str, Any]],
    side: str,
    size: int,
    rng: random.Random,
) -> list[str]:
    """Prefer high-confidence ids, fill with low, cap at ``size``."""
    high: list[str] = []
    low: list[str] = []
    for case_id, entry in assignments.items():
        if entry["side"] != side:
            continue
        if entry["confidence"] == "high":
            high.append(case_id)
        else:
            low.append(case_id)
    rng.shuffle(high)
    rng.shuffle(low)
    return (high + low)[:size]


def _cap(text: str) -> str:
    if len(text) <= TOOL_RESULT_CHAR_CAP:
        return text
    return text[: TOOL_RESULT_CHAR_CAP - 1] + "…"


class LabelTools:
    """Client-tool implementations bound to one tree + profiles."""

    def __init__(
        self,
        by_id: dict[str, dict[str, Any]],
        profiles: dict[str, dict[str, Any]],
        case_meta: dict[str, dict[str, Any]],
        shown_ids: dict[str, set[str]],
        sample_size: int,
    ) -> None:
        self.by_id = by_id
        self.profiles = profiles
        self.case_meta = case_meta
        self.shown_ids = shown_ids
        self.sample_size = sample_size

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch one client tool call by name."""
        if name == "node_cases":
            return self.node_cases(args)
        if name == "case_quotes":
            return self.case_quotes(args)
        if name == "search_quotes":
            return self.search_quotes(args)
        raise KeyError(name)

    def node_cases(self, args: dict[str, Any]) -> str:
        """Up to ``n`` (default 20) unseen cases from one child.

        High-confidence assignments only unless ``confidence`` says otherwise.
        Returned ids are added to ``shown_ids``.
        """
        node = self.by_id[args["node_id"]]
        raw_n = args["n"] if "n" in args else None
        n = 20 if raw_n is None else int(raw_n)
        raw_confidence = args["confidence"] if "confidence" in args else None
        confidence = "high" if raw_confidence is None else raw_confidence
        parent = self.by_id[node["parent_id"]]
        side = "left" if parent["left_id"] == node["id"] else "right"
        assignments = parent["assignments"]
        already = (
            self.shown_ids[node["id"]]
            if node["id"] in self.shown_ids
            else set()
        )
        rng = random.Random(f"{SAMPLE_SEED}:extra:{node['id']}")
        pool = sample_side(assignments, side, len(assignments), rng)
        extra: list[str] = []
        for case_id in pool:
            if case_id in already:
                continue
            if (
                confidence != "any"
                and assignments[case_id]["confidence"] != confidence
            ):
                continue
            extra.append(case_id)
            if len(extra) >= n:
                break
        rows = [
            _compact_case(
                case_id,
                assignments[case_id],
                self.profiles[case_id],
                self.case_meta[case_id],
            )
            for case_id in extra
        ]
        already.update(extra)
        self.shown_ids[node["id"]] = already
        return _cap(
            json.dumps(
                {"node_id": node["id"], "cases": rows}, ensure_ascii=False
            )
        )

    def case_quotes(self, args: dict[str, Any]) -> str:
        """One case's application summary, quotations, and reasoning chain."""
        case_id = str(args["case_id"])
        profile = self.profiles[case_id]
        payload = {
            "id": case_id,
            "application_summary": profile["application_summary"],
            "relevant_texts": profile["relevant_texts"],
            "reasoning_chain": profile["reasoning_chain"],
        }
        return _cap(json.dumps(payload, ensure_ascii=False))

    def search_quotes(self, args: dict[str, Any]) -> str:
        """Case-insensitive substring search over a node's ``relevant_texts``.

        Stops at 20 matching cases, four quotes each.
        """
        node = self.by_id[args["node_id"]]
        query = str(args["query"])
        needle = query.lower()
        hits: list[dict[str, Any]] = []
        for case_id in node["cases"]:
            cid = str(case_id)
            profile = self.profiles[cid]
            matched = [
                t
                for t in profile["relevant_texts"]
                if needle in str(t).lower()
            ]
            if not matched:
                continue
            hits.append(
                {
                    "id": cid,
                    "quotes": [
                        _truncate(str(t), RELEVANT_TEXT_CHARS)
                        for t in matched[:4]
                    ],
                }
            )
            if len(hits) >= 20:
                break
        return _cap(
            json.dumps(
                {"node_id": node["id"], "query": query, "hits": hits},
                ensure_ascii=False,
            )
        )


# ── Revision ─────────────────────────────────────────────────────────────────


def _format_cases(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    return json.dumps(rows, indent=2, ensure_ascii=False)


def _live(
    node: dict[str, Any], overlay_nodes: dict[str, dict[str, Any]], field: str
) -> str:
    """The overlay's revision of ``field`` when there is one, else the tree's."""
    nid = node["id"]
    if nid in overlay_nodes and field in overlay_nodes[nid]:
        return overlay_nodes[nid][field]
    return node[field]


def used_labels_except(
    nodes: list[dict[str, Any]],
    overlay_nodes: dict[str, dict[str, Any]],
    exclude_ids: set[str],
) -> list[str]:
    """Live non-root labels, skipping ``exclude_ids`` and empty labels."""
    labels: list[str] = []
    for node in nodes:
        if node["id"] == ROOT_ID or node["id"] in exclude_ids:
            continue
        label = _live(node, overlay_nodes, "label")
        if label:
            labels.append(label)
    return labels


def _build_revision_prompt(
    constraint: str,
    parent: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    overlay_nodes: dict[str, dict[str, Any]],
    used_labels: list[str],
    not_labels: list[str],
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    sample_size: int,
) -> str:
    used_text = (
        "\n".join(f"- {label}" for label in used_labels)
        if used_labels
        else "(none)"
    )
    not_text = (
        "\n".join(f"- {label}" for label in not_labels)
        if not_labels
        else "(none)"
    )
    return (REVISION_USER_TEMPLATE + TOOLS_ADDENDUM).format(
        formalization=utils.FORMALIZATION,
        constraint=constraint,
        axis_type=parent["axis_type"],
        split_criterion_short=_live(parent, overlay_nodes, "split_criterion_short"),
        decision_rule=parent["decision_rule"],
        left_id=left["id"],
        left_description=_live(left, overlay_nodes, "description"),
        right_id=right["id"],
        right_description=_live(right, overlay_nodes, "description"),
        not_labels=not_text,
        used_labels=used_text,
        sample_size=sample_size,
        n_left=len(left_rows),
        n_right=len(right_rows),
        left_cases=_format_cases(left_rows),
        right_cases=_format_cases(right_rows),
    )


def _normalize_revision(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_criterion_short": str(raw["split_criterion_short"]).strip(),
        "left_label": str(raw["left_label"]).strip(),
        "left_summary": str(raw["left_summary"]).strip(),
        "left_description": str(raw["left_description"]).strip(),
        "right_label": str(raw["right_label"]).strip(),
        "right_summary": str(raw["right_summary"]).strip(),
        "right_description": str(raw["right_description"]).strip(),
        "rationale": str(raw["rationale"]).strip(),
    }


def _call_revision(
    prompt: str, execute: Callable[[str, dict[str, Any]], str]
) -> dict[str, Any]:
    parsed = api_calls.query_llm_with_tools(
        prompt,
        system_prompt=_LABEL_SYSTEM,
        model=LABEL_MODEL,
        tools=client_tools(),
        execute=execute,
        terminal_tool=TERMINAL_TOOL,
        web_search=True,
    )
    return _normalize_revision(parsed)


def _internal_pairs(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    internal = [
        node
        for node in nodes
        if not node["is_leaf"] and node["left_id"] and node["right_id"]
    ]
    return sorted(internal, key=lambda node: (node["depth"], node["id"]))


def _revise_pair(
    parent: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    overlay_nodes: dict[str, dict[str, Any]],
    nodes: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    case_meta: dict[str, dict[str, Any]],
    constraint: str,
) -> None:
    left = by_id[parent["left_id"]]
    right = by_id[parent["right_id"]]
    assignments = parent["assignments"]
    rng = random.Random(f"{SAMPLE_SEED}:{parent['id']}")
    left_ids = sample_side(assignments, "left", SAMPLE_SIZE, rng)
    right_ids = sample_side(assignments, "right", SAMPLE_SIZE, rng)
    used = used_labels_except(nodes, overlay_nodes, {left["id"], right["id"]})
    not_labels = tree_echo._path_not_labels(
        tree_expand._lineage_steps_for(parent["id"], by_id)
    )
    prompt = _build_revision_prompt(
        constraint,
        parent,
        left,
        right,
        overlay_nodes,
        used,
        not_labels,
        [
            _compact_case(i, assignments[i], profiles[i], case_meta[i])
            for i in left_ids
        ],
        [
            _compact_case(i, assignments[i], profiles[i], case_meta[i])
            for i in right_ids
        ],
        SAMPLE_SIZE,
    )
    shown_ids = {left["id"]: set(left_ids), right["id"]: set(right_ids)}
    execute = LabelTools(
        by_id, profiles, case_meta, shown_ids, SAMPLE_SIZE
    ).execute
    result = _call_revision(prompt, execute)
    overlay_nodes.setdefault(parent["id"], {})["split_criterion_short"] = result[
        "split_criterion_short"
    ]
    for child, side in ((left, "left"), (right, "right")):
        overlay_nodes.setdefault(child["id"], {}).update(
            label=result[f"{side}_label"],
            summary=result[f"{side}_summary"],
            description=result[f"{side}_description"],
            rationale=result["rationale"],
            parent_id=parent["id"],
        )


def revise_labels(run_dir: Path) -> Path:
    """Write the overlay for one completed tree run; return its path.

    Revises every sibling pair top-down.
    """
    run_dir = Path(run_dir)
    corpus = json.loads((run_dir / "corpus_level.json").read_text())
    meta = json.loads((run_dir / "tree_run_metadata.json").read_text())
    profiles = {
        str(row["case_id"] if "case_id" in row else row["id"]): row["profile"]
        for row in json.loads(Path(meta["case_level_path"]).read_text())
    }
    index = json.loads((run_dir / "case_index.json").read_text())
    case_meta = {str(row["case_id"]): row for row in index["cases"]}
    nodes = corpus["nodes"]
    by_id = {node["id"]: node for node in nodes}
    overlay_nodes: dict[str, dict[str, Any]] = {}

    pairs = _internal_pairs(nodes)
    print(f"[labels] {run_dir.name}: {len(pairs)} sibling pair(s)")
    for parent in pairs:
        print(
            f"[labels] {parent['id']}: {parent['left_id']} / {parent['right_id']}"
        )
        _revise_pair(
            parent,
            by_id,
            overlay_nodes,
            nodes,
            profiles,
            case_meta,
            meta["constraint"],
        )

    out_path = run_dir / OVERLAY_NAME
    out_path.write_text(
        json.dumps(
            {
                "model": LABEL_MODEL,
                "sample_size": SAMPLE_SIZE,
                "nodes": overlay_nodes,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"[labels] wrote {out_path} ({len(overlay_nodes)} node(s))")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doctrine", action="append", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    for doctrine in args.doctrine:
        revise_labels(
            root / "doctrines" / doctrine / "analysis" / "runs" / args.run_id
        )


if __name__ == "__main__":
    main()
