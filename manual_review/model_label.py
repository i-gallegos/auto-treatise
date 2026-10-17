"""Second rater for the manual-review placements.

Gemini 2.5 Pro reads the secondary-source quotes stored for a case, its
majority opinion and the Auto-Treatise walk, then records the same verdict the
human recorded: agree, disagree, no-sources-agree or no-sources-disagree. It
never sees the human's verdict or notes.

    python -m manual_review.model_label --dry-run
    bash manual_review/gemini_judge.sh
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

from auto_treatise.labels import apply_overlay
from auto_treatise.labels import load_overlay
from auto_treatise.majority import MIN_BOUNDARY_PARAGRAPH_CHARS
from manual_review.data import assignment_rows
from manual_review.data import cases_db_path
from manual_review.data import CaseAnnotation
from manual_review.data import connect_readonly
from manual_review.data import DEFAULT_OUT_DIR
from manual_review.data import DOCTRINE_LABELS
from manual_review.data import load_run
from manual_review.data import read_annotations
from manual_review.data import runs_dir

SCHEMA = "model_review_v1"
DEFAULT_MODEL = "gemini-2.5-pro"
MAX_OUTPUT_TOKENS = 16384
THINKING_BUDGET = 4096

CASE_VERDICTS: tuple[str, ...] = (
    "agree",
    "disagree",
    "no-sources-agree",
    "no-sources-disagree",
)
BRANCH_VERDICTS: tuple[str, ...] = ("agree", "disagree")
BRANCH_SIDES: tuple[str, ...] = ("left", "right")

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tree_agreement", "tree_characterization", "branches"],
    "properties": {
        "tree_agreement": {"type": "string", "enum": list(CASE_VERDICTS)},
        "tree_characterization": {"type": "string"},
        "branches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["depth", "node_id", "side", "verdict"],
                "properties": {
                    "depth": {"type": "integer"},
                    "node_id": {"type": "string"},
                    "side": {"type": "string", "enum": list(BRANCH_SIDES)},
                    "verdict": {
                        "type": "string",
                        "enum": list(BRANCH_VERDICTS),
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You are an legal expert annotator in a legal annotation study. Compare an induced decision tree, the Auto-Treatise, to the secondary sources collected for one case and to that case's majority opinion. Record whether the tree's placement matches those materials. Judge only the placement you are shown.

You will be shown, in this order: the constraint the tree is about, the secondary-source snippets, the majority opinion, then the Auto-Treatise placement.

## CASE-LEVEL VERDICT
Return exactly one of these four:
- agree: a secondary source is shown, and at least one source characterizes the case the way this placement does.
- disagree: a secondary source is shown, and the sources characterize the case differently from this placement.
- no-sources-agree: no secondary source is shown. Judged from the majority opinion, the placement matches what the court did.
- no-sources-disagree: no secondary source is shown. Judged from the majority opinion, the placement does not match what the court did.
If any secondary source is present, return agree or disagree. Do not return a no-sources verdict for a source that addresses a different issue. Return a no-sources verdict only when no secondary source is shown.
A source characterizes the case the way the placement does when it describes this case as an instance of that placement's legal characterization. The placement may be the leaf. When the source does not speak to the leaf, the placement may be an internal node the source does address. Language that appears only in a concurrence does not characterize the disposition of the majority.

## PER-BRANCH MARKS
Mark a branch only when a snippet addresses that split's question. Say whether the snippet places the case on the side the tree took (agree) or the other side (disagree). Omit every split the sources do not address. An omitted split means the sources are silent there. When the case-level verdict is no-sources-agree or no-sources-disagree, return an empty branches list.
"""


_LEAK_FIELDS = ("notes", "human_characterization", "tree_characterization")


def output_dir(model: str, run_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip()).strip("-")
    return DEFAULT_OUT_DIR / "model_reviews" / f"{safe}_{run_id}"


def judged_cases(
    run_id: str, db_path: Path | None = None
) -> list[CaseAnnotation]:
    """Training cases the human judged on ``run_id``, in review order."""
    return [
        case
        for case in read_annotations(run_id, db_path)
        if case.split == "train" and case.tree_agreement
    ]


def load_selection(csv_path: Path) -> dict[str, dict[str, str]]:
    """``selected_cases.csv`` keyed by ``review_id``."""
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        return {row["review_id"]: row for row in csv.DictReader(fh)}


def load_constraint(doctrine: str, run_id: str) -> str:
    """The constraint this tree was induced on."""
    meta = json.loads(
        (runs_dir(doctrine) / run_id / "tree_run_metadata.json").read_text()
    )
    return meta["constraint"].strip()


def majority_span(
    paragraphs: list[str], info: dict[str, Any]
) -> tuple[str, str]:
    """Majority text from the stored ``majority_start`` / ``end_index`` /
    offset.
    """
    n = len(paragraphs)
    start = 0
    if "majority_start" in info and info["majority_start"] is not None:
        start = int(info["majority_start"])
    end = n
    if "end_index" in info and info["end_index"] is not None:
        end = int(info["end_index"])
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    parts = [
        paragraph.strip()
        for paragraph in paragraphs[start:end]
        if paragraph.strip()
    ]
    offset = info["end_char_offset"] if "end_char_offset" in info else None
    if offset is not None and end < n:
        head = paragraphs[end][: int(offset)].strip()
        if len(head) >= MIN_BOUNDARY_PARAGRAPH_CHARS:
            parts.append(head)
    text = "\n\n".join(parts)
    if not text:
        return "", "missing"
    status = info["status"] if "status" in info else ""
    if status == "majority_only" and start == 0 and end == n and offset is None:
        return text, "majority_only"
    return text, "majority_boundary"


def load_opinions(cases: list[CaseAnnotation]) -> dict[str, tuple[str, str]]:
    """``review_id -> (majority text, provenance)``, joined on opinion id."""
    found: dict[str, tuple[str, str]] = {}
    for doctrine in {case.doctrine for case in cases}:
        conn = connect_readonly(cases_db_path(doctrine))
        for case in cases:
            if case.doctrine != doctrine:
                continue
            row = conn.execute(
                """SELECT opinion_paragraphs_clean_json, majority_boundary_json FROM cases WHERE CAST(courtlistener_opinion_id AS TEXT) = ? LIMIT 1""",
                (str(case.cl_opinion_id),),
            ).fetchone()
            if row is None or not row[0] or not row[1]:
                found[case.review_id] = ("", "missing")
            else:
                found[case.review_id] = majority_span(
                    json.loads(row[0]), json.loads(row[1])
                )
        conn.close()
    return found


def load_lineages(
    cases: list[CaseAnnotation],
    run_id: str,
    selection_csv: Path,
) -> dict[str, dict[str, Any]]:
    """Overlay-labelled walk for each judged case, keyed by ``review_id``."""
    selection = load_selection(selection_csv)
    lineages: dict[str, dict[str, Any]] = {}
    for doctrine in {case.doctrine for case in cases}:
        run = load_run(doctrine, run_id)
        overlay = load_overlay(run.path)
        corpus = {"root_id": run.root_id, "nodes": list(run.nodes.values())}
        patched = apply_overlay(corpus, overlay)
        run.nodes = {node["id"]: node for node in patched["nodes"]}
        records = [
            selection[case.review_id]
            for case in cases
            if case.doctrine == doctrine
        ]
        for row in assignment_rows(doctrine, records, run=run):
            leaf_id = row["leaf_id"]
            lineages[row["review_id"]] = {
                "leaf_id": leaf_id,
                "leaf_label": row["leaf_label"],
                "leaf_summary": (
                    run.nodes[leaf_id]["summary"] if leaf_id else ""
                ),
                "steps": row["steps"],
            }
    return lineages


def case_packet(
    annotation: CaseAnnotation,
    lineage: dict[str, Any],
    opinion_text: str,
    opinion_provenance: str,
    constraint: str,
) -> dict[str, Any]:
    """The materials the model is allowed to see.
    """
    sources = []
    for source in annotation.sources:
        sources.append(
            {
                "citation": source.citation,
                "quotes": [
                    {"pages": quote.pages, "quote": quote.quote}
                    for quote in source.quotes
                ],
            }
        )
    return {
        "review_id": annotation.review_id,
        "doctrine": annotation.doctrine,
        "doctrine_label": (
            DOCTRINE_LABELS[annotation.doctrine]
            if annotation.doctrine in DOCTRINE_LABELS
            else annotation.doctrine
        ),
        "constraint": constraint,
        "split": annotation.split,
        "case_name": annotation.case_name,
        "citation": annotation.effective_citation,
        "sources": sources,
        "opinion_text": opinion_text,
        "opinion_provenance": opinion_provenance,
        "leaf_id": lineage["leaf_id"],
        "leaf_label": lineage["leaf_label"],
        "leaf_summary": lineage["leaf_summary"],
        "steps": lineage["steps"],
    }


def render_sources(sources: list[dict[str, Any]]) -> str:
    """The secondary-source block of the prompt, one numbered block each."""
    if not sources:
        return "No secondary source was found for this case."
    blocks = []
    for index, source in enumerate(sources, start=1):
        citation = str(source["citation"] or "").strip()
        lines = [f"Source {index}: {citation}".rstrip()]
        quotes = [
            quote
            for quote in source["quotes"]
            if str(quote["quote"] or "").strip()
        ]
        if not quotes:
            lines.append("(no quotation stored)")
        for quote in quotes:
            pages = str(quote["pages"] or "").strip()
            pin = f"at {pages}: " if pages else ""
            lines.append(pin + str(quote["quote"]).strip())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_lineage(packet: dict[str, Any]) -> str:
    """The placement block of the prompt: the leaf, then each branch."""
    leaf = packet["leaf_label"] or packet["leaf_id"] or "(no leaf)"
    lines = [f"Leaf: {leaf}"]
    if packet["leaf_id"]:
        lines.append(f"Leaf id: {packet['leaf_id']}")
    if packet["leaf_summary"]:
        lines.append(f"Summary: {packet['leaf_summary']}")
    if not packet["steps"]:
        lines.append("The tree recorded no branch for this case.")
        return "\n".join(lines)
    lines.append("")
    lines.append("Branches, root first. Each is a split the tree applied.")
    for step in packet["steps"]:
        lines.append(
            f"\nDepth {step['depth']}: {step['node_id']} -> {step['side']}"
        )
        if step["axis"]:
            lines.append(f"Question: {step['axis']}")
        if step["label"]:
            lines.append(f"Branch taken: {step['label']}")
        if step["why"]:
            lines.append(f"Why the tree took it: {step['why']}")
    return "\n".join(lines)


def render_prompt(packet: dict[str, Any]) -> str:
    """User message: constraint, sources, majority opinion, then the
    placement."""
    identity = "\n".join(
        [
            f"Case: {packet['case_name']}",
            f"Citation: {packet['citation'] or '(none on file)'}",
            f"Doctrine: {packet['doctrine_label']}",
        ]
    )
    constraint = packet["constraint"] or "(constraint text is not on file)"
    return "\n".join(
        [
            "# LEGAL RULE",
            "The Auto-Treatise is a decision tree of how courts have interpreted and applied the following legal rule in different ways. Judge the placement against that rule, not against some other issue the opinion happens to decide.",
            "",
            constraint,
            "",
            "# CASE",
            identity,
            "",
            "# SECONDARY SOURCES",
            render_sources(packet["sources"]),
            "",
            "# MAJORITY OPINION",
            "",
            packet["opinion_text"] or "(majority opinion text is not on file)",
            "",
            "# AUTO-TREATISE PLACEMENT",
            render_lineage(packet),
            "",
            "# TASK",
            "Judge this placement against the secondary sources and the majority opinion, using only the four case-level verdicts named in your instructions. If any secondary source is present, the case-level verdict must be agree or disagree. The placement may be the leaf, or an internal node when the source does not speak to the leaf. Mark a branch only when a snippet addresses that split. Omit every other branch.",
        ]
    )


def leaked_human_fields(
    prompt: str,
    annotation: CaseAnnotation,
    opinion_text: str = "",
) -> list[str]:
    """Long reviewer prose that is not itself a source quotation or the opinion.
    """
    source_blob = "\n".join(
        quote.quote for source in annotation.sources for quote in source.quotes
    )
    found = []
    values = {
        "notes": annotation.notes,
        "human_characterization": annotation.human_characterization,
        "tree_characterization": annotation.tree_characterization,
    }
    for name in _LEAK_FIELDS:
        body = values[name].strip()
        if len(body) < 80:
            continue
        if body in source_blob or body in opinion_text:
            continue
        if body in prompt:
            found.append(name)
    return found


def parse_model_response(payload: str | dict) -> dict[str, Any]:
    """Validate one structured response."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    verdict = payload["tree_agreement"]
    if verdict not in CASE_VERDICTS:
        raise ValueError(
            f"tree_agreement {verdict!r} is not one of {CASE_VERDICTS}"
        )
    branches = []
    for branch in payload["branches"]:
        if branch["side"] not in BRANCH_SIDES:
            raise ValueError(
                f"branch side {branch['side']!r} is not left or right"
            )
        if branch["verdict"] not in BRANCH_VERDICTS:
            raise ValueError(
                f"branch verdict {branch['verdict']!r} is not agree or disagree"
            )
        branches.append(
            {
                "depth": int(branch["depth"]),
                "node_id": str(branch["node_id"]),
                "side": branch["side"],
                "verdict": branch["verdict"],
            }
        )
    return {
        "tree_agreement": verdict,
        "tree_characterization": str(payload["tree_characterization"]),
        "branches": branches,
    }


def build_packets(
    cases: list[CaseAnnotation],
    lineages: dict[str, dict[str, Any]],
    opinions: dict[str, tuple[str, str]],
    constraints: dict[str, str],
) -> list[dict[str, Any]]:
    packets = []
    for case in cases:
        text, provenance = opinions[case.review_id]
        packets.append(
            case_packet(
                case,
                lineages[case.review_id],
                text,
                provenance,
                constraints[case.doctrine],
            )
        )
    return packets


def label_record(
    packet: dict[str, Any], parsed: dict[str, Any], raw: Any
) -> dict[str, Any]:
    """The labels.json row for one parsed response, with the walk it judged."""
    return {
        "review_id": packet["review_id"],
        "doctrine": packet["doctrine"],
        "split": packet["split"],
        "n_sources": len(packet["sources"]),
        "opinion_provenance": packet["opinion_provenance"],
        "tree_agreement": parsed["tree_agreement"],
        "tree_characterization": parsed["tree_characterization"],
        "branches": parsed["branches"],
        "walk": [
            {
                "depth": step["depth"],
                "node_id": step["node_id"],
                "side": step["side"],
            }
            for step in packet["steps"]
        ],
        "raw": raw,
    }


def _run_gemini_batch(
    packets: list[dict[str, Any]], model: str, *, gcs_prefix: str
) -> list[str]:
    """One Vertex batch; the response text for each packet, in order."""
    from vertex import run_requests

    by_key = run_requests(
        [(packet["review_id"], render_prompt(packet)) for packet in packets],
        model=model,
        thinking_budget=THINKING_BUDGET,
        mode="batch",
        system_instruction=SYSTEM_PROMPT,
        response_schema=RESPONSE_SCHEMA,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        gcs_prefix=gcs_prefix,
    )
    return [by_key[packet["review_id"]] for packet in packets]


def make_packets(
    cases: list[CaseAnnotation], run_id: str
) -> list[dict[str, Any]]:
    """Prompt packets for ``cases`` on ``run_id``. Exits if a prompt contains
    reviewer prose."""
    opinions = load_opinions(cases)
    lineages = load_lineages(
        cases,
        run_id,
        DEFAULT_OUT_DIR / "selected_cases.csv",
    )
    constraints = {
        doctrine: load_constraint(doctrine, run_id)
        for doctrine in {case.doctrine for case in cases}
    }
    packets = build_packets(cases, lineages, opinions, constraints)
    leaks = []
    for packet, case in zip(packets, cases):
        leaks.extend(
            (packet["review_id"], name)
            for name in leaked_human_fields(
                render_prompt(packet), case, packet["opinion_text"]
            )
        )
    if leaks:
        sys.exit(f"prompt contains reviewer prose: {leaks}")
    return packets


def write_labels(
    packets: list[dict[str, Any]], run_id: str, model: str = DEFAULT_MODEL
) -> Path:
    """Judge the packets in one batch and write ``labels.json``."""
    texts = _run_gemini_batch(
        packets, model, gcs_prefix=f"manual-review/{model}/{run_id}"
    )
    payload = {
        "schema": SCHEMA,
        "model": model,
        "run_id": run_id,
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": [
            label_record(packet, parse_model_response(text), text)
            for packet, text in zip(packets, texts)
        ],
    }
    path = output_dir(model, run_id) / "labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, metavar="RUN_ID")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pack the prompts and do not call the API",
    )
    args = parser.parse_args()

    packets = make_packets(judged_cases(args.run), args.run)
    if args.dry_run:
        for packet in packets:
            print(
                f"""{packet['review_id']}  sources={len(packet['sources'])}  opinion_chars={len(packet['opinion_text'])}  steps={len(packet['steps'])}  prompt_chars={len(render_prompt(packet))}  provenance={packet['opinion_provenance']}"""
            )
        print(f"dry-run: {len(packets)} prompt(s), no API call")
        return
    path = write_labels(packets, args.run, args.model)
    print(f"wrote {path} ({len(packets)} labeled)")


if __name__ == "__main__":
    main()
