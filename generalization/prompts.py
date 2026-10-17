"""Prompts for the generalization task.

Three prompts, built here and executed in ``views`` / ``classify`` /
``annotate``:

* ``build_extraction_prompt`` — write ``x^[u]`` for one case: the rule-relevant
  issue, the posture, and the facts, with the outcome withheld (``views``).
* ``build_routing_prompt``    — classify ``x^[u]`` at one internal node.
* ``build_annotation_prompt`` — label the court leaf and each party's leaf
  from the majority opinion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from auto_treatise.utils import FORMALIZATION  # noqa: E402
from generalization.schemas import COMPETING_THEORIES_DESC  # noqa: E402
from generalization.schemas import CONSTRAINT_ENGAGEMENT_DESC  # noqa: E402
from generalization.schemas import OTHER_ISSUES_DESC  # noqa: E402
from generalization.schemas import QUESTION_PRESENTED_DESC  # noqa: E402
from generalization.schemas import ROLE_DESC  # noqa: E402

CLASSIFIER_SYSTEM = (
    """You are a legal analyst placing a new court opinion within an existing tree of how courts have interpreted a single legal rule. The tree is a binary tree. Each internal node states a doctrinal distinction that divides the cases beneath it into two branches. Your task at each node is to apply that distinction to a new case and decide which branch it belongs on. You will be shown only the new case's facts, procedural posture, and the issue presented. You will NOT be shown how the court reasoned or what it held. Decide which branch the case belongs on by predicting, from the material available, how a court confronting these facts and this issue would approach the rule. You must choose a side even when the material is thin; use the confidence field to record how cleanly the distinction applies.

Boilerplate recitation of a legal rule is not itself evidence of engagement. Distinguish likewise between the theory a party pressed and the rule the court will adopt. The distinctions in this tree record the ground a court took, and courts routinely take up the salient theory in order to reject it. That a fact pattern is shaped like one side of a distinction is weak evidence that the court went that way. Where the two readings diverge, predict the rule, not the fact pattern. """
    + FORMALIZATION
)

#: System prompt for annotation. The input is the constraint profile and labeled
#: paragraphs, or the full majority opinion when no paragraph was relevant.
ANNOTATOR_SYSTEM = (
    """You are a legal analyst annotating where a court opinion belongs within an existing tree of how courts have interpreted a single legal rule. The tree is a binary tree. Each internal node states a doctrinal distinction that divides the cases beneath it into two branches.

You will usually be shown the case as the tree itself represents its cases: a CONSTRAINT PROFILE (issue, material facts, how the court applied the rule, its reasoning chain) followed by the paragraphs of the majority opinion relevant to the rule, pre-segmented into sentences tagged [Role | Attribution | Stance]:
  - Role:        Facts | Procedure | Issue | Argument | Holding | Reasoning
  - Attribution: CurrentCourt | Appellant | Appellee | Amicus | LowerCourt
  - Stance:      Disputed | Undisputed
The profile is a summary; the labeled paragraphs are the evidence. Where they conflict, follow the paragraphs. Occasionally, when no paragraph was found relevant, you will be shown the full majority opinion instead.

At each node you record a decision for every role still on this branch:

- COURT: which branch the court's own analysis took. This is ground truth. Base it on sentences attributed to CurrentCourt with Role Holding or Reasoning — what the court actually did with the rule — not on what it might have done, and not on a theory a party pressed in order to be rejected. The court must choose LEFT or RIGHT.
- PARTY A and PARTY B: which branch each litigant pressed on this distinction, read from sentences attributed to that party (Role Argument) and from the court's account of their arguments. These are the two sides arguing the constraint question. Set the role field once, at the root. """ + ROLE_DESC + """

Do not invent a party position. If the material does not show that party taking a side on THIS distinction, use not_pressed. A court that recites a party's argument in order to reject it is evidence of what the party pressed, and separately of what the court did.

A later node may ask only for the roles still on this branch: the others already took a different fork. """
    + FORMALIZATION
)

EXTRACTOR_SYSTEM = (
    """You are a legal analyst preparing a case summary for a reader who must predict how a court will resolve a question before reading the answer.

You will be given the full text of a court's majority opinion and a legal rule under study. Write the summary a well-prepared reader would have had when the case was argued, before the court ruled: the question the court had to answer about the rule, the posture it arrived in, and the facts in the record bearing on it.

## INCLUDE:
Only facts knowable BEFORE the court ruled. If a sentence could only have been written by someone who knows the outcome, remove it or rewrite it.

## NEVER INCLUDE:
- The holding or the disposition
- The court's reasoning
- Any authority, citation, or case name offered as the governing rule

# QUESTION PRESENTED (question_presented, other_issues)
"""
    + QUESTION_PRESENTED_DESC
    + """
"""
    + OTHER_ISSUES_DESC
    + """

# PARTY ARGUMENTS (competing_theories)
The reader must predict which rule the court will apply. The strongest evidence before the ruling is which rules the parties argued for.
"""
    + COMPETING_THEORIES_DESC
    + """
Legal characterizations belong here because they are attributed. Drop every word that scores an argument, including correctly, persuasively, unavailingly, and any hint of which side prevailed. A reported theory must not tell the reader whether the court took it.

# NOT ENGAGED (constraint_engagement)
This opinion was gathered because it cites the authority the rule comes from. The case may cite that authority for something else, and then the rule under study is not present.
"""
    + CONSTRAINT_ENGAGEMENT_DESC
    + """

# SPARSE OPINIONS
Some opinions recite almost no facts. Write what is there and stop. Do not fill gaps with what a case like this usually involves. Do not infer facts from the outcome. Do not pad.

Write plain, concrete prose. Prefer the record's own vocabulary for facts."""
)

_CONSTRAINT_BLOCK = """# LEGAL RULE

{rule}
"""

_PATH_BLOCK = """# TREE

{path}
"""

_NODE_BLOCK = """# DISTINCTION

{node}
"""

_FACTS_BLOCK = """# NEW CASE

{header}

{view}
"""

_AUTHORITY_BLOCK = """# BINDING AUTHORITIES

Below are the cases already on the tree that stand in an authority relation to the court deciding the new case, with the position each was given. The new case was decided by a court that has to work within them.

- BINDING is vertical stare decisis: this court must follow these. Where a binding case sits on the tree is strong evidence for where this one belongs, unless the new case is materially distinguishable on the very feature the distinction keys on — in which case say what that feature is. If these binding courts conflict, say so in your reasoning.
- SAME COURT is this court's own prior decisions. Not formally binding on itself (and for a district court, not binding on other judges of the same district), but courts are reluctant to depart from them without saying so.
- WITHIN THE SAME JURISDICTION is persuasive only: sister district courts of the same circuit, or coordinate courts of the same state. Weigh it as the local consensus, not as a rule.

A group that is empty means that the tree contains no such case.

{context}
"""

_FULL_BLOCK = """# MAJORITY OPINION

{header}

{view}
"""

_SEGMENTED_BLOCK = """# CASE (constraint profile and labeled relevant paragraphs of the majority opinion)

{header}

{view}
"""

#: ``view_kind`` -> the block a case's text is shown in.
_VIEW_BLOCKS = {"majority": _FULL_BLOCK, "segmented_profile": _SEGMENTED_BLOCK}


_ROUTING_TASK = """# TASK

Decide whether this case belongs on the LEFT or the RIGHT branch of the distinction above. Return the side, a rationale, and your confidence."""

_LITIGANTS_BLOCK = """# LITIGANTS

{litigants}
"""


@dataclass(frozen=True)
class RoutingPrompt:
    """One routing decision, split for provider-side prefix caching."""

    prefix: str
    suffix: str

    @property
    def text(self) -> str:
        return f"{self.prefix}\n\n{self.suffix}"

    def __str__(self) -> str:
        return self.text


def _render_path(tree, node_id: str) -> str:
    """The chain of distinctions already applied to reach ``node_id``.
    """
    chain: list[str] = []
    current = tree.nodes[node_id]
    while current["parent_id"]:
        parent = tree.nodes[current["parent_id"]]
        side = "LEFT" if parent["left_id"] == current["id"] else "RIGHT"
        criterion = (
            parent["split_criterion_short"] or parent["split_criterion"] or ""
        )
        chain.append(
            f"- {criterion} -> {side} ({current['label'] or current['id']})"
        )
        current = parent
    if not chain:
        return "This is the root of the tree. No distinctions have been applied yet."
    return "\n".join(reversed(chain))


def build_routing_prompt(
    tree,
    node_id: str,
    rule: str,
    case_header: str,
    case_view: str,
    authority_context: str,
) -> RoutingPrompt:
    """Prompt for one routing decision over ``x^[u]``.
    """
    view_block = _FACTS_BLOCK.format(header=case_header, view=case_view)
    cached = [_CONSTRAINT_BLOCK.format(rule=rule), view_block]
    if authority_context.strip():
        cached.append(
            _AUTHORITY_BLOCK.format(context=authority_context.strip())
        )
    varying = [
        _PATH_BLOCK.format(path=_render_path(tree, node_id)),
        _NODE_BLOCK.format(node=tree.describe_node(node_id)),
        _ROUTING_TASK,
    ]
    return RoutingPrompt(
        prefix="\n\n".join(cached), suffix="\n\n".join(varying)
    )


def _annotation_task(
    roles: tuple[str, ...],
    party_names: dict[str, str] | None = None,
) -> str:
    names = party_names or {}
    lines = [
        "# TASK",
        "",
        "At this distinction, record a decision for each remaining role:",
    ]
    if "court" in roles:
        lines.append(
            """- COURT: the branch the court's own analysis took (LEFT or RIGHT). This is ground truth."""
        )
    for key, label in (("party_a", "PARTY A"), ("party_b", "PARTY B")):
        if key not in roles:
            continue
        who = (
            names[key]
            if key in names
            else "name this litigant in the role field"
        )
        lines.append(
            f"""- {label} ({who}): the branch this party pressed, or not_pressed if they took no position on THIS distinction."""
        )
    lines.append("")
    lines.append(
        """A court that recites an argument in order to reject it is evidence of what that party pressed, and separately of what the court did."""
    )
    if len(roles) < 3:
        lines.append(
            "Roles not listed already took a different branch; do not re-decide them."
        )
    return "\n".join(lines)


def _litigants_block(party_names: dict[str, str] | None) -> str | None:
    if not party_names:
        return None
    lines = []
    for key, label in (("party_a", "party_a"), ("party_b", "party_b")):
        if key in party_names:
            lines.append(f"- {label} is the {party_names[key]}")
    if not lines:
        return None
    return _LITIGANTS_BLOCK.format(litigants="\n".join(lines))


def build_annotation_prompt(
    tree,
    node_id: str,
    rule: str,
    case_header: str,
    full_view: str,
    roles: tuple[str, ...] = ("court", "party_a", "party_b"),
    party_names: dict[str, str] | None = None,
    view_kind: str = "majority",
) -> RoutingPrompt:
    """Prompt for one annotation decision over the majority opinion.
    """
    remaining = tuple(r for r in ("court", "party_a", "party_b") if r in roles)
    view_block = _VIEW_BLOCKS[view_kind].format(header=case_header, view=full_view)
    prefix = "\n\n".join(
        [
            _CONSTRAINT_BLOCK.format(rule=rule),
            _PATH_BLOCK.format(path=_render_path(tree, node_id)),
            _NODE_BLOCK.format(node=tree.describe_node(node_id)),
        ]
    )
    blocks = [view_block]
    litigants = _litigants_block(party_names)
    if litigants is not None:
        blocks.append(litigants)
    blocks.append(_annotation_task(remaining, party_names))
    return RoutingPrompt(prefix=prefix, suffix="\n\n".join(blocks))


# --------------------------------------------------------------------------- #
# Extraction and audit
# --------------------------------------------------------------------------- #

_EXTRACTION_TASK = """# FULL OPINION

{header}

{opinion}

# TASK

Write the pre-decision view of this case described in your instructions. Before you return it, re-read each sentence. Strike anything that could only have been written by someone who knows how the court ruled."""


def build_extraction_prompt(
    rule: str, case_header: str, opinion: str
) -> RoutingPrompt:
    """Prompt for writing one case's ``x^[u]``.
    """
    return RoutingPrompt(
        prefix=_CONSTRAINT_BLOCK.format(rule=rule),
        suffix=_EXTRACTION_TASK.format(header=case_header, opinion=opinion),
    )
