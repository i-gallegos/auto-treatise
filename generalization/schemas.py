from __future__ import annotations

from typing import Any, Sequence

#: Roles the annotator can place at a node. Order is fixed so walker keys and
#: artifact rows stay stable.
ANNOTATION_ROLES = ("court", "party_a", "party_b")

# Field descriptions that ``prompts`` repeats. Both modules read these
# constants, so the schema and the prompt cannot drift apart.
ROLE_DESC = "Name this litigant: petitioner/respondent, appellant/appellee, plaintiff/defendant, or the opinion's own labels. Name each litigant once and keep the identity stable. A role is a person, not a tree side."
QUESTION_PRESENTED_DESC = "State the issue this case raises that bears on the constraint, as an open question the court had yet to answer: 'whether X', not 'X'. If the case raises several issues, use the one the constraint governs. A competent lawyer must be able to lose the question on the material you supply. Do not use 'already', 'nonetheless', 'squarely', 'the very', 'even', or any comparative that makes one answer obvious. If constraint_engagement is 'not_engaged', state the closest account of what the case is about. Do not invent a question presented."
OTHER_ISSUES_DESC = "List the other issues this case decides, one short clause each. This sets them apart from the question presented. Use an empty array when the constraint question is the only one."
COMPETING_THEORIES_DESC = "List the competing rules the parties asked the court to adopt on the constraint question, normally one per side. These were on the record before the court ruled. List a theory only if the opinion attributes it to a party. Do not list a theory the court raised on its own. Use an empty array when the opinion does not report the parties' positions."
CONSTRAINT_ENGAGEMENT_DESC = "Rate how much the opinion engages the constraint. 'squarely': the case turns on what the constraint means or how it applies. 'incidentally': the constraint is raised but does not decide the case. 'assumed': the constraint is settled background framing another dispute. 'not_engaged': the constraint plays no part in the opinion. The case is in this corpus because it cites the seed authority for an unrelated proposition or shares vocabulary with it. Use 'not_engaged' for absence, not brevity. A one-paragraph treatment is 'incidentally'. A case about a different privilege, statute, or doctrine is 'not_engaged'."


def routing_schema() -> dict[str, Any]:
    """One classifier routing decision at one internal node.
    """
    return {
        "name": "node_routing",
        "schema": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "Use 1-2 sentences. Name the feature of THIS case that the split criterion keys on. Do not restate the criterion.",
                },
                "side": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "The side the split criterion assigns this case to.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": "Use 'low' when the case sits on the boundary or the feature the criterion keys on is absent from the material shown.",
                },
            },
            "required": ["rationale", "side", "confidence"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _annotation_decision_schema(
    side_enum: tuple[str, ...], include_role: bool
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    if include_role:
        properties["role"] = {
            "type": "string",
            "description": ROLE_DESC,
        }
        required.append("role")
    properties["rationale"] = {
        "type": "string",
        "description": """Use 1-2 sentences. Name the feature of THIS case that the split criterion keys on for this role. Do not restate the criterion.""",
    }
    properties["side"] = {
        "type": "string",
        "enum": list(side_enum),
        "description": """The side this role takes. The court must choose left or right. Use not_pressed for a party that took no position on THIS distinction. Do not invent a party position.""",
    }
    properties["confidence"] = {
        "type": "string",
        "enum": ["high", "low"],
        "description": """Use 'low' when the case sits on the boundary or the feature the criterion keys on is thin in the opinion.""",
    }
    required.extend(["rationale", "side", "confidence"])
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def annotation_routing_schema(
    roles: Sequence[str] | None = None,
    named_parties: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Annotator decision at one node, for the roles still on this branch.

    ``roles`` defaults to all three. ``named_parties`` are party keys whose
    identity was already recorded, so the schema does not ask for ``role``
    again.
    """
    remaining = tuple(
        r for r in ANNOTATION_ROLES if r in (roles or ANNOTATION_ROLES)
    )
    if not remaining:
        raise ValueError("annotation schema needs at least one role")
    named = set(named_parties or ())
    properties: dict[str, Any] = {}
    for role in remaining:
        if role == "court":
            properties[role] = _annotation_decision_schema(
                ("left", "right"), False
            )
        else:
            properties[role] = _annotation_decision_schema(
                ("left", "right", "not_pressed"),
                include_role=role not in named,
            )
    return {
        "name": "annotation_routing",
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


def fact_view_schema() -> dict[str, Any]:
    """``x^[u]``: the constraint-relevant issue, and the case that raises it.
    """
    return {
        "name": "constraint_fact_view",
        "schema": {
            "type": "object",
            "properties": {
                "question_presented": {
                    "type": "string",
                    "description": QUESTION_PRESENTED_DESC,
                },
                "procedural_posture": {
                    "type": "string",
                    "description": "Use 2-4 sentences. State how the question reached this court and on what motion or appeal. Include the standard of review and who bears the burden if the opinion states them. Omit either if the opinion is silent. Describe the posture when the court took up the question, not what it did: no ruling, no disposition, no 'the court granted'.",
                },
                "material_facts": {
                    "type": "string",
                    "description": "Use 1-3 paragraphs of plain prose. State the facts bearing on the question presented as the record had them BEFORE the court answered it: who the parties are, what happened, what each side sought. Include a fact if a court would need it to answer the question. Attribute every legal characterization to the party that advanced it. Report a disputed fact as disputed. Where the record cuts both ways, give both. Do not infer a fact from the outcome.",
                },
                "other_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": OTHER_ISSUES_DESC,
                },
                "competing_theories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "party": {
                                "type": "string",
                                "description": "Who advanced this theory, in the opinion's own terms (e.g. 'petitioner', 'the Government', 'the BIA', 'amici for respondent').",
                            },
                            "theory": {
                                "type": "string",
                                "description": "State the rule this party urged the court to apply to the constraint question, as the rule they asked for, not the court's answer. Use 1-2 sentences. Do not say whether it succeeded. Do not use 'correctly', 'persuasively', 'unavailingly', or any word that scores the argument.",
                            },
                        },
                        "required": ["party", "theory"],
                        "additionalProperties": False,
                    },
                    "description": COMPETING_THEORIES_DESC,
                },
                "constraint_engagement": {
                    "type": "string",
                    "enum": [
                        "squarely",
                        "incidentally",
                        "assumed",
                        "not_engaged",
                    ],
                    "description": CONSTRAINT_ENGAGEMENT_DESC,
                },
            },
            "required": [
                "question_presented",
                "procedural_posture",
                "material_facts",
                "other_issues",
                "competing_theories",
                "constraint_engagement",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }
