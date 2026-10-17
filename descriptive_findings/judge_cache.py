"""The judge-panel cache: paths, schema, load/save and author coding."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CACHE_DIR = Path(__file__).resolve().parent / "cache" / "judge_panels"


#: Provenance of an entry's judge list. ``manual`` entries are written by hand
#: and never overwritten.
SOURCE_LLM = "llm"
SOURCE_MANUAL = "manual"

LARGE_PANEL = 5

WINDOW_PARAGRAPHS = 5
WINDOW_MIN_CHARS = 1_500
WINDOW_MAX_CHARS = 6_000
TAIL_CHARS = 1_200


JUDGE_SCHEMA: dict[str, Any] = {
    "name": "opinion_panel",
    "schema": {
        "type": "object",
        "properties": {
            "per_curiam": {
                "type": "boolean",
                "description": "True if the opinion is issued per curiam.",
            },
            "judges": {
                "type": "array",
                "description": "Every judge who decided this case.",
                "items": {
                    "type": "object",
                    "properties": {
                        "last_name": {
                            "type": "string",
                            "description": "Surname as printed.",
                        },
                        "first_name": {
                            "type": "string",
                            "description": "Given name, or empty string if not printed.",
                        },
                        "middle_name": {
                            "type": "string",
                            "description": "Middle name or initial, or empty string.",
                        },
                        "role": {
                            "type": "string",
                            "enum": [
                                "authored",
                                "joined",
                                "concurring",
                                "dissenting",
                                "unknown",
                            ],
                            "description": "This judge's role in this case.",
                        },
                    },
                    "required": [
                        "last_name",
                        "first_name",
                        "middle_name",
                        "role",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["per_curiam", "judges"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ── cache ─────────────────────────────────────────────────────────────────────


def cache_path(doctrine: str) -> Path:
    return CACHE_DIR / f"{doctrine}.json"


def load_cache(doctrine: str) -> dict[str, Any]:
    """A doctrine's panel cache, always with ``meta`` and ``cases`` keys."""
    path = cache_path(doctrine)
    if not path.exists():
        return {"meta": {}, "cases": {}}
    with path.open() as f:
        payload = json.load(f)
    payload.setdefault("meta", {})
    payload.setdefault("cases", {})
    return payload


#: Court levels with an appointing president. Mirrors
#: ``judge_party._ARTICLE_III_LEVELS`` to avoid a cycle.
ARTICLE_III_LEVELS = frozenset(
    {"scotus", "federal_appellate", "federal_district"}
)


def author_mode(
    judges: list[dict[str, Any]] | None, per_curiam: bool = False
) -> str:
    """How ``judge_author_party`` should be derived from a panel.
    """
    judges = judges or []
    if any(j.get("role") == "authored" for j in judges):
        return "authored"
    if len(judges) == 1:
        return "authored"
    if per_curiam or len(judges) > LARGE_PANEL:
        return "per_curiam"
    return "unknown"


def author_indices(entry: dict[str, Any]) -> list[int]:
    """Positions in ``entry["judges"]`` of the judges who wrote the opinion.
    """
    judges = entry.get("judges") or []
    roles = [j.get("role") for j in judges]
    if "authored" in roles:
        return [i for i, r in enumerate(roles) if r == "authored"]
    if len(judges) == 1:
        return [0]
    if entry.get("per_curiam") or len(judges) > LARGE_PANEL:
        return [i for i, r in enumerate(roles) if r in ("joined", "unknown")]
    return []


def save_cache(doctrine: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` to the doctrine's cache file."""
    path = cache_path(doctrine)
    path.parent.mkdir(parents=True, exist_ok=True)
    for entry in payload["cases"].values():
        entry.setdefault("source", SOURCE_LLM)
    payload["meta"] = {
        **payload["meta"],
        "window": f"""paragraphs[:{WINDOW_PARAGRAPHS}] clamp({WINDOW_MIN_CHARS},{WINDOW_MAX_CHARS}) + tail[{TAIL_CHARS}]""",
        "n_cases": len(payload["cases"]),
        "n_manual": sum(
            1
            for e in payload["cases"].values()
            if e.get("source") == SOURCE_MANUAL
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, sort_keys=True)
    return path


# ── database ──────────────────────────────────────────────────────────────────


def clean_db_path(doctrine: str) -> Path:
    return REPO_ROOT / "doctrines" / doctrine / "data" / "citing_cases_clean.db"
