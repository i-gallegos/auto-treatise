"""Extract the deciding panel from each opinion's caption.

A cheap model reads the opening and closing of each opinion (the ``Before:``
line, the docket line, or the signature block) and lists the judges. Results
are cached in ``cache/judge_panels/<doctrine>.json``; cases already in the
cache, including hand-entered ones, are not re-extracted. ``--fill-from-fjc``
then replaces an Article III author that ``judge_party`` cannot match with the
one FJC judge on that court on that date whom the name can be.

    python -m descriptive_findings.judge_extract --doctrine loper_bright bruen
    python -m descriptive_findings.judge_extract --doctrine bruen --fill-from-fjc
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from .judge_cache import ARTICLE_III_LEVELS
from .judge_cache import author_indices
from .judge_cache import clean_db_path
from .judge_cache import JUDGE_SCHEMA
from .judge_cache import load_cache
from .judge_cache import save_cache
from .judge_cache import SOURCE_LLM
from .judge_cache import SOURCE_MANUAL
from .judge_cache import TAIL_CHARS
from .judge_cache import WINDOW_MAX_CHARS
from .judge_cache import WINDOW_MIN_CHARS
from .judge_cache import WINDOW_PARAGRAPHS
from .judge_party import FJC_CSV
from .judge_party import FjcDirectory
from .judge_party import norm_court
from .judge_party import norm_name
from .judge_party import surname_key

DEFAULT_MODEL = "gpt-5.1"
_HEAD_MARKER = "=== OPENING ==="
_TAIL_MARKER = "=== CLOSING (end of the opinion) ==="
_PANEL_MARKER = "=== PANEL LINE (found later in the opinion) ==="
_PANEL_LINE = re.compile(
    r"\bBefore:?\s+[A-Z][A-Za-z.,'\- ]{3,150}?(?:Circuit|District|Chief|Senior)\s+Judge",
    re.S,
)


# ── extraction window ─────────────────────────────────────────────────────────


def extraction_window(
    paragraphs: list[str] | None,
    full_text: str | None,
    *,
    n_paragraphs: int = WINDOW_PARAGRAPHS,
    min_chars: int = WINDOW_MIN_CHARS,
    max_chars: int = WINDOW_MAX_CHARS,
    tail_chars: int = TAIL_CHARS,
) -> str:
    """The text to send to the model: opening and closing.
    """
    text = "\n\n".join(p for p in (paragraphs or [])[:n_paragraphs] if p)
    if len(text) < min_chars:
        fallback = (full_text or "").strip()
        if len(fallback) > len(text):
            text = fallback
    head = text[:max_chars]

    body = (full_text or "").strip()
    tail = body[-tail_chars:] if len(body) > len(head) else ""
    # A short opinion is entirely inside the head; appending its own ending
    # would just pay twice for the same text.
    if tail and tail in head:
        tail = ""

    # Rescue a panel line that fell between the head cap and the tail.
    panel = ""
    match = _PANEL_LINE.search(body)
    if match and match.group(0) not in head and match.group(0) not in tail:
        panel = match.group(0)

    if not tail and not panel:
        return head

    parts = [f"{_HEAD_MARKER}\n{head}"]
    if panel:
        parts.append(f"{_PANEL_MARKER}\n{panel}")
    if tail:
        parts.append(f"{_TAIL_MARKER}\n{tail}")
    return "\n\n".join(parts)


# ── prompt and schema ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You extract the deciding judges from the caption of a U.S. judicial opinion. You are precise and you never guess: a name you cannot find in the text is a name you do not report."""


USER_TEMPLATE = """Below is a U.S. judicial opinion, given as its opening text and
(where the opinion is long enough) its closing text. The two sections are
labelled. Read both: an appellate opinion names its panel at the start, while a
district opinion usually names nobody at the start and identifies its judge only
in the signature block at the end.

List every judge who decided this case — the appellate panel, or the single
district or bankruptcy judge who issued the opinion.

Include:
  - judges named on a "Before:" or "Present:" panel line
  - the judge who signed the opinion, including a signature block at the end
    ("/s/ Susan O. Hickey", "JOHN H. ENGLAND, III", a name above or below
    "United States District Judge" / "Magistrate Judge" / "Circuit Judge")
  - judges who concurred or dissented in this case
  - the district judge named in an appellate opinion's docket line ONLY IF no
    appellate panel is listed

Exclude:
  - attorneys, counsel, and the parties (a caption listing bar numbers and
    email addresses is counsel, not judges)
  - the clerk of court
  - a magistrate judge whose report is merely under review, unless no other
    judge is named
  - judges named only as authors of *cited* opinions
  - judges who merely voted in an en banc poll. If the document is an ORDER
    on a petition for rehearing en banc, most of the names in it are the full
    court recording a vote, and they did NOT decide this case. List only the
    original three-judge merits panel plus any judge who wrote a separate
    opinion (concurring in or dissenting from the denial). A U.S. court of
    appeals decides cases in panels of three: more than three "joined" judges
    means you have almost certainly swept in an en banc poll.

Give names exactly as printed, even if the text looks garbled by OCR — do not
correct or complete a name. If a first or middle name is not printed, return an
empty string for it; do not infer one. If no judge is named anywhere in the
text, return an empty list rather than guessing. If the opinion is per curiam,
set per_curiam to true and still list any panel members named.

OPINION TEXT
------------
{caption}
"""


def load_caption_rows(
    db_path: Path, case_ids: Iterable[Any] | None = None
) -> list[dict[str, Any]]:
    """``case_id``, ``opinion_id``, ``case_name`` and the caption window."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = """SELECT id, courtlistener_opinion_id, case_name, court_name, court_level, decision_date, opinion_text_clean, opinion_paragraphs_clean_json FROM cases"""
        params: tuple[Any, ...] = ()
        if case_ids is not None:
            ids = [str(c) for c in case_ids]
            if not ids:
                return []
            sql += f" WHERE id IN ({','.join('?' * len(ids))})"
            params = tuple(ids)
        rows = conn.execute(sql + " ORDER BY id", params).fetchall()
    finally:
        conn.close()

    out = []
    for row in rows:
        try:
            paragraphs = json.loads(
                row["opinion_paragraphs_clean_json"] or "[]"
            )
        except json.JSONDecodeError:
            paragraphs = []
        if not isinstance(paragraphs, list):
            paragraphs = []
        out.append(
            {
                "case_id": str(row["id"]),
                "opinion_id": str(row["courtlistener_opinion_id"] or "")
                or None,
                "case_name": row["case_name"],
                "court_name": row["court_name"],
                "court_level": row["court_level"],
                "decision_date": row["decision_date"],
                "caption": extraction_window(
                    paragraphs, row["opinion_text_clean"]
                ),
            }
        )
    return out


# ── extraction ────────────────────────────────────────────────────────────────


def _coerce_payload(raw: Any) -> dict[str, Any] | None:
    """Normalize a model response into ``{per_curiam, judges}`` or None."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None

    judges = []
    for item in raw.get("judges") or []:
        if not isinstance(item, dict):
            continue
        last = str(item.get("last_name") or "").strip()
        if not last:
            continue
        role = str(item.get("role") or "unknown").strip().lower()
        judges.append(
            {
                "last_name": last,
                "first_name": str(item.get("first_name") or "").strip(),
                "middle_name": str(item.get("middle_name") or "").strip(),
                "role": (
                    role
                    if role
                    in {
                        "authored",
                        "joined",
                        "concurring",
                        "dissenting",
                        "unknown",
                    }
                    else "unknown"
                ),
            }
        )
    return {"per_curiam": bool(raw.get("per_curiam")), "judges": judges}


def extract_doctrine(
    doctrine: str, *, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """Extract panels for the cases not yet in the doctrine's cache.

    Temperature is 1: ``batch_query_llm`` always sends it, and the GPT-5
    family accepts nothing else, so a rerun can differ. The cache file is the
    record of what was used.
    """
    rows = load_caption_rows(clean_db_path(doctrine))
    payload = load_cache(doctrine)
    done = payload["cases"]
    pending = [
        r for r in rows if r["case_id"] not in done and r["caption"].strip()
    ]
    print(
        f"""{doctrine}: {len(rows)} cases, {len(done)} cached, {len(pending)} to extract -> {model}"""
    )
    if not pending:
        return payload

    import api_calls

    items = [
        {
            "content": USER_TEMPLATE.format(caption=r["caption"]),
            "system_prompt": SYSTEM_PROMPT,
        }
        for r in pending
    ]
    responses = api_calls.batch_query_llm(
        items,
        model=model,
        temperature=1.0,
        response_format=JUDGE_SCHEMA,
        custom_id_prefix=f"judges-{doctrine}",
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_malformed = 0
    for row, (_, raw) in zip(pending, responses):
        parsed = _coerce_payload(raw)
        if parsed is None:
            n_malformed += 1
            continue
        payload["cases"][row["case_id"]] = {
            **parsed,
            "source": SOURCE_LLM,
            "opinion_id": row["opinion_id"],
            "case_name": row["case_name"],
            "court_level": row["court_level"],
            "model": model,
            "extracted_at": now,
        }
    payload["meta"]["model"] = model
    save_cache(doctrine, payload)
    print(f"  {n_malformed} malformed responses skipped")
    return payload


# ── FJC fill for unmatched authors ────────────────────────────────────────────

SOURCE_FJC_UNIQUE = "fjc_unique"
FJC_FILL_SURNAME_ALONE = 0.8
FJC_FILL_WITH_FIRST = 0.5
FJC_FILL_MARGIN = 0.2


def _name_ratio(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def unique_on_court_candidate(
    judge: dict[str, Any],
    on_court: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """The one FJC judge on this court on this date that this name can be.
    """
    last = surname_key(judge.get("last_name"))
    first = (
        norm_name(judge.get("first_name")).split(" ")[0]
        if judge.get("first_name")
        else ""
    )
    # Split on any non-letter before normalizing, so "JAMES/PONATO" and
    # "HERNANDEZ’*COVINGTON" come apart instead of fusing into one token.
    tokens = [
        norm_name(t)
        for t in re.split(r"[^\w']+|_|\d", judge.get("last_name") or "")
    ]
    tokens = [t for t in tokens if len(t) >= 2]
    if not on_court:
        return None, "no_fjc_judge_on_court"

    def first_ok(given: str, fjc_first: str) -> bool:
        # Equal, or one a 3+ letter prefix of the other ("Cin" / "Cindy").
        f = fjc_first.split(" ")[0]
        return len(given) >= 3 and (f.startswith(given) or given.startswith(f))

    def surname_ok(token: str, a: dict[str, Any], floor: float) -> bool:
        parts = a["surname"].split(" ")
        return _name_ratio(token, a["surname"]) >= floor or any(
            _name_ratio(token, p) >= floor for p in parts
        )

    # 1. A full name in the last-name field ("JILL PRYOR", "JAMES/PONATO"):
    #    one token is the first name, another is close to the surname.
    if len(tokens) >= 2:
        hits = [
            a
            for a in on_court
            if any(first_ok(t, a["first"]) for t in tokens)
            and any(
                surname_ok(t, a, FJC_FILL_SURNAME_ALONE)
                for t in tokens
                if not first_ok(t, a["first"])
            )
        ]
        if len(hits) == 1:
            return hits[0], "full_name_in_field"

    # 2. First name agrees, surname is a plausible OCR rendering.
    if first:
        cands = list(tokens) or [last]
        hits = [
            a
            for a in on_court
            if first_ok(first, a["first"])
            and any(
                surname_ok(t, a, FJC_FILL_WITH_FIRST)
                or (len(t) >= 2 and a["surname"].startswith(t))
                for t in cands + [last]
            )
        ]
        if len(hits) == 1:
            return hits[0], "first_and_surname"
        if len(hits) > 1:
            return None, "several_fit"

    # 3. Surname alone, with a clear margin.
    scored = sorted(
        ((_name_ratio(last, a["surname"]), a) for a in on_court),
        key=lambda x: -x[0],
    )
    if scored and scored[0][0] >= FJC_FILL_SURNAME_ALONE:
        best_score, best = scored[0]
        runner = next(
            (r for r, a in scored[1:] if a["nid"] != best["nid"]), 0.0
        )
        initial = first[:1]
        if best_score - runner >= FJC_FILL_MARGIN and (
            not initial or best["first_initial"] == initial
        ):
            return best, "surname_clear"
    return None, "no_unique_candidate"


def fill_unique_on_court(doctrine: str) -> list[dict[str, Any]]:
    """Fill unmatched Article III authors whose court had one judge they can be.
    """
    payload = load_cache(doctrine)
    clean = clean_db_path(doctrine)
    conn = sqlite3.connect(f"file:{clean}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta = {
            str(r["id"]): dict(r)
            for r in conn.execute(
                "SELECT id, court_name, court_level, circuit, decision_date FROM cases"
            )
        }
    finally:
        conn.close()

    directory = FjcDirectory.load()
    by_court: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in directory.appointments:
        by_court[a["court_key"]].append(a)

    report: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    touched = False
    for case_id, entry in payload.get("cases", {}).items():
        if entry.get("source") == SOURCE_MANUAL:
            continue
        m = meta.get(case_id)
        if (
            not m
            or (entry.get("court_level") or m["court_level"])
            not in ARTICLE_III_LEVELS
        ):
            continue
        date = m["decision_date"]
        on_court = [
            a
            for a in by_court.get(norm_court(m["court_name"]), [])
            if directory._in_window(a, date)
        ]
        for i in author_indices(entry):
            judge = entry["judges"][i]
            found, _ = directory.match(
                judge,
                court_name=m["court_name"],
                circuit=m["circuit"],
                date=date,
            )
            if found:
                continue  # judge_party already resolves this author
            cand, rule = unique_on_court_candidate(judge, on_court)
            row = {
                "doctrine": doctrine,
                "case_id": case_id,
                "case_name": entry.get("case_name") or "",
                "court_name": m["court_name"],
                "decision_date": date,
                "extracted": f"{judge.get('last_name', '')}, {judge.get('first_name', '')}".strip(
                    ", "
                ),
                "rule": rule,
                "filled": bool(cand),
                "fjc_name": cand["full_name"] if cand else "",
                "party": cand["party"] if cand else "",
            }
            report.append(row)
            if not cand:
                continue
            judge.setdefault(
                "extracted_name",
                {
                    k: judge.get(k, "")
                    for k in ("last_name", "first_name", "middle_name")
                },
            )
            # Raw FJC spelling, not the normalized key, so the record reads
            # naturally and surname_key() reproduces the key on the next run.
            fjc_last, fjc_first, fjc_middle = _fjc_raw_name(cand)
            judge.update(
                {
                    "last_name": fjc_last,
                    "first_name": fjc_first,
                    "middle_name": fjc_middle,
                }
            )
            judge["fjc_fill"] = {
                "nid": cand["nid"],
                "rule": rule,
                "filled_at": now,
            }
            entry["source"] = SOURCE_FJC_UNIQUE
            touched = True
    if touched:
        save_cache(doctrine, payload)
    return report


_FJC_RAW: dict[str, tuple[str, str, str]] | None = None


def _fjc_raw_name(appointment: dict[str, Any]) -> tuple[str, str, str]:
    """(Last, First, Middle) exactly as the FJC CSV prints them."""
    global _FJC_RAW
    if _FJC_RAW is None:
        csv.field_size_limit(10_000_000)
        with FJC_CSV.open(newline="", encoding="utf-8-sig") as f:
            _FJC_RAW = {
                (r.get("nid") or "").strip(): (
                    (r.get("Last Name") or "").strip(),
                    (r.get("First Name") or "").strip(),
                    (r.get("Middle Name") or "").strip(),
                )
                for r in csv.DictReader(f)
            }
    return _FJC_RAW[appointment["nid"]]


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Extract deciding judges from opinion captions."
    )
    p.add_argument("--doctrine", nargs="+", required=True)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Any model query_llm routes: 'gpt*' -> OpenAI, 'gemini*' -> Vertex.",
    )
    p.add_argument(
        "--fill-from-fjc",
        action="store_true",
        help="""Fill Article III authors judge_party cannot match when exactly one FJC judge on that court on that date fits the name.""",
    )
    args = p.parse_args(argv)

    for doctrine in args.doctrine:
        if not args.fill_from_fjc:
            extract_doctrine(doctrine, model=args.model)
            continue
        rows = fill_unique_on_court(doctrine)
        for r in sorted(
            rows, key=lambda r: (not r["filled"], int(r["case_id"]))
        ):
            mark = (
                f"-> {r['fjc_name']} ({r['party']})"
                if r["filled"]
                else "   left"
            )
            print(
                f"""{doctrine:<13}{r['case_id']:>5}  {r['extracted'][:28]:<28} {mark:<42} {r['rule']}"""
            )
        print(
            f"""filled {sum(r['filled'] for r in rows)} of {len(rows)} unmatched authors"""
        )


if __name__ == "__main__":
    main()
