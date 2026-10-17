"""Match extracted judges to the FJC directory and code each case's author party.

``judge_author_party`` is the appointing party of the opinion's authors: every
signer, else the only judge, else every judge who joined a per curiam (or an
unsigned panel larger than ``LARGE_PANEL``). One shared party is that party;
different parties are ``mixed``. State and Article I courts are
``non_article_iii``; a case with an unidentified or unmatched author is
``unknown``.

    python -m descriptive_findings.judge_party --doctrine bruen --write-db
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
import unicodedata

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from descriptive_findings.judge_cache import author_indices  # noqa: E402
from descriptive_findings.judge_cache import author_mode  # noqa: E402
from descriptive_findings.judge_cache import clean_db_path  # noqa: E402
from descriptive_findings.judge_cache import load_cache  # noqa: E402

FJC_CSV = Path(__file__).resolve().parent / "judges.csv"
PARTY_CODE = {
    "republican": "R",
    "democratic": "D",
}
FAR_FUTURE = "9999-12-31"


# ── court normalization ───────────────────────────────────────────────────────


def norm_court(name: str | None) -> str:
    """Fold CourtListener and FJC spellings of a court onto one key.
    """
    s = (name or "").lower()
    s = s.replace("united states", "u.s.")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


_ORDINAL_CIRCUIT = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "district of columbia": "dc",
    "federal": "fed",
}


def circuit_of_fjc_court(name: str) -> str | None:
    """The circuit a given FJC court sits in, or None if it has none.
    """
    from courtlistener.courts import _STATE_NAMES  # noqa: PLC0415
    from courtlistener.courts import STATE_CIRCUIT

    low = norm_court(name)
    m = re.search(r"court of appeals for the (.+?) circuit$", low)
    if m:
        return _ORDINAL_CIRCUIT.get(m.group(1).strip())
    if "district court" in low or "circuit court" in low:
        if "district of columbia" in low:
            return "dc"
        # Longest state name wins: "West Virginia" must beat "Virginia".
        best: tuple[int, str | None] = (0, None)
        for usps, state in _STATE_NAMES.items():
            if state.lower() in low and len(state) > best[0]:
                best = (len(state), usps)
        if best[1]:
            return STATE_CIRCUIT.get(best[1])
    return None


# ── FJC directory ─────────────────────────────────────────────────────────────


def norm_name(value: str | None) -> str:
    """Casefold a name for comparison; keeps internal hyphens and
    apostrophes."""
    # Fold accents before stripping non-letters: FJC prints "Nelson Stephen
    # Román", and dropping the "á" outright keyed him as "romn", so every
    # "ROMAN" in a caption went unmatched.
    s = unicodedata.normalize("NFKD", value or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    s = re.sub(r"[^a-z'\- ]+", "", s)
    return " ".join(s.split())


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def surname_key(value: str | None) -> str:
    """The comparable part of a surname.
    """
    s = norm_name(value)
    tokens = [t for t in s.replace(".", "").split() if t]
    while tokens and tokens[-1].strip(".") in _SUFFIXES:
        tokens.pop()
    if not tokens:
        return ""
    # Keep nobiliary particles attached: "van dyke", "de la cruz".
    particles = {
        "van",
        "von",
        "de",
        "del",
        "della",
        "di",
        "da",
        "la",
        "le",
        "du",
        "st",
    }
    start = len(tokens) - 1
    while start > 0 and tokens[start - 1] in particles:
        start -= 1
    return " ".join(tokens[start:])


def _iso(value: str | None) -> str | None:
    """FJC dates are already ISO; blanks and junk become None."""
    s = (value or "").strip()
    return s if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else None


class FjcDirectory:
    """FJC appointments, indexed for (court, surname, date) lookup."""

    def __init__(self, appointments: list[dict[str, Any]]):
        self.appointments = appointments
        self.by_court_surname: dict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        self.by_circuit_surname: dict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        self.by_surname: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.surnames: set[str] = set()
        for a in appointments:
            self.by_court_surname[(a["court_key"], a["surname"])].append(a)
            if a["circuit"]:
                self.by_circuit_surname[(a["circuit"], a["surname"])].append(a)
            self.by_surname[a["surname"]].append(a)
            self.surnames.add(a["surname"])

    @classmethod
    def load(cls, path: Path | None = None) -> "FjcDirectory":
        """Read the FJC CSV export (``FJC_CSV`` by default), one appointment
        per (judge, court seat)."""
        csv_path = path or FJC_CSV
        if not csv_path.exists():
            raise FileNotFoundError(
                f"""FJC directory not found at {csv_path}. Download the Biographical Directory of Article III Federal Judges export from fjc.gov and save it there."""
            )
        # The professional-career fields run long; the default field limit
        # raises on some rows.
        csv.field_size_limit(10_000_000)

        appointments: list[dict[str, Any]] = []
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                surname = surname_key(row.get("Last Name"))
                if not surname:
                    continue
                first = norm_name(row.get("First Name"))
                for i in range(1, 7):
                    court = (row.get(f"Court Name ({i})") or "").strip()
                    if not court:
                        continue
                    party_raw = (
                        row.get(f"Party of Appointing President ({i})") or ""
                    ).strip()
                    party = PARTY_CODE.get(
                        party_raw.lower(), "O" if party_raw else None
                    )
                    court_key = norm_court(court)
                    appointments.append(
                        {
                            "nid": (row.get("nid") or "").strip(),
                            "surname": surname,
                            "first": first,
                            "first_initial": first[:1],
                            "middle": norm_name(row.get("Middle Name")),
                            "full_name": " ".join(
                                p
                                for p in (
                                    (row.get("First Name") or "").strip(),
                                    (row.get("Middle Name") or "").strip(),
                                    (row.get("Last Name") or "").strip(),
                                    (row.get("Suffix") or "").strip(),
                                )
                                if p
                            ),
                            "party": party,
                            "party_raw": party_raw,
                            "appointing_president": (
                                row.get(f"Appointing President ({i})") or ""
                            ).strip(),
                            "court_name": court,
                            "court_key": court_key,
                            "court_type": (
                                row.get(f"Court Type ({i})") or ""
                            ).strip(),
                            "circuit": circuit_of_fjc_court(court),
                            # Commission date is when service actually begins;
                            # confirmation and nomination both precede it.  A
                            # recess appointee has neither, so fall back in
                            # that order.
                            "start": (
                                _iso(row.get(f"Commission Date ({i})"))
                                or _iso(
                                    row.get(f"Recess Appointment Date ({i})")
                                )
                                or _iso(row.get(f"Confirmation Date ({i})"))
                            ),
                            "end": _iso(row.get(f"Termination Date ({i})")),
                        }
                    )
        return cls(appointments)

    @staticmethod
    def _in_window(appointment: dict[str, Any], date: str | None) -> bool:
        """Was this judge on this court on this date?
        """
        if not date:
            return True
        start = appointment["start"] or ""
        end = appointment["end"] or FAR_FUTURE
        # Senior status keeps a judge deciding cases, and FJC records
        # termination only when they leave the bench entirely, so `end` is the
        # right bound.
        return (not start or start <= date) and date <= end

    def match(
        self,
        judge: dict[str, Any],
        *,
        court_name: str | None,
        circuit: str | None,
        date: str | None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve one extracted judge. Returns (appointment, match_method)."""
        surname = surname_key(judge.get("last_name"))
        if not surname:
            return None, "unmatched"
        first_initial = norm_name(judge.get("first_name"))[:1]

        court_key = norm_court(court_name)
        for bucket, method in (
            (self.by_court_surname.get((court_key, surname), []), "exact"),
            (
                self.by_circuit_surname.get((circuit or "", surname), []),
                "circuit",
            ),
            (self.by_surname.get(surname, []), "national"),
        ):
            live = [a for a in bucket if self._in_window(a, date)]
            if not live:
                continue
            if len(live) == 1:
                return live[0], method
            if first_initial:
                narrowed = [
                    a for a in live if a["first_initial"] == first_initial
                ]
                if len(narrowed) == 1:
                    return narrowed[0], (
                        "initial" if method == "exact" else method
                    )
                if narrowed:
                    live = narrowed
            # Still ambiguous. If every candidate shares a party the answer is
            # the same either way, so take it and say so.
            parties = {a["party"] for a in live}
            if len(parties) == 1:
                return live[0], f"{method}_ambiguous_same_party"
            return None, f"{method}_ambiguous"

        if surname not in self.surnames:
            return None, "unmatched_not_in_fjc"
        return None, "unmatched_wrong_court_or_date"


# ── panel resolution ──────────────────────────────────────────────────────────

_ARTICLE_III_LEVELS = frozenset(
    {"scotus", "federal_appellate", "federal_district"}
)


def _is_article_iii(court_name: str | None, court_level: str | None) -> bool:
    """Does this court's bench have an appointing president at all?
    """
    level = (court_level or "").strip()
    if level in _ARTICLE_III_LEVELS:
        return True
    if level:
        return False
    # No level recorded: fall back to the name.
    low = norm_court(court_name)
    return (
        bool(
            re.search(
                r"u s (court of appeals|district court|supreme court)", low
            )
        )
        and "bankruptcy" not in low
    )


def _author_party(
    resolved: list[dict[str, Any]],
    *,
    per_curiam: bool,
    in_scope: bool,
) -> tuple[str, str, list[str]]:
    """The case's author party, how the authors were chosen, and their FJC ids.
    """
    if not in_scope:
        return "non_article_iii", "non_article_iii", []
    if not resolved:
        return "unknown", "unknown", []

    mode = author_mode(resolved, per_curiam)
    indices = author_indices({"judges": resolved, "per_curiam": per_curiam})
    authors = [resolved[i] for i in indices]
    nids = [str(a["nid"]) for a in authors if a["nid"]]
    if not authors:
        return "unknown", mode, []
    if mode == "per_curiam" and any(a["role"] == "unknown" for a in authors):
        return "unknown", mode, nids
    if any(not a["party"] for a in authors):
        return "unknown", mode, nids
    parties = {a["party"] for a in authors}
    if len(parties) == 1:
        return next(iter(parties)), mode, nids
    return "mixed", mode, nids


def resolve_panel(
    judges: list[dict[str, Any]],
    *,
    directory: FjcDirectory,
    court_name: str | None,
    court_level: str | None,
    circuit: str | None,
    date: str | None,
    per_curiam: bool = False,
) -> dict[str, Any]:
    """Attach a party to every extracted judge and code the author party."""
    in_scope = _is_article_iii(court_name, court_level)
    resolved: list[dict[str, Any]] = []

    for judge in judges:
        if not in_scope:
            appointment, method = None, "court_outside_fjc"
        else:
            appointment, method = directory.match(
                judge, court_name=court_name, circuit=circuit, date=date
            )
        resolved.append(
            {
                "last_name": judge.get("last_name", ""),
                "first_name": judge.get("first_name", ""),
                "role": judge.get("role", "unknown"),
                "nid": appointment["nid"] if appointment else None,
                "fjc_name": appointment["full_name"] if appointment else None,
                "party": appointment["party"] if appointment else None,
                "appointing_president": (
                    appointment["appointing_president"] if appointment else None
                ),
                "match_method": method,
            }
        )

    author, mode, nids = _author_party(
        resolved, per_curiam=per_curiam, in_scope=in_scope
    )
    return {
        "judges": resolved,
        "judge_author_party": author,
        "author_mode": mode,
        "author_nids": nids,
        "n_judges": len(resolved),
        "n_matched": sum(1 for j in resolved if j["party"]),
        "in_scope": in_scope,
    }


# ── corpus pass ───────────────────────────────────────────────────────────────


def _case_court_rows(db_path: Path) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {
            str(r["id"]): {
                "case_id": str(r["id"]),
                "opinion_id": str(r["courtlistener_opinion_id"] or "") or None,
                "case_name": r["case_name"],
                "court_name": r["court_name"],
                "court_level": r["court_level"],
                "circuit": r["circuit"],
                "decision_date": r["decision_date"],
            }
            for r in conn.execute(
                """SELECT id, courtlistener_opinion_id, case_name, court_name, court_level, circuit, decision_date FROM cases"""
            )
        }
    finally:
        conn.close()


def resolve_doctrine(
    doctrine: str, *, directory: FjcDirectory
) -> list[dict[str, Any]]:
    """One row per case, with the resolved panel attached."""
    panels = load_cache(doctrine)["cases"]
    out: list[dict[str, Any]] = []
    for case_id, meta in _case_court_rows(clean_db_path(doctrine)).items():
        entry = panels[case_id] if case_id in panels else None
        resolved = resolve_panel(
            entry["judges"] if entry else [],
            directory=directory,
            court_name=meta["court_name"],
            court_level=meta["court_level"],
            circuit=meta["circuit"],
            date=meta["decision_date"],
            per_curiam=bool(entry["per_curiam"]) if entry else False,
        )
        out.append({**meta, "doctrine": doctrine, **resolved})
    return out


def write_db_columns(doctrine: str, rows: list[dict[str, Any]]) -> int:
    """Write ``judge_author_party`` to ``citing_cases_clean.db``, adding the
    column if absent. ``case_index.json`` carries it into each run."""
    conn = sqlite3.connect(clean_db_path(doctrine))
    try:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(cases)")}
        if "judge_author_party" not in existing:
            conn.execute("ALTER TABLE cases ADD COLUMN judge_author_party TEXT")
        conn.executemany(
            "UPDATE cases SET judge_author_party = ? WHERE id = ?",
            [(r["judge_author_party"], r["case_id"]) for r in rows],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="""Match extracted judges to the FJC directory and code the author party."""
    )
    p.add_argument("--doctrine", nargs="+", required=True)
    p.add_argument(
        "--write-db",
        action="store_true",
        help="Write judge_author_party into citing_cases_clean.db.",
    )
    args = p.parse_args(argv)

    directory = FjcDirectory.load()
    for doctrine in args.doctrine:
        rows = resolve_doctrine(doctrine, directory=directory)
        values = Counter(r["judge_author_party"] for r in rows)
        print(f"{doctrine:<16} {len(rows)} cases  {dict(values.most_common())}")
        if args.write_db:
            n = write_db_columns(doctrine, rows)
            print(f"  wrote {n} rows to {clean_db_path(doctrine).name}")


if __name__ == "__main__":
    main()
