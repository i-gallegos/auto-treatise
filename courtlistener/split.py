"""Temporal train / dev / test split of a doctrine's corpus.

The corpus is first collapsed to one row per decision, then optionally
restricted to cases whose ``citation_depth`` is at least ``--min-citation-depth``
(read from the raw fetch DB ``citing_cases.db``). The cutoff ``tau`` is the
decision date at position ``floor(q * n)`` of the sorted dates. Train is every
case dated before ``tau``. The later cases are shuffled within each court tier
(strata from ``AUTHORITY_RANK``) and dealt alternately into dev and test.

    python -m courtlistener.split -d loper_bright -q 0.7
    python -m courtlistener.split -d bruen -q 0.7 --min-citation-depth 2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import math
from pathlib import Path
import random
import sqlite3

from courtlistener.courts import AUTHORITY_RANK
from courtlistener.courts import COURT_LEVELS
from courtlistener.dedup_opinions import surviving_case_ids

ROOT = Path(__file__).resolve().parent.parent
TIER_ORDER = ("l3", "l2", "l1", "unranked")
TIERS = {
    level: ("l3" if rank >= 3 else "l2" if rank == 2 else "l1")
    for level, rank in AUTHORITY_RANK.items()
}
TIERS.update(
    {level: "unranked" for level in COURT_LEVELS if level not in TIERS}
)


def load_cases(db_path: Path) -> list[dict]:
    """Cases with a decision date, sorted by (date, id)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT id, courtlistener_opinion_id, decision_date, court_level FROM cases"""
    ).fetchall()
    conn.close()
    return [
        {
            "case_id": str(case_id),
            "opinion_id": str(opinion_id),
            "date": date.strip() if date else "",
            "tier": TIERS[level] if level in TIERS else "unranked",
        }
        for case_id, opinion_id, date, level in rows
    ]


def read_citation_depths(raw_db: Path) -> dict[str, int]:
    """Deepest ``citation_depth`` per opinion id in the raw fetch DB."""
    conn = sqlite3.connect(raw_db)
    depths: dict[str, int] = {}
    for opinion_id, depth in conn.execute(
        """SELECT courtlistener_opinion_id, citation_depth FROM cases WHERE courtlistener_opinion_id IS NOT NULL AND citation_depth IS NOT NULL"""
    ):
        key = str(opinion_id)
        depths[key] = max(depths[key], depth) if key in depths else depth
    conn.close()
    return depths


def make_split(
    cases: list[dict], q: float, seed: int
) -> tuple[str, list, list, list, list]:
    """``(tau, train, dev, test, undated)`` case-id lists."""
    undated = [c["case_id"] for c in cases if not c["date"]]
    dated = sorted(
        (c for c in cases if c["date"]),
        key=lambda c: (c["date"], int(c["case_id"])),
    )
    tau = dated[int(math.floor(q * len(dated)))]["date"]
    train = [c["case_id"] for c in dated if c["date"] < tau]
    heldout = [c for c in dated if c["date"] >= tau]

    rng = random.Random(seed)
    halves: tuple[list, list] = ([], [])
    dealt = 0
    for tier in TIER_ORDER:
        members = [c for c in heldout if c["tier"] == tier]
        rng.shuffle(members)
        for case in members:
            halves[dealt % 2].append(case)
            dealt += 1
    dev, test = (
        [
            c["case_id"]
            for c in sorted(half, key=lambda c: (c["date"], int(c["case_id"])))
        ]
        for half in halves
    )
    return tau, train, dev, test, undated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doctrine", "-d", required=True)
    parser.add_argument("--quantile", "-q", type=float, required=True)
    parser.add_argument("--min-citation-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=None, help="default: data/split_q70.json"
    )
    args = parser.parse_args()

    data_dir = ROOT / "doctrines" / args.doctrine / "data"
    clean_db = data_dir / "citing_cases_clean.db"
    keep = surviving_case_ids(clean_db)
    cases = [c for c in load_cases(clean_db) if c["case_id"] in keep]
    if args.min_citation_depth is not None:
        depths = read_citation_depths(data_dir / "citing_cases.db")
        cases = [
            c
            for c in cases
            if c["opinion_id"] in depths
            and depths[c["opinion_id"]] >= args.min_citation_depth
        ]

    tau, train, dev, test, undated = make_split(cases, args.quantile, args.seed)
    out = (
        Path(args.out)
        if args.out
        else data_dir / f"split_q{round(args.quantile * 100)}.json"
    )
    out.write_text(
        json.dumps(
            {
                "doctrine": args.doctrine,
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "split": {
                    "tau": tau,
                    "requested_quantile": args.quantile,
                    "seed": args.seed,
                    "min_citation_depth": args.min_citation_depth,
                },
                "counts": {
                    "train": len(train),
                    "dev": len(dev),
                    "test": len(test),
                    "undated": len(undated),
                },
                "train": train,
                "dev": dev,
                "test": test,
                "undated": undated,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"""{args.doctrine}: tau={tau} train={len(train)} dev={len(dev)} test={len(test)} undated={len(undated)} -> {out}"""
    )


if __name__ == "__main__":
    main()
