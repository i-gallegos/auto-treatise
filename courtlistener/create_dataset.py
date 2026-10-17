"""Fetch a doctrine's citing opinions from CourtListener and clean them (§3.2).

Writes ``data/citing_cases.db`` (``fetch_courtlistener``) and
``data/citing_cases_clean.db`` (``clean_courtlistener``) under
``doctrines/<doctrine>/``.

    python -m courtlistener.create_dataset --doctrine casa --cluster-id 10618550
"""

import argparse
from pathlib import Path

from courtlistener.clean_courtlistener import MIN_CHARS
from courtlistener.clean_courtlistener import clean_to_output_db
from courtlistener.fetch_courtlistener import fetch

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("citation", nargs="?", help="e.g. '585 U.S. 296'")
    parser.add_argument("--cluster-id", type=int)
    parser.add_argument("--opinion-id", type=int)
    parser.add_argument("--doctrine", required=True)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument(
        "--force", action="store_true", help="overwrite the cleaned DB"
    )
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = parser.parse_args()

    data_dir = ROOT / "doctrines" / args.doctrine / "data"
    raw_db = data_dir / "citing_cases.db"
    clean_db = data_dir / "citing_cases_clean.db"
    fetch(
        raw_db, args.citation, args.cluster_id, args.opinion_id, args.min_depth
    )
    clean_to_output_db(raw_db, clean_db, args.force, args.min_chars)


if __name__ == "__main__":
    main()
