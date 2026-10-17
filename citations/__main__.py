"""Builds the citation graph the UI shows under the Tree tab.

Fetches each corpus case's citations from CourtListener and writes
``citation_graph.json`` into the run directory, beside ``corpus_level.json``.
Set COURTLISTENER_API_KEY in `.env`. Run from the repository root.

    python -m citations --doctrine casa --run-id <run>
"""

import argparse
import json
import os
from pathlib import Path

from auto_treatise import doctrine as doctrine_paths
from citations import edges
from citations import payload
from courtlistener import fetch_courtlistener as cl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doctrine", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--test",
        action="store_true",
        help="use the run under analysis/test/",
    )
    args = parser.parse_args()

    run_dir = (
        doctrine_paths.analysis_root(args.doctrine, args.test)
        / "runs"
        / args.run_id
    )
    cases = payload.load_cases(doctrine_paths.db_path(args.doctrine))
    client = cl.CLClient(os.environ["COURTLISTENER_API_KEY"])
    citations = edges.fetch_edges(
        client, [case["opinion_id"] for case in cases]
    )
    graph = payload.build_payload(
        cases, citations, payload.load_tree(run_dir)
    )
    out = Path(run_dir) / "citation_graph.json"
    out.write_text(json.dumps(graph, separators=(",", ":")))
    print(
        f"Wrote {out}: {len(graph['nodes'])} cases, "
        f"{len(graph['links'])} citations"
    )


if __name__ == "__main__":
    main()
