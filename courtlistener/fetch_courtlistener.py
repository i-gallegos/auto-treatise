"""Fetch every opinion that cites a seed case from CourtListener.

The seed is resolved to its CourtListener sub-opinion ids. Every citing opinion
is paged from ``/opinions-cited/`` (recording how many times it cites the seed,
its ``citation_depth``), and each one's text and cluster metadata are then
fetched into a SQLite ``cases`` table. Opinions already fetched are skipped.

    python -m courtlistener.fetch_courtlistener "597 U.S. 1" \\
        --db doctrines/bruen/data/citing_cases.db

Set COURTLISTENER_API_KEY in `.env` . The seed can instead be given as ``--cluster-id`` or
``--opinion-id``.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests

from courtlistener import courts

load_dotenv()

BASE_URL = "https://www.courtlistener.com/api/rest/v4"
PAGE_SIZE = 100
ANON_RPM = 4
AUTH_RPM = 50

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    courtlistener_opinion_id    INTEGER UNIQUE,
    courtlistener_cluster_id    INTEGER,
    courtlistener_url           TEXT,
    citation_depth              INTEGER,
    case_name                   TEXT,
    decision_date               TEXT,
    court_id                    TEXT,
    court_name                  TEXT,
    court_abbreviation          TEXT,
    court_level                 TEXT,
    jurisdiction                TEXT,
    circuit                     TEXT,
    courtlistener_docket_id     INTEGER,
    docket_number               TEXT,
    judges                      TEXT,
    precedential_status         TEXT,
    citation_count              INTEGER,
    citation                    TEXT,
    opinion_text                TEXT,
    opinion_html                TEXT,
    text_source                 TEXT,
    fetch_status                TEXT DEFAULT 'discovered',
    date_fetched                TEXT
);
"""


class CLClient:
    """Rate-limited CourtListener API client."""

    def __init__(self, token=None):
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Token {token}"
        self.min_interval = 60.0 / (AUTH_RPM if token else ANON_RPM)
        self.last = 0.0

    def request(self, method, url, **kwargs):
        """JSON response, or None on a 404. Sleeps and retries on a 429."""
        while True:
            wait = self.min_interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            r = self.session.request(method, url, timeout=30, **kwargs)
            if r.status_code == 429:
                time.sleep(int(r.headers["Retry-After"]))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    def get(self, url, params=None):
        return self.request("GET", url, params=params)

    def post(self, url, data):
        return self.request("POST", url, data=data)


def url_id(url, resource):
    """Numeric id from a resource URL, e.g. ``.../opinions/123/`` -> 123."""
    return int(re.search(rf"/{resource}/(\d+)/", url).group(1))


def court_slug(value):
    """Court slug from a court hyperlink such as ``.../courts/ca9/``."""
    return re.search(r"/courts/([^/?]+)/?", value).group(1) if value else ""


def resolve_seed(client, citation=None, opinion_id=None, cluster_id=None):
    """``(sub-opinion ids, cluster id)`` of the seed case."""
    if cluster_id is None and opinion_id is not None:
        cluster_id = url_id(
            client.get(f"{BASE_URL}/opinions/{opinion_id}/")["cluster"],
            "clusters",
        )
    if cluster_id is None:
        matches = client.post(
            f"{BASE_URL}/citation-lookup/", data={"text": citation}
        )
        cluster_id = next(
            m["clusters"][0]["id"]
            for m in matches
            if m["status"] == 200 and m["clusters"]
        )
    cluster = client.get(f"{BASE_URL}/clusters/{cluster_id}/")
    print(
        f"""Seed: {cluster['case_name']} ({cluster['date_filed']}), cluster {cluster_id}"""
    )
    return [url_id(u, "opinions") for u in cluster["sub_opinions"]], cluster_id


def discover(client, conn, seed_opinion_ids, min_depth):
    """Insert every opinion citing any seed sub-opinion, keeping the largest
    ``citation_depth`` when it cites several."""
    for seed_id in seed_opinion_ids:
        url = f"{BASE_URL}/opinions-cited/"
        params = {"cited_opinion": seed_id, "page_size": PAGE_SIZE}
        while url:
            page = client.get(url, params)
            params = None
            for record in page["results"]:
                if record["depth"] < min_depth:
                    continue
                citing_id = url_id(record["citing_opinion"], "opinions")
                conn.execute(
                    """INSERT INTO cases (courtlistener_opinion_id, courtlistener_url, citation_depth) VALUES (?, ?, ?) ON CONFLICT (courtlistener_opinion_id) DO UPDATE SET citation_depth = MAX(citation_depth, excluded.citation_depth)""",
                    (
                        citing_id,
                        f"https://www.courtlistener.com/opinion/{citing_id}/",
                        record["depth"],
                    ),
                )
            conn.commit()
            url = page["next"]


def opinion_text(data):
    """``(plain text, html)`` of an ``/opinions/<id>/`` response."""
    html = (
        data["html_with_citations"]
        or data["html"]
        or data["html_lawbox"]
        or data["html_columbia"]
        or data["html_anon_2020"]
    )
    text = data["plain_text"]
    if not text and html:
        text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    return text, html


def cluster_metadata(client, cluster_id):
    """Case, docket and normalized court fields of a cluster. The court is
    read from the docket, as clusters do not carry it."""
    cluster = client.get(f"{BASE_URL}/clusters/{cluster_id}/")
    docket_id = url_id(cluster["docket"], "dockets")
    docket = client.get(f"{BASE_URL}/dockets/{docket_id}/")
    court = courts.normalize_court(court_slug(docket["court"]))
    citations = [
        f'{c["volume"]} {c["reporter"]} {c["page"]}'
        for c in cluster["citations"]
    ]
    return {
        "case_name": cluster["case_name"],
        "decision_date": cluster["date_filed"],
        **court,
        "courtlistener_docket_id": docket_id,
        "docket_number": docket["docket_number"],
        "judges": cluster["judges"],
        "precedential_status": cluster["precedential_status"],
        "citation_count": cluster["citation_count"],
        "citation": json.dumps(list(dict.fromkeys(citations))),
    }


def fetch_texts(client, conn):
    """Fetch text and metadata for every discovered opinion."""
    rows = conn.execute(
        """SELECT id, courtlistener_opinion_id FROM cases WHERE fetch_status = 'discovered' ORDER BY citation_depth DESC, courtlistener_opinion_id"""
    ).fetchall()
    for i, (row_id, opinion_id) in enumerate(rows, 1):
        opinion = client.get(f"{BASE_URL}/opinions/{opinion_id}/")
        cluster_id = url_id(opinion["cluster"], "clusters")
        text, html = opinion_text(opinion)
        fields = cluster_metadata(client, cluster_id)
        fields["courtlistener_cluster_id"] = cluster_id
        fields["date_fetched"] = datetime.now().isoformat()
        if text:
            fields["opinion_text"] = text
            fields["opinion_html"] = html
            fields["text_source"] = (
                "opinion_plain_text"
                if opinion["plain_text"]
                else "opinion_html_converted"
            )
            fields["fetch_status"] = "fetched"
        else:
            fields["fetch_status"] = "no_text_available"
        conn.execute(
            f"""UPDATE cases SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?""",
            [*fields.values(), row_id],
        )
        conn.commit()
        print(
            f"[{i}/{len(rows)}] opinion {opinion_id}: {fields['fetch_status']}"
        )


def fetch(db, citation=None, cluster_id=None, opinion_id=None, min_depth=1):
    """Fetch the citing opinions of a seed case into the SQLite ``db``."""
    client = CLClient(os.environ["COURTLISTENER_API_KEY"])
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    seed_opinion_ids, _ = resolve_seed(
        client, citation, opinion_id, cluster_id
    )
    discover(client, conn, seed_opinion_ids, min_depth)
    print(
        f"""Found {conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0]} citing opinions"""
    )
    fetch_texts(client, conn)
    conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("citation", nargs="?", help="e.g. '597 U.S. 1'")
    parser.add_argument("--cluster-id", type=int)
    parser.add_argument("--opinion-id", type=int)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--min-depth",
        type=int,
        default=1,
        help="keep only opinions citing the seed at least this many times",
    )
    args = parser.parse_args()
    fetch(
        args.db,
        args.citation,
        args.cluster_id,
        args.opinion_id,
        args.min_depth,
    )


if __name__ == "__main__":
    main()
