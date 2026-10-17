"""Outbound citation edges of a doctrine's corpus, from CourtListener.
"""

from courtlistener import fetch_courtlistener as cl


def fetch_edges(
    client: cl.CLClient, opinion_ids: list[int]
) -> dict[int, dict[int, int]]:
    """Fetches every opinion each of ``opinion_ids`` cites.
    """
    edges = {}
    for i, opinion_id in enumerate(opinion_ids, 1):
        cited = {}
        url = f"{cl.BASE_URL}/opinions-cited/"
        params = {"citing_opinion": opinion_id, "page_size": cl.PAGE_SIZE}
        while url:
            page = client.get(url, params)
            params = None
            for record in page["results"]:
                cited_id = cl.url_id(record["cited_opinion"], "opinions")
                cited[cited_id] = max(
                    cited[cited_id] if cited_id in cited else 0,
                    record["depth"],
                )
            url = page["next"]
        edges[opinion_id] = cited
        print(f"[{i}/{len(opinion_ids)}] opinion {opinion_id}: {len(cited)}")
    return edges
