"""Embedding nearest neighbour and the factor baselines.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sqlite3
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

# Embeddings come from OpenAI.
from api_calls import create_embedding  # noqa: E402

from . import data  # noqa: E402
from .artifacts import _meta  # noqa: E402
from .artifacts import _pass_status  # noqa: E402
from .artifacts import _prepare  # noqa: E402
from .artifacts import _sort_case_rows  # noqa: E402
from .artifacts import _write  # noqa: E402
from .metrics import weighted_distribution  # noqa: E402

#: Parallel embedding calls.
MAX_WORKERS = 8

#: ``text-embedding-3-large`` context. Windows sit under it; overlap keeps a
#: split from dropping the sentence at the cut.
_EMBED_WINDOW = 8000
_EMBED_OVERLAP = 200
_EMBED_ENCODING = None


def _encoding():
    """Loaded on first use, so a run served entirely from the cache needs
    no download."""
    global _EMBED_ENCODING
    if _EMBED_ENCODING is None:
        _EMBED_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _EMBED_ENCODING


# --------------------------------------------------------------------------- #
# embeddings.db
# --------------------------------------------------------------------------- #


def _embeddings_db(doctrine: str) -> Path:
    return data.doctrine_dir(doctrine) / "data" / "embeddings.db"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _connect(doctrine: str) -> sqlite3.Connection:
    path = _embeddings_db(doctrine)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
          case_id    TEXT NOT NULL,
          model      TEXT NOT NULL,
          text_hash  TEXT NOT NULL,
          embedding  BLOB NOT NULL,
          PRIMARY KEY (case_id, model, text_hash)
        )
        """)
    return conn


def _get_vectors(
    conn: sqlite3.Connection,
    items: list[tuple[str, str]],
    model: str,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str, str]]]:
    """Lookup ``(case_id, text)`` rows. Returns ``(found, missing)``.
    """
    found: dict[str, np.ndarray] = {}
    missing: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for case_id, text in items:
        if case_id in seen:
            continue
        seen.add(case_id)
        text_hash = _text_hash(text)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE case_id=? AND model=? AND text_hash=?",
            (case_id, model, text_hash),
        ).fetchone()
        if row is None:
            missing.append((case_id, text, text_hash))
        else:
            found[case_id] = _unpack(row[0])
    return found, missing


def _put_vectors(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, Any]],
    model: str,
) -> None:
    """Insert ``(case_id, text_hash, vector)`` rows.
    Call from the main thread."""
    conn.executemany(
        """INSERT OR REPLACE INTO embeddings (case_id, model, text_hash, embedding) VALUES (?, ?, ?, ?)""",
        [
            (case_id, model, text_hash, _pack(vector))
            for case_id, text_hash, vector in rows
        ],
    )
    conn.commit()


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into overlapping windows that fit the embedding model."""
    enc = _encoding()
    ids = enc.encode(text)
    if len(ids) <= _EMBED_WINDOW:
        return [text]
    step = _EMBED_WINDOW - _EMBED_OVERLAP
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        chunks.append(enc.decode(ids[start : start + _EMBED_WINDOW]))
        if start + _EMBED_WINDOW >= len(ids):
            break
        start += step
    return chunks


def _mean_pool(vectors: list) -> np.ndarray:
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean = mean / norm
    return mean


def _embed_one(text: str, model: str) -> tuple[np.ndarray, int]:
    """Embed ``text``, pooling windows when it exceeds the model context.
    """
    chunks = _chunk_text(text)
    vectors = [create_embedding(chunk, model=model) for chunk in chunks]
    return _mean_pool(vectors), len(chunks)


def _embed_texts(
    doctrine: str, items: list[tuple[str, str]], model: str
) -> dict[str, np.ndarray]:
    """Return a vector per case, reading ``embeddings.db`` and
    embedding misses."""
    conn = _connect(doctrine)
    try:
        found, missing = _get_vectors(conn, items, model)
        if missing:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                results = list(
                    pool.map(lambda row: _embed_one(row[1], model), missing)
                )
            n_split = sum(1 for _, n in results if n > 1)
            if n_split:
                n_windows = sum(n for _, n in results if n > 1)
                print(
                    f"""[embedding_nn] {n_split} overlong text(s) split into {n_windows} windows"""
                )
            written = [
                (case_id, text_hash, vector)
                for (case_id, _, text_hash), (vector, _) in zip(
                    missing, results
                )
            ]
            _put_vectors(conn, written, model)
            for case_id, _, vector in written:
                found[case_id] = np.asarray(vector, dtype=np.float32)
        return found
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _top_neighbours(
    ids: list[str],
    train_ids: list[str],
    vecs: dict[str, np.ndarray],
    m: int,
) -> dict[str, list[tuple[str, float]]]:
    """``{held_out_id: [(train_id, cosine), ...]}`` of length ``m``,
    highest first."""
    if not ids or not train_ids:
        return {case_id: [] for case_id in ids}

    train = np.asarray([vecs[t] for t in train_ids], dtype=np.float32)
    held = np.asarray([vecs[c] for c in ids], dtype=np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True).clip(min=1e-12)
    held /= np.linalg.norm(held, axis=1, keepdims=True).clip(min=1e-12)
    sims = held @ train.T

    k = min(m, train.shape[0])
    if k < train.shape[0]:
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    else:
        top = np.broadcast_to(np.arange(k), (held.shape[0], k)).copy()

    neighbours: dict[str, list[tuple[str, float]]] = {}
    for i, case_id in enumerate(ids):
        ranked = sorted(
            ((train_ids[j], float(sims[i, j])) for j in top[i]),
            key=lambda kv: -kv[1],
        )
        neighbours[case_id] = ranked
    return neighbours


def run_embedding_nn(
    doctrine: str,
    run_id: str,
    case_ids: list[str],
    m: int = data.DEFAULT_M,
    model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    """Leaf distribution per held-out case from its ``m`` nearest neighbours.
    """
    tree, _, accepted, _ = _prepare(doctrine, run_id, case_ids)
    case_to_leaf = tree.case_to_leaf()
    # Asymmetric on purpose: the training pool is embedded from the distilled
    # majority opinion, the held-out cases from x^[u]. See data.retrieval_pool.
    train_ids, train_views = data.retrieval_pool(doctrine, tree)
    held_views = data.fact_views(doctrine, accepted)
    ids = [c for c in accepted if c in held_views]

    path = data.predictions_path(doctrine, run_id, "embedding_nn")
    _pass_status("embedding_nn", len(accepted), len(ids))

    items = [(c, train_views[c]) for c in train_ids] + [
        (c, held_views[c]) for c in ids
    ]
    vecs = _embed_texts(doctrine, items, model)
    neighbours = _top_neighbours(ids, train_ids, vecs, m)

    predictions = []
    for case_id in ids:
        scored = neighbours[case_id]
        predictions.append(
            {
                "case_id": case_id,
                "distribution": weighted_distribution(
                    [(case_to_leaf[t], s) for t, s in scored]
                ),
                "neighbours": [
                    {"case_id": t, "similarity": s, "leaf": case_to_leaf[t]}
                    for t, s in scored
                ],
            }
        )

    payload = {
        "meta": _meta(
            doctrine,
            run_id,
            tree,
            model=model,
            m=m,
            n_candidates=len(train_ids),
        ),
        "predictions": _sort_case_rows(predictions),
    }
    _write(path, payload)
    return payload


# --------------------------------------------------------------------------- #
# Factor baselines: circuit, author party
# --------------------------------------------------------------------------- #

_PARTY_VALUES = frozenset({"R", "D"})


def _text_field(
    row: dict, field: str, allowed: frozenset[str] | None = None
) -> str | None:
    if field not in row or row[field] is None:
        return None
    text = str(row[field]).strip()
    if not text:
        return None
    if allowed is not None and text not in allowed:
        return None
    return text


def _majority(n_left: int, n_right: int) -> str:
    """The child with more training cases. A tie goes left."""
    if n_right > n_left:
        return "right"
    return "left"


def _count_side(
    left_ids: list[str],
    right_ids: list[str],
    values: dict[str, str | None],
    wanted: str,
) -> tuple[str, bool]:
    """Side with more training cases equal to ``wanted``. A tie falls back."""
    n_left = sum(
        1
        for case_id in left_ids
        if case_id in values and values[case_id] == wanted
    )
    n_right = sum(
        1
        for case_id in right_ids
        if case_id in values and values[case_id] == wanted
    )
    if n_left == n_right:
        return _majority(len(left_ids), len(right_ids)), True
    if n_left > n_right:
        return "left", False
    return "right", False


def _child_cases(nodes: list[dict]) -> dict[str, dict]:
    """Per interior node, the training cases under each child."""
    by_id = {node["id"]: node for node in nodes}
    out: dict[str, dict] = {}
    for node in nodes:
        if "left_id" not in node or "right_id" not in node:
            continue
        left_id, right_id = node["left_id"], node["right_id"]
        if not left_id or not right_id:
            continue
        if left_id not in by_id or right_id not in by_id:
            continue
        left, right = by_id[left_id], by_id[right_id]
        left_ids = (
            [str(c) for c in left["cases"]]
            if "cases" in left and left["cases"]
            else []
        )
        right_ids = (
            [str(c) for c in right["cases"]]
            if "cases" in right and right["cases"]
            else []
        )
        if not left_ids or not right_ids:
            continue
        out[node["id"]] = {"left_ids": left_ids, "right_ids": right_ids}
    return out


def _heldout_rows(doctrine: str, case_ids: list[str]) -> dict[str, dict]:
    """Circuit and author party for cases absent from the run's case index."""
    if not case_ids:
        return {}
    conn = data._connect(doctrine)
    try:
        columns = set(data._columns(conn))
        wanted = [c for c in ("circuit", "judge_author_party") if c in columns]
        if not wanted:
            return {}
        marks = ",".join("?" for _ in case_ids)
        rows = conn.execute(
            f"SELECT id, {', '.join(wanted)} FROM cases WHERE id IN ({marks})",
            [int(c) if str(c).isdigit() else c for c in case_ids],
        ).fetchall()
    finally:
        conn.close()
    return {str(row[0]): dict(zip(wanted, row[1:])) for row in rows}


def _gold_steps(tree: data.Tree, row: dict) -> list[dict]:
    steps = []
    path = row["path"] if "path" in row and row["path"] else []
    for step in path:
        if "node_id" not in step or "side" not in step or not step["side"]:
            continue
        node_id = step["node_id"]
        if node_id not in tree.nodes:
            continue
        node = tree.nodes[node_id]
        if "is_leaf" in node and node["is_leaf"]:
            continue
        steps.append(
            {
                "node_id": node_id,
                "depth": int(node["depth"]),
                "gold_side": step["side"],
            }
        )
    return steps


def _train_values(
    index: dict[str, dict],
    case_ids: list[str],
    field: str,
    allowed: frozenset[str] | None,
) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for case_id in case_ids:
        if case_id not in index:
            values[case_id] = None
            continue
        values[case_id] = _text_field(index[case_id], field, allowed)
    return values


def run_factor(
    doctrine: str, run_id: str, case_ids: list[str]
) -> dict[str, Any]:
    """Route each held-out case by circuit or author party.

    At a gold node the training pool is that child's ``cases`` list, which
    repeats every descendant, so it is the subtree. Circuit and author party
    take the child with more training cases sharing the held-out value. A tie
    falls back to the child with more training cases.

    Cases with no circuit, and cases whose author party is not ``R`` or ``D``,
    are written with that feature null and left out of its series.
    """
    from descriptive_findings.analyze import load_run

    tree, _, accepted, _ = _prepare(doctrine, run_id, case_ids)
    stored = data.read_json(data.labels_path(doctrine, run_id)) or {}
    label_rows = (
        stored["labels"] if "labels" in stored and stored["labels"] else []
    )
    labels = {row["case_id"]: row for row in label_rows}
    if not labels:
        raise SystemExit(
            f"""[factor] no {data.LABELS_FILE} for {doctrine}/{run_id}; run the annotate stage first"""
        )
    views = data.fact_views(doctrine, accepted)
    ids = [c for c in accepted if c in views and c in labels]

    path = data.predictions_path(doctrine, run_id, "factor")

    nodes, index = load_run(data.run_dir(doctrine, run_id), doctrine)
    splits = _child_cases(nodes)

    missing = [c for c in ids if c not in index]
    extra = _heldout_rows(doctrine, missing)

    def held(
        case_id: str, field: str, allowed: frozenset[str] | None = None
    ) -> str | None:
        if case_id in index:
            return _text_field(index[case_id], field, allowed)
        if case_id in extra:
            return _text_field(extra[case_id], field, allowed)
        return None

    rows = []
    for case_id in ids:
        circuit = held(case_id, "circuit")
        party = held(case_id, "judge_author_party", _PARTY_VALUES)
        decisions = []
        for step in _gold_steps(tree, labels[case_id]):
            if step["node_id"] not in splits:
                raise SystemExit(
                    f"""[factor] gold node {step['node_id']} has no children ({doctrine}/{case_id})"""
                )
            info = splits[step["node_id"]]
            gold = step["gold_side"]
            block = {
                "node_id": step["node_id"],
                "depth": step["depth"],
                "gold_side": gold,
                "circuit": None,
                "party": None,
            }
            if circuit is not None:
                values = _train_values(
                    index, info["left_ids"] + info["right_ids"], "circuit", None
                )
                side, fallback = _count_side(
                    info["left_ids"], info["right_ids"], values, circuit
                )
                block["circuit"] = {
                    "side": side,
                    "correct": side == gold,
                    "fallback": fallback,
                }
            if party is not None:
                values = _train_values(
                    index,
                    info["left_ids"] + info["right_ids"],
                    "judge_author_party",
                    _PARTY_VALUES,
                )
                side, fallback = _count_side(
                    info["left_ids"], info["right_ids"], values, party
                )
                block["party"] = {
                    "side": side,
                    "correct": side == gold,
                    "fallback": fallback,
                }
            decisions.append(block)
        rows.append(
            {
                "case_id": case_id,
                "circuit": circuit,
                "party": party,
                "decisions": decisions,
            }
        )

    def _rate(feature: str) -> tuple[float | None, int]:
        flags = [
            bool(decision[feature]["correct"])
            for row in rows
            for decision in row["decisions"]
            if decision[feature] is not None
        ]
        if not flags:
            return None, 0
        return sum(flags) / len(flags), len(flags)

    circuit_acc, n_circuit = _rate("circuit")
    party_acc, n_party = _rate("party")
    payload = {
        "meta": _meta(
            doctrine,
            run_id,
            tree,
            n_cases=len(rows),
            n_with_circuit=sum(1 for row in rows if row["circuit"] is not None),
            n_with_party=sum(1 for row in rows if row["party"] is not None),
            circuit_accuracy=circuit_acc,
            party_accuracy=party_acc,
        ),
        "predictions": _sort_case_rows(rows),
    }
    _write(path, payload)

    def _shown(rate: float | None) -> str:
        if rate is None:
            return "n/a"
        return f"{rate:.3f}"

    print(
        f"""[factor] {len(rows)} case(s): circuit {_shown(circuit_acc)} (n={n_circuit}), party {_shown(party_acc)} (n={n_party})"""
    )
    return payload
