"""Statistical tests for the facet and date associations (scipy).

``correction=False`` keeps 2x2 chi-squares from being Yates-corrected, and the
FDR uses ``method="bh"``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import false_discovery_control
from scipy.stats import fisher_exact
from scipy.stats import mannwhitneyu
from scipy.stats.contingency import association

MONTE_CARLO_N = 20_000


# ── contingency tables ────────────────────────────────────────────────────────


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p."""
    if a + b + c + d == 0:
        return 1.0
    if (a + b) in (0, a + b + c + d) or (a + c) in (0, a + b + c + d):
        return 1.0  # no variation on one axis; scipy would return nan margins
    return float(fisher_exact([[a, b], [c, d]])[1])


def _chi2(table: list[list[int]]):
    """(statistic, p, dof, min expected)"""
    arr = np.asarray(table, dtype=int)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2 or arr.sum() == 0:
        return 0.0, 1.0, 0, 0.0
    if (arr.sum(axis=1) == 0).any() or (arr.sum(axis=0) == 0).any():
        return 0.0, 1.0, 0, 0.0
    res = chi2_contingency(arr, correction=False)
    return (
        float(res.statistic),
        float(res.pvalue),
        int(res.dof),
        float(res.expected_freq.min()),
    )


def _monte_carlo_p(
    table: list[list[int]], observed: float, n_iter: int, seed: int
) -> float:
    """Permutation p: reshuffle one axis' labels, holding both margins' totals.
    """
    arr = np.asarray(table, dtype=int)
    rows, cols = arr.shape
    row_labels = np.repeat(np.arange(rows), arr.sum(axis=1))
    col_labels = np.repeat(np.arange(cols), arr.sum(axis=0))
    n = row_labels.size
    if n == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    hits = 0
    chunk = max(1, min(n_iter, 2000))
    done = 0
    while done < n_iter:
        size = min(chunk, n_iter - done)
        done += size
        perms = np.argsort(rng.random((size, n)), axis=1)
        shuffled = col_labels[perms]
        flat = row_labels[None, :] * cols + shuffled
        counts = np.apply_along_axis(
            np.bincount, 1, flat, minlength=rows * cols
        )
        tables = counts.reshape(size, rows, cols).astype(float)

        row_tot = tables.sum(axis=2, keepdims=True)
        col_tot = tables.sum(axis=1, keepdims=True)
        expected = row_tot * col_tot / n
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                expected > 0, (tables - expected) ** 2 / expected, 0.0
            )
        hits += int((terms.sum(axis=(1, 2)) >= observed - 1e-12).sum())

    return (hits + 1) / (n_iter + 1)


def cramers_v(table: list[list[int]]) -> float:
    """Cramer's V of a contingency table; 0 for empty or degenerate tables."""
    arr = np.asarray(table, dtype=int)
    if arr.sum() == 0 or min(arr.shape) < 2:
        return 0.0
    if (arr.sum(axis=1) == 0).any() or (arr.sum(axis=0) == 0).any():
        return 0.0
    return float(association(arr, method="cramer"))


def test_table(
    table: list[list[int]], *, seed: int = 0, monte_carlo: int = MONTE_CARLO_N
) -> dict[str, Any]:
    """Pick the right test for the table's shape and sparsity."""
    n = int(sum(sum(r) for r in table))
    chi2_stat, chi2_p, dof, min_exp = _chi2(table)
    if len(table) == 2 and len(table[0]) == 2:
        method = "fisher"
        p = fisher_exact_2x2(table[0][0], table[0][1], table[1][0], table[1][1])
    elif min_exp < 5:
        method = "chi2_montecarlo"
        p = _monte_carlo_p(table, chi2_stat, monte_carlo, seed)
    else:
        method = "chi2"
        p = chi2_p
    return {
        "n": n,
        "chi2": round(chi2_stat, 4),
        "dof": dof,
        "p": p,
        "method": method,
        "cramers_v": round(cramers_v(table), 4),
    }


def benjamini_hochberg(ps: Sequence[float]) -> list[float]:
    """BH-adjusted p-values (q-values), in the input order."""
    if not len(ps):
        return []
    return [
        float(q)
        for q in false_discovery_control(
            np.asarray(ps, dtype=float), method="bh"
        )
    ]


def attach_fdr(
    rows: list[dict[str, Any]], *, p_key: str = "p"
) -> list[dict[str, Any]]:
    """Write ``q_bh`` and ``n_tests_in_family`` onto each row."""
    qs = benjamini_hochberg([r[p_key] for r in rows])
    m = len(rows)
    for row, q in zip(rows, qs):
        row["q_bh"] = q
        row["n_tests_in_family"] = m
    return rows


# ── ranks ─────────────────────────────────────────────────────────────────────


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Cliff's δ / rank-biserial: 2U/(n1 n2) - 1.
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    return round(2.0 * u / (n1 * n2) - 1.0, 4)


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Two-sided Mann–Whitney U of x vs y."""
    a = list(x)
    b = list(y)
    n1, n2 = len(a), len(b)
    empty = {
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "u": 0.0,
        "p": 1.0,
        "method": "mannwhitney",
        "rank_biserial": 0.0,
        "median1": None,
        "median2": None,
    }
    if n1 == 0 or n2 == 0:
        return empty
    if all(v == a[0] for v in a) and all(v == a[0] for v in b):
        empty["median1"] = float(a[0])
        empty["median2"] = float(b[0])
        return empty
    res = mannwhitneyu(a, b, alternative="two-sided")
    u = float(res.statistic)
    return {
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "u": round(u, 4),
        "p": float(res.pvalue),
        "method": "mannwhitney",
        "rank_biserial": rank_biserial(u, n1, n2),
        "median1": round(float(np.median(a)), 4),
        "median2": round(float(np.median(b)), 4),
    }
