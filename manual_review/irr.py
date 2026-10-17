"""Cohen's kappa between the human review and the Gemini second rater.

Verdicts are collapsed to agree (agree, no-sources-agree) versus disagree
(disagree, no-sources-disagree). The 95% CI is a percentile bootstrap over
cases.

    python -m manual_review.irr
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import random
from typing import Any

from manual_review.model_label import CASE_VERDICTS
from manual_review.model_label import DEFAULT_MODEL
from manual_review.model_label import judged_cases
from manual_review.model_label import output_dir

LABELS = ("agree", "disagree")
N_BOOT = 1000
SEED = 0


@dataclass
class PairedCase:
    review_id: str
    doctrine: str
    human: str
    model: str


def binary_verdict(label: str) -> str:
    if label in ("agree", "no-sources-agree"):
        return "agree"
    if label in ("disagree", "no-sources-disagree"):
        return "disagree"
    raise ValueError(f"{label!r} is not a scored verdict")


def confusion_matrix(
    y_human: list[str], y_model: list[str]
) -> dict[str, dict[str, int]]:
    """Rows are the human label, columns the model label."""
    table = {label: {other: 0 for other in LABELS} for label in LABELS}
    for left, right in zip(y_human, y_model):
        table[left][right] += 1
    return table


def kappa(y_human: list[str], y_model: list[str]) -> float | None:
    """Cohen's kappa ``(p_o - p_e) / (1 - p_e)``; ``None`` if ``p_e`` is 1."""
    table = confusion_matrix(y_human, y_model)
    n = len(y_human)
    p_o = sum(table[label][label] for label in LABELS) / n
    p_e = (
        sum(
            sum(table[label].values())
            * sum(table[row][label] for row in LABELS)
            for label in LABELS
        )
        / n**2
    )
    if p_e == 1:
        return None
    return (p_o - p_e) / (1 - p_e)


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_kappa(
    y_human: list[str],
    y_model: list[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile 95% CI, resampling cases with replacement. Resamples where
    kappa is undefined are dropped."""
    rng = random.Random(seed)
    n = len(y_human)
    scores = []
    for _ in range(n_boot):
        indexes = [rng.randrange(n) for _ in range(n)]
        score = kappa(
            [y_human[i] for i in indexes], [y_model[i] for i in indexes]
        )
        if score is not None:
            scores.append(score)
    scores.sort()
    return _percentile(scores, 0.025), _percentile(scores, 0.975)


def pair_cases(
    model_cases: list[dict[str, Any]], humans_by_id: dict[str, Any]
) -> list[PairedCase]:
    """Join the model's labels to the human verdicts on ``review_id``."""
    paired = []
    for row in model_cases:
        human = humans_by_id[row["review_id"]]
        assert human.tree_agreement in CASE_VERDICTS, human.review_id
        paired.append(
            PairedCase(
                review_id=row["review_id"],
                doctrine=human.doctrine,
                human=binary_verdict(human.tree_agreement),
                model=binary_verdict(row["tree_agreement"]),
            )
        )
    return paired


def score(pairs: list[PairedCase], *, n_boot: int, seed: int) -> dict[str, Any]:
    human = [pair.human for pair in pairs]
    model = [pair.model for pair in pairs]
    low, high = bootstrap_kappa(human, model, n_boot=n_boot, seed=seed)
    return {
        "n": len(pairs),
        "kappa": kappa(human, model),
        "observed": sum(a == b for a, b in zip(human, model)) / len(pairs),
        "ci_low": low,
        "ci_high": high,
        "confusion": confusion_matrix(human, model),
    }


def build_report(
    pairs: list[PairedCase], *, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    """Kappa for all cases and for each doctrine."""
    doctrines = list(dict.fromkeys(pair.doctrine for pair in pairs))
    strata = {"all": pairs}
    for doctrine in doctrines:
        strata[doctrine] = [p for p in pairs if p.doctrine == doctrine]
    return {
        name: score(group, n_boot=n_boot, seed=seed + 1 + 10 * offset)
        for offset, (name, group) in enumerate(strata.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, metavar="RUN_ID")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    labels_path = output_dir(args.model, args.run) / "labels.json"
    model_cases = json.loads(labels_path.read_text())["cases"]
    humans = {case.review_id: case for case in judged_cases(args.run)}
    report = build_report(
        pair_cases(model_cases, humans), n_boot=args.n_boot, seed=args.seed
    )
    for name, row in report.items():
        print(
            f"""{name:<14} n={row['n']:<4} kappa={row['kappa']:.3f}  observed={row['observed']:.3f}  95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"""
        )
    report_path = labels_path.parent / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
