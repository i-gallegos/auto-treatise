"""Generalization: place held-out cases into an existing Auto-Treatise.

Each stage runs per doctrine against one tree run (--run-id) on the test split. Run from the repository root.

  extract    write x^[u] (facts, posture, issue, theories) for the split
  segment    annotator segmentation and profile of each held-out opinion
  annotate   gold leaf labels (Gemini 2.5 Pro)
  classify   tree descent on x^[u]
  forced     teacher-forced tree decisions along the gold path
  embedding  text-embedding-3-large nearest-neighbour baseline
  factor     circuit and appointing-party baselines along the gold path
  report     per-case rows and the pooled summary with 95% CIs
  all        every stage above, in order

Example:

  python -m generalization --stage all \\
      --doctrine <d> --run-id <run>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from auto_treatise import doctrine as doctrine_paths  # noqa: E402

from . import data  # noqa: E402
from . import views  # noqa: E402

#: The held-out split the paper evaluates on.
SPLIT = "test"

ANNOTATOR_MODEL = "gemini-2.5-pro"

STAGES = (
    "extract",
    "segment",
    "annotate",
    "classify",
    "forced",
    "embedding",
    "factor",
    "report",
)


def _split_ids(doctrine: str, split: str) -> list[str]:
    cfg = doctrine_paths.load_config(doctrine)
    payload = doctrine_paths.load_split(doctrine, cfg["split"]["file"])
    return [str(i) for i in payload[split]]


def _resolve_run(doctrine: str, run_id: str) -> tuple[str, str]:
    """Return ``(run_id, model_name)`` for the tree run."""
    tree = data.load_tree(doctrine, run_id)
    return run_id, tree.metadata["model_name"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m generalization",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", required=True, choices=STAGES + ("all",))
    p.add_argument(
        "--doctrine",
        nargs="+",
        required=True,
        metavar="DOCTRINE",
        help="Doctrines to run (directories under doctrines/).",
    )
    p.add_argument(
        "--run-id",
        dest="run_id",
        required=True,
        help="Tree run to evaluate against.",
    )
    p.add_argument(
        "--annotator-model",
        dest="annotator_model",
        default=ANNOTATOR_MODEL,
        help=f"Model for gold labels and segmentation (default: {ANNOTATOR_MODEL}).",
    )
    return p


def _run_extract(doctrine: str, case_ids: list[str], model: str) -> None:
    constraint = doctrine_paths.load_config(doctrine)["tree"]["constraint"]
    summary = views.extract(
        doctrine=doctrine,
        case_ids=case_ids,
        constraint=constraint,
        model=model,
    )
    print(
        f"""[extract] done  {doctrine}  model={model}  written={summary['n_written']}  cached={summary['n_cached']}"""
    )


def _run_summaries(doctrines: list[str], run_ids: list[str]) -> None:
    """The pooled summary, once every doctrine has its inputs."""
    from . import metrics

    if len(set(run_ids)) != 1:
        print(
            f"""[report] skipping pooled summary: doctrines resolve to different runs {sorted(set(run_ids))}; pin one with --run-id"""
        )
        return
    run_id = run_ids[0]
    missing = metrics.summary_inputs_missing(doctrines, run_id)
    if missing:
        shown = ", ".join(str(p) for p in missing[:4])
        print(
            f"""[report] skipping pooled summary: {len(missing)} input(s) missing ({shown}); run --stage forced and --stage factor"""
        )
        return
    written = metrics.write_summaries(doctrines, run_id)
    for path in written.values():
        print(f"[report] wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    stages = STAGES if args.stage == "all" else (args.stage,)
    later = [s for s in stages if s not in ("extract", "report")]
    run_ids: list[str] = []
    for doctrine in args.doctrine:
        run_id, run_model = _resolve_run(doctrine, args.run_id)
        run_ids.append(run_id)
        case_ids = _split_ids(doctrine, SPLIT)
        print(
            f"""[run] {doctrine}  stage={args.stage}  split={SPLIT}  n={len(case_ids)}  run_id={run_id}  model={run_model}  annotator={args.annotator_model}"""
        )
        if "extract" in stages:
            _run_extract(doctrine, case_ids, run_model)
        if later:
            # Extraction sees every case; every later stage drops the cases
            # it flagged as not engaging the constraint.
            scored, dropped = views.partition_engaged(doctrine, case_ids)
            if dropped:
                print(
                    f"""[run] {doctrine}: dropping {len(dropped)} of {len(case_ids)} {SPLIT} cases flagged constraint_engagement={views.NOT_ENGAGED}"""
                )
        for stage in later:
            if stage == "segment":
                from .annotate import run_segmentation

                tree = data.load_tree(doctrine, run_id)
                run_segmentation(
                    doctrine,
                    data.accepted_ids(doctrine, tree, scored),
                    constraint=tree.metadata["constraint"],
                    model=args.annotator_model,
                )
            elif stage == "annotate":
                from .annotate import run_annotation

                run_annotation(
                    doctrine, run_id, scored, args.annotator_model
                )
            elif stage == "classify":
                from .classify import run_classification

                run_classification(doctrine, run_id, scored, run_model)
            elif stage == "forced":
                from .classify import run_forced

                run_forced(doctrine, run_id, scored, run_model)
            elif stage == "embedding":
                from .baselines import run_embedding_nn

                run_embedding_nn(doctrine, run_id, scored)
            elif stage == "factor":
                from .baselines import run_factor

                run_factor(doctrine, run_id, scored)
        print(f"[run] done  {doctrine}  {args.stage}")
    if "report" in stages:
        _run_summaries(list(args.doctrine), run_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
