"""
Build the Auto-Treatise tree for one or more doctrines.

Run from the repository root.

Stages
------
  segment   Relevance and sentence segmentation on the train split. Held-out
            rows get majority isolation only (regex, no LLM). Results are
            cached per row in the doctrine's DB, so re-running is cheap.
  build     Extract case profiles and induce the binary dendrogram on the
            train split. Writes doctrines/<d>/analysis/runs/<run_id>/.

Examples
--------
  # Segmentation (model comes from each doctrine's config.toml)
  python -m auto_treatise --stage segment --doctrine <d> [<d> ...]

  # Tree build
  python -m auto_treatise --stage build --doctrine <d> --tree-model gpt-5.2 \\
      --run-id <run>
"""

import argparse
import json
import sys

from .doctrine import analysis_root as _analysis_root
from .doctrine import db_path as _db_path
from .doctrine import load_config as _load_config
from .doctrine import load_split as _load_split

TEST_N = 10


def _limit(ids: list, test: bool) -> list:
    return ids[:TEST_N] if test else ids


def stage_segment(args: argparse.Namespace) -> None:
    """Relevance + segmentation on train; majority isolation on held-out."""
    from .majority import store_boundaries
    from .paragraph_pipeline import prepare_segmentations

    for doctrine in args.doctrine:
        cfg = _load_config(doctrine)
        seg_cfg = cfg["segmentation"]
        split_data = _load_split(doctrine, cfg["split"]["file"])
        train_ids = [int(i) for i in _limit(split_data["train"], args.test)]
        held_out_ids = []
        if not args.test:
            held_out_ids = sorted(
                {
                    int(i)
                    for i in (
                        split_data["dev"]
                        + split_data["test"]
                        + split_data["undated"]
                    )
                }
            )
        model = args.segmenter_model or seg_cfg["model"]
        out_dir = _analysis_root(doctrine, args.test)
        db_path = _db_path(doctrine)

        print(
            f"""[run] segment  doctrine={doctrine}  train_ids={len(train_ids)}  held_out_majority={len(held_out_ids)}  model={model}{'  test' if args.test else ''}"""
        )
        if args.dry_run:
            continue

        if held_out_ids:
            store_boundaries(db_path, held_out_ids)

        results = prepare_segmentations(
            query=seg_cfg["query"],
            db_path=db_path,
            ids=train_ids,
            majority_only=seg_cfg["majority_only"],
            model=model,
        )
        if args.test:
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "doctrine": doctrine,
                "stage": "segment",
                "n": len(train_ids),
                "ids": [str(i) for i in train_ids],
                "model": model,
                "results": results,
            }
            out_path = out_dir / "segmentations.json"
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"[run] segment  wrote {out_path}")
        print(f"[run] segment  done  doctrine={doctrine}")


def stage_build(args: argparse.Namespace) -> None:
    """Extract case profiles and induce the tree on the train split."""
    if not args.tree_model:
        sys.exit("[run] ERROR: --tree-model is required for the build stage")

    from .tree import MAX_RECURSION_DEPTH
    from .tree import run_pipeline
    from .tree import TEST_MAX_RECURSION_DEPTH

    for doctrine in args.doctrine:
        cfg = _load_config(doctrine)
        split_data = _load_split(doctrine, cfg["split"]["file"])
        train_ids = [int(i) for i in _limit(split_data["train"], args.test)]
        analysis_root = _analysis_root(doctrine, args.test)
        run_type = "test" if args.test else "full"
        max_recursion_depth = (
            TEST_MAX_RECURSION_DEPTH if args.test else MAX_RECURSION_DEPTH
        )

        print(
            f"""[run] build  doctrine={doctrine}  train_ids={len(train_ids)}  model={args.tree_model}  run_type={run_type}  analysis={analysis_root}  max_depth={max_recursion_depth}  input_mode={args.input_mode}  profile={'off' if args.no_profile else 'on'}"""
        )
        if args.dry_run:
            continue

        run_pipeline(
            doctrine=doctrine,
            constraint=cfg["tree"]["constraint"],
            subset_ids=train_ids if train_ids else None,
            model=args.tree_model,
            analysis_root=analysis_root,
            run_type=run_type,
            max_recursion_depth=max_recursion_depth,
            input_mode=args.input_mode,
            use_profile=not args.no_profile,
            run_id=args.run_id,
        )
        print(f"[run] build  done  doctrine={doctrine}")


STAGES = {
    "segment": stage_segment,
    "build": stage_build,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m auto_treatise",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        required=True,
        choices=list(STAGES),
        help="Pipeline stage to run.",
    )
    p.add_argument(
        "--doctrine",
        nargs="+",
        required=True,
        metavar="DOCTRINE",
        help="One or more doctrine names (directories under doctrines/).",
    )
    p.add_argument(
        "--segmenter-model",
        dest="segmenter_model",
        default=None,
        help="""Override the segmenter model from config.toml.  By default the model in [segmentation] is used.""",
    )
    p.add_argument(
        "--input-mode",
        dest="input_mode",
        default="segmented",
        choices=["segmented", "relevant", "full"],
        help="""Build: text the profiler reads before induction. segmented (default) is labeled relevant paragraphs. relevant is those same paragraphs as plain text, with no sentence labels. full is the stored opinion, with no relevance filter and no segmentation. relevant and full write a separate profile cache and a separate run directory, and leave analysis/case_level.json unchanged.""",
    )
    p.add_argument(
        "--no-profile",
        dest="no_profile",
        action="store_true",
        help="""Build: skip the constraint profile. The splitter reads the --input-mode text directly. Writes a separate run directory whose name ends in _noprofile.""",
    )
    p.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Build: name of the run directory under analysis/runs/. Required.",
    )
    p.add_argument(
        "--tree-model",
        dest="tree_model",
        default=None,
        help="Model for tree building. Required for build.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=f"""Smoke-test the stage on the first {TEST_N} train cases and write artifacts under doctrines/<d>/analysis/test/.""",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the resolved config and call plan without hitting any API.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.stage == "build" and not args.tree_model:
        parser.error("--tree-model is required for the build stage")
    if args.stage == "build" and not args.run_id:
        parser.error("--run-id is required for the build stage")
    if args.no_profile and args.stage != "build":
        parser.error("--no-profile is only valid with --stage build")
    if args.input_mode != "segmented" and args.stage != "build":
        parser.error("--input-mode is only valid with --stage build")
    if args.dry_run:
        print("[run] DRY RUN — no API calls will be made")
        print(f"  stage           : {args.stage}")
        print(f"  doctrines       : {args.doctrine}")
        print(f"  segmenter-model : {args.segmenter_model or 'from config'}")
        print(f"  tree-model      : {args.tree_model}")
        print(f"  input-mode      : {args.input_mode}")
        print(f"  profile         : {'off' if args.no_profile else 'on'}")
        print(
            f"  test            : {args.test}"
            + (f" (first {TEST_N} cases)" if args.test else "")
        )

    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
