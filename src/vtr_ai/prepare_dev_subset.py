from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap_annotations import export_bootstrap_annotations
from .review_subset import rank_files_for_review
from .select_review_subset import export_selected_review_subset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a seeded dev-review subset from current outputs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_path = report_dir / "bootstrap_seeded.jsonl"
    priority_path = report_dir / "review_priority.json"
    subset_path = report_dir / "dev_subset.jsonl"

    export_bootstrap_annotations(
        args.input_dir,
        args.output_dir,
        bootstrap_path,
        seed_gold_entities=True,
    )

    ranked = rank_files_for_review(args.input_dir, args.output_dir)
    priority_payload = {
        "total_files": len(ranked),
        "top_files": ranked[: args.limit],
    }
    priority_path.write_text(json.dumps(priority_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    export_selected_review_subset(
        bootstrap_path,
        priority_path,
        subset_path,
        limit=args.limit,
    )

    print(
        json.dumps(
            {
                "bootstrap_seeded_path": str(bootstrap_path.resolve()),
                "review_priority_path": str(priority_path.resolve()),
                "dev_subset_path": str(subset_path.resolve()),
                "limit": args.limit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
