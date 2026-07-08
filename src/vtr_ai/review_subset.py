from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .validation import validate_output_directory


def rank_files_for_review(
    input_dir: str | Path,
    output_dir: str | Path,
) -> list[dict]:
    validate_output_directory(input_dir, output_dir)
    output_root = Path(output_dir)
    file_stats: list[dict] = []

    for output_path in sorted(output_root.glob("*.json"), key=lambda path: path.name):
        data = json.loads(output_path.read_text(encoding="utf-8"))
        counts = Counter(item["type"] for item in data)
        unmatched = sum(
            1
            for item in data
            if item["type"] in {"CHẨN_ĐOÁN", "THUỐC"} and not item["candidates"]
        )
        assertions = sum(len(item["assertions"]) for item in data)
        type_diversity = sum(1 for count in counts.values() if count > 0)
        score = unmatched * 10 + assertions * 2 + type_diversity + len(data) * 0.1
        file_stats.append(
            {
                "file_name": output_path.name.replace(".json", ".txt"),
                "entities": len(data),
                "unmatched_candidates": unmatched,
                "assertion_count": assertions,
                "type_diversity": type_diversity,
                "type_counts": dict(counts),
                "score": round(score, 3),
            }
        )

    file_stats.sort(
        key=lambda item: (
            -item["unmatched_candidates"],
            -item["assertion_count"],
            -item["type_diversity"],
            -item["entities"],
            item["file_name"],
        )
    )
    return file_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank input files to prioritize manual review for dev/gold creation.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ranked = rank_files_for_review(args.input_dir, args.output_dir)
    payload = {
        "total_files": len(ranked),
        "top_files": ranked[: args.limit],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
