from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .validation import ValidationError, validate_output_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict sanity checks for a Viettel subset output directory such as raw 68-100."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_path")
    return parser


def sanity_check(input_dir: str | Path, output_dir: str | Path) -> dict:
    validated = validate_output_directory(input_dir, output_dir)
    duplicate_candidates = 0
    empty_files: list[str] = []
    suspicious_empty_semantic_files: list[str] = []
    by_type = Counter()

    for output_path in validated:
        rows = json.loads(output_path.read_text(encoding="utf-8"))
        if not rows:
            empty_files.append(output_path.name)
            continue
        semantic_rows = [row for row in rows if row["type"] in {"CHẨN_ĐOÁN", "THUỐC"}]
        if semantic_rows and all(not row["candidates"] for row in semantic_rows):
            suspicious_empty_semantic_files.append(output_path.name)
        for row in rows:
            by_type[row["type"]] += 1
            candidates = [str(item) for item in row.get("candidates", [])]
            if len(candidates) != len(set(candidates)):
                duplicate_candidates += 1

    return {
        "validated_files": len(validated),
        "entity_type_counts": dict(by_type),
        "duplicate_candidate_entities": duplicate_candidates,
        "empty_files": empty_files,
        "suspicious_empty_semantic_files": suspicious_empty_semantic_files,
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = sanity_check(args.input_dir, args.output_dir)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
