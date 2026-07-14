from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize raw *.txt and mapped span *.json files from bootstrap_annotations.jsonl."
    )
    parser.add_argument("--bootstrap_jsonl", required=True)
    parser.add_argument("--output_text_dir", required=True)
    parser.add_argument("--output_span_dir", required=True)
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--end_id", type=int, default=100)
    return parser


def materialize_bootstrap_dataset(
    bootstrap_jsonl: str | Path,
    output_text_dir: str | Path,
    output_span_dir: str | Path,
    start_id: int = 1,
    end_id: int = 100,
) -> int:
    text_root = Path(output_text_dir)
    span_root = Path(output_span_dir)
    text_root.mkdir(parents=True, exist_ok=True)
    span_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for raw_line in Path(bootstrap_jsonl).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        file_name = str(row["file_name"])
        stem = Path(file_name).stem
        if not stem.isdigit():
            continue
        numeric_id = int(stem)
        if not (start_id <= numeric_id <= end_id):
            continue

        raw_text = str(row.get("text", ""))
        entities = row.get("predicted_entities", [])
        if not isinstance(entities, list):
            entities = []

        (text_root / file_name).write_text(raw_text, encoding="utf-8")
        (span_root / f"{stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    written = materialize_bootstrap_dataset(
        bootstrap_jsonl=args.bootstrap_jsonl,
        output_text_dir=args.output_text_dir,
        output_span_dir=args.output_span_dir,
        start_id=args.start_id,
        end_id=args.end_id,
    )
    print(
        "Materialized "
        f"{written} files into {Path(args.output_text_dir).resolve()} and {Path(args.output_span_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
