from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize raw *.txt files from bootstrap_annotations.jsonl for a numeric file-id range."
    )
    parser.add_argument("--bootstrap_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    return parser


def materialize_bootstrap_raw_subset(
    bootstrap_jsonl: str | Path,
    output_dir: str | Path,
    start_id: int,
    end_id: int,
) -> int:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
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
        if start_id <= numeric_id <= end_id:
            (output_root / file_name).write_text(str(row["text"]), encoding="utf-8")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    written = materialize_bootstrap_raw_subset(
        bootstrap_jsonl=args.bootstrap_jsonl,
        output_dir=args.output_dir,
        start_id=args.start_id,
        end_id=args.end_id,
    )
    print(f"Wrote {written} raw input files to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
