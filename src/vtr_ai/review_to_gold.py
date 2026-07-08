from __future__ import annotations

import argparse
import json
from pathlib import Path


def _select_entities(record: dict) -> list[dict]:
    if "gold_entities" in record:
        entities = record["gold_entities"]
    else:
        entities = record.get("predicted_entities", [])
    if not isinstance(entities, list):
        raise ValueError("entities payload must be a list")
    return entities


def export_review_jsonl_to_gold_dir(
    review_jsonl_path: str | Path,
    output_dir: str | Path,
) -> Path:
    source = Path(review_jsonl_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        file_name = str(record["file_name"])
        stem = Path(file_name).stem
        entities = _select_entities(record)
        output_path = destination / f"{stem}.json"
        output_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert reviewed bootstrap JSONL into gold_dir JSON files.")
    parser.add_argument("--review_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = export_review_jsonl_to_gold_dir(args.review_jsonl, args.output_dir)
    print(f"Gold directory exported to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
