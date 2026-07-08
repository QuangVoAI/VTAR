from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .validation import validate_output_directory


@dataclass(slots=True)
class UnmatchedRecord:
    file_name: str
    text: str
    entity_type: str
    position: list[int]
    assertions: list[str]
    context: str


def _build_context(raw_text: str, start: int, end: int, radius: int) -> str:
    left = max(0, start - radius)
    right = min(len(raw_text), end + radius)
    snippet = raw_text[left:right].replace("\n", " ")
    return " ".join(snippet.split())


def collect_unmatched_records(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    entity_types: set[str] | None = None,
    context_radius: int = 80,
) -> list[UnmatchedRecord]:
    validate_output_directory(input_dir, output_dir)
    records: list[UnmatchedRecord] = []
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    for input_path in sorted(input_root.glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        output_path = output_root / f"{input_path.stem}.json"
        data = json.loads(output_path.read_text(encoding="utf-8"))
        for item in data:
            entity_type = str(item["type"])
            if entity_types and entity_type not in entity_types:
                continue
            if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
                continue
            if item["candidates"]:
                continue
            start, end = item["position"]
            records.append(
                UnmatchedRecord(
                    file_name=output_path.name,
                    text=str(item["text"]),
                    entity_type=entity_type,
                    position=[int(start), int(end)],
                    assertions=[str(value) for value in item["assertions"]],
                    context=_build_context(raw_text, int(start), int(end), context_radius),
                )
            )
    return records


def summarize_unmatched(records: list[UnmatchedRecord]) -> dict:
    by_type = Counter(record.entity_type for record in records)
    by_text = Counter((record.entity_type, record.text) for record in records)
    return {
        "total": len(records),
        "by_type": dict(by_type),
        "top_texts": [
            {"type": entity_type, "text": text, "count": count}
            for (entity_type, text), count in by_text.most_common(25)
        ],
        "records": [asdict(record) for record in records],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze unmatched ICD-10/RxNorm candidates from Viettel-style outputs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--entity_type", choices=["CHẨN_ĐOÁN", "THUỐC", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--context_radius", type=int, default=80)
    parser.add_argument("--output_path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    entity_types = None if args.entity_type == "all" else {args.entity_type}
    records = collect_unmatched_records(
        args.input_dir,
        args.output_dir,
        entity_types=entity_types,
        context_radius=args.context_radius,
    )
    payload = summarize_unmatched(records)
    payload["records"] = payload["records"][: args.limit]
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
