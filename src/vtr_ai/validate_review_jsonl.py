from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import ASSERTION_TYPES, ENTITY_TYPES


class ReviewValidationError(Exception):
    pass


def validate_review_entity(raw_text: str, entity: dict, file_name: str) -> None:
    required_keys = {"text", "position", "type", "assertions", "candidates"}
    if set(entity) != required_keys:
        raise ReviewValidationError(f"{file_name}: keys mismatch {set(entity)}")
    if entity["type"] not in ENTITY_TYPES:
        raise ReviewValidationError(f"{file_name}: invalid type {entity['type']}")
    if not isinstance(entity["position"], list) or len(entity["position"]) != 2:
        raise ReviewValidationError(f"{file_name}: invalid position {entity['position']}")
    start, end = entity["position"]
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(raw_text):
        raise ReviewValidationError(f"{file_name}: invalid position bounds {entity['position']}")
    if raw_text[start:end] != entity["text"]:
        raise ReviewValidationError(f"{file_name}: text/position mismatch for {entity['text']!r}")
    if not isinstance(entity["assertions"], list) or any(item not in ASSERTION_TYPES for item in entity["assertions"]):
        raise ReviewValidationError(f"{file_name}: invalid assertions {entity['assertions']}")
    if entity["type"] not in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"} and entity["assertions"]:
        raise ReviewValidationError(f"{file_name}: assertions not allowed for type {entity['type']}")
    if not isinstance(entity["candidates"], list) or any(not isinstance(item, str) for item in entity["candidates"]):
        raise ReviewValidationError(f"{file_name}: invalid candidates {entity['candidates']}")
    if entity["type"] not in {"CHẨN_ĐOÁN", "THUỐC"} and entity["candidates"]:
        raise ReviewValidationError(f"{file_name}: candidates not allowed for type {entity['type']}")


def validate_review_jsonl(path: str | Path) -> int:
    count = 0
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        file_name = str(record.get("file_name", f"<line {line_number}>"))
        raw_text = str(record.get("text", ""))
        for field_name in ("predicted_entities", "gold_entities"):
            if field_name not in record:
                continue
            entities = record[field_name]
            if not isinstance(entities, list):
                raise ReviewValidationError(f"{file_name}: {field_name} must be a list")
            for entity in entities:
                if not isinstance(entity, dict):
                    raise ReviewValidationError(f"{file_name}: entity in {field_name} must be a dict")
                validate_review_entity(raw_text, entity, file_name)
        count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate reviewed bootstrap JSONL before converting to gold/train.")
    parser.add_argument("--review_jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    count = validate_review_jsonl(args.review_jsonl)
    print(f"Validated {count} review records from {Path(args.review_jsonl).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
