from __future__ import annotations

from pathlib import Path
import json

from .schemas import ASSERTION_TYPES, ENTITY_TYPES


class ValidationError(Exception):
    pass


def validate_entity_dict(raw_text: str, entity: dict, file_name: str) -> None:
    required_keys = {"text", "position", "type", "assertions", "candidates"}
    if set(entity) != required_keys:
        raise ValidationError(f"{file_name}: keys mismatch {set(entity)}")
    if entity["type"] not in ENTITY_TYPES:
        raise ValidationError(f"{file_name}: invalid type {entity['type']}")
    if not isinstance(entity["position"], list) or len(entity["position"]) != 2:
        raise ValidationError(f"{file_name}: invalid position {entity['position']}")
    start, end = entity["position"]
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(raw_text):
        raise ValidationError(f"{file_name}: invalid position bounds {entity['position']}")
    if raw_text[start:end] != entity["text"]:
        raise ValidationError(f"{file_name}: text/position mismatch for {entity['text']!r}")
    if not isinstance(entity["assertions"], list) or any(item not in ASSERTION_TYPES for item in entity["assertions"]):
        raise ValidationError(f"{file_name}: invalid assertions {entity['assertions']}")
    if entity["type"] not in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"} and entity["assertions"]:
        raise ValidationError(f"{file_name}: assertions not allowed for type {entity['type']}")
    if not isinstance(entity["candidates"], list) or any(not isinstance(item, str) for item in entity["candidates"]):
        raise ValidationError(f"{file_name}: invalid candidates {entity['candidates']}")
    if entity["type"] not in {"CHẨN_ĐOÁN", "THUỐC"} and entity["candidates"]:
        raise ValidationError(f"{file_name}: candidates not allowed for type {entity['type']}")


def validate_output_directory(input_dir: str | Path, output_dir: str | Path) -> list[Path]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    input_files = sorted(input_root.glob("*.txt"), key=lambda path: path.name)
    output_files = sorted(output_root.glob("*.json"), key=lambda path: path.name)
    if len(input_files) != len(output_files):
        raise ValidationError(
            f"output count mismatch: expected {len(input_files)} files, found {len(output_files)}"
        )
    validated_files: list[Path] = []
    for input_path in input_files:
        output_path = output_root / f"{input_path.stem}.json"
        if not output_path.exists():
            raise ValidationError(f"missing output file {output_path.name}")
        raw_text = input_path.read_text(encoding="utf-8")
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValidationError(f"{output_path.name}: root JSON must be a list")
        for entity in data:
            if not isinstance(entity, dict):
                raise ValidationError(f"{output_path.name}: entity must be a dict")
            validate_entity_dict(raw_text, entity, output_path.name)
        validated_files.append(output_path)
    return validated_files
