from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .validation import ValidationError, validate_entity_dict


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remap split 68.1/68.2 style span JSON back onto raw 68.txt/69.txt style files."
    )
    parser.add_argument("--raw_input_dir", required=True)
    parser.add_argument("--split_input_dir", required=True)
    parser.add_argument("--split_json_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path")
    return parser


def _split_sort_key(path: Path) -> tuple[int, int]:
    stem_parts = path.stem.split(".")
    root = int(stem_parts[0])
    chunk = int(stem_parts[1]) if len(stem_parts) > 1 and stem_parts[1].isdigit() else 0
    return root, chunk


def _find_segment_offset(raw_text: str, segment_text: str, cursor: int) -> int:
    direct = raw_text.find(segment_text, cursor)
    if direct != -1:
        return direct

    all_hits: list[int] = []
    start = 0
    while True:
        hit = raw_text.find(segment_text, start)
        if hit == -1:
            break
        all_hits.append(hit)
        start = hit + 1

    if len(all_hits) == 1:
        return all_hits[0]

    if not all_hits:
        raise ValidationError("segment text was not found in raw source")

    raise ValidationError(f"segment text matched multiple locations in raw source: {all_hits[:5]}")


def remap_split_spans_to_raw(
    raw_input_dir: str | Path,
    split_input_dir: str | Path,
    split_json_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> list[dict]:
    raw_root = Path(raw_input_dir)
    split_input_root = Path(split_input_dir)
    split_json_root = Path(split_json_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    grouped_inputs: dict[str, list[Path]] = defaultdict(list)
    for split_path in sorted(split_input_root.glob("*.txt"), key=_split_sort_key):
        grouped_inputs[split_path.stem.split(".")[0]].append(split_path)

    manifest: list[dict] = []
    for root_name, split_paths in sorted(grouped_inputs.items(), key=lambda item: int(item[0])):
        raw_path = raw_root / f"{root_name}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"missing raw input file {raw_path.name}")

        raw_text = raw_path.read_text(encoding="utf-8")
        merged_entities: list[dict] = []
        cursor = 0

        for split_path in split_paths:
            split_text = split_path.read_text(encoding="utf-8")
            split_offset = _find_segment_offset(raw_text, split_text, cursor)
            cursor = split_offset + len(split_text)

            split_json_path = split_json_root / f"{split_path.stem}.json"
            if not split_json_path.exists():
                raise FileNotFoundError(f"missing split json file {split_json_path.name}")
            entities = json.loads(split_json_path.read_text(encoding="utf-8"))
            if not isinstance(entities, list):
                raise ValidationError(f"{split_json_path.name}: root JSON must be a list")

            manifest.append(
                {
                    "raw_file": raw_path.name,
                    "split_file": split_path.name,
                    "offset": split_offset,
                    "length": len(split_text),
                }
            )

            for entity in entities:
                if not isinstance(entity, dict):
                    raise ValidationError(f"{split_json_path.name}: entity must be a dict")
                start, end = [int(value) for value in entity["position"]]
                remapped = {
                    "text": str(entity["text"]),
                    "position": [start + split_offset, end + split_offset],
                    "type": str(entity["type"]),
                    "assertions": list(entity.get("assertions", [])),
                    "candidates": list(entity.get("candidates", [])),
                }
                validate_entity_dict(raw_text, remapped, raw_path.name)
                merged_entities.append(remapped)

        merged_entities.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"], item["text"]))
        (output_root / f"{root_name}.json").write_text(
            json.dumps(merged_entities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if manifest_path:
        Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    remap_split_spans_to_raw(
        raw_input_dir=args.raw_input_dir,
        split_input_dir=args.split_input_dir,
        split_json_dir=args.split_json_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
    )
    print(f"Remapped raw span JSON written to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
