from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validation import validate_output_directory


def export_gold_dir_to_ner_jsonl(
    input_dir: str | Path,
    gold_dir: str | Path,
    output_path: str | Path,
) -> Path:
    validate_output_directory(input_dir, gold_dir)
    input_root = Path(input_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for input_path in sorted(input_root.glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        gold_path = Path(gold_dir) / f"{input_path.stem}.json"
        entities = json.loads(gold_path.read_text(encoding="utf-8"))
        payload = {
            "text": raw_text,
            "entities": [
                {
                    "position": entity["position"],
                    "type": entity["type"],
                }
                for entity in entities
            ],
        }
        lines.append(json.dumps(payload, ensure_ascii=False))

    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert gold_dir JSON files into NER training JSONL.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--gold_dir", required=True)
    parser.add_argument("--output_path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = export_gold_dir_to_ner_jsonl(args.input_dir, args.gold_dir, args.output_path)
    print(f"NER training JSONL exported to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
