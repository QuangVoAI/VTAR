from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gold_to_ner_jsonl import export_gold_dir_to_ner_jsonl
from .review_to_gold import export_review_jsonl_to_gold_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert reviewed JSONL into gold_dir and NER training JSONL.")
    parser.add_argument("--review_jsonl", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--gold_dir", required=True)
    parser.add_argument("--ner_output_path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    gold_dir = export_review_jsonl_to_gold_dir(args.review_jsonl, args.gold_dir)
    ner_jsonl = export_gold_dir_to_ner_jsonl(args.input_dir, gold_dir, args.ner_output_path)
    print(
        json.dumps(
            {
                "gold_dir": str(Path(gold_dir).resolve()),
                "ner_jsonl": str(Path(ner_jsonl).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
