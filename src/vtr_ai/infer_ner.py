from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .knowledge_base import load_knowledge_base
from .model_ner import load_ner_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone NER inference using the configured model backend.")
    parser.add_argument("--text")
    parser.add_argument("--input_file")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.text and not args.input_file:
        parser.error("Provide either --text or --input_file")
    if args.text and args.input_file:
        parser.error("Use only one of --text or --input_file")

    text = args.text or Path(args.input_file).read_text(encoding="utf-8")
    config = load_config(args.config)
    kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
    runtime = load_ner_runtime(config.ner, kb)
    predictions = runtime.predict(text)
    payload = [
        {
            "text": text[item.start : item.end],
            "position": [item.start, item.end],
            "type": item.label,
            "score": item.score,
        }
        for item in predictions
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
