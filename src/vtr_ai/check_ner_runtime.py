from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .knowledge_base import load_knowledge_base
from .model_ner import (
    load_ner_runtime,
    resolve_transformer_metadata_path,
    validate_ner_checkpoint,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the configured NER checkpoint/runtime before full inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--text", help="Optional sample text to run a smoke prediction after runtime load")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
    validate_ner_checkpoint(config.ner)
    runtime = load_ner_runtime(config.ner, kb)

    metadata_path = None
    if config.ner.checkpoint_path is not None:
        metadata_path = resolve_transformer_metadata_path(config.ner.checkpoint_path, config.ner.metadata_path)

    payload: dict[str, object] = {
        "backend": config.ner.backend,
        "provider": config.ner.provider,
        "checkpoint_path": str(config.ner.checkpoint_path) if config.ner.checkpoint_path else None,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "runtime_class": runtime.__class__.__name__,
    }

    if args.text:
        payload["predictions"] = [
            {
                "text": args.text[item.start : item.end],
                "position": [item.start, item.end],
                "type": item.label,
                "score": item.score,
            }
            for item in runtime.predict(args.text)
        ]

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
