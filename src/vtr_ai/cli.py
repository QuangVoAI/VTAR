from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io_utils import read_txt_files, write_entities
from .knowledge_base import load_knowledge_base
from .pipeline import ClinicalNlpPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline clinical concept extraction pipeline.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
    pipeline = ClinicalNlpPipeline(config, kb)
    input_files = read_txt_files(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in input_files:
        entities = pipeline.process_text(input_path.read_text(encoding="utf-8"))
        output_path = output_dir / f"{input_path.stem}.json"
        write_entities(output_path, entities, config.output.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

