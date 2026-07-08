from __future__ import annotations

import argparse
from pathlib import Path

from .cli import run_inference
from .io_utils import reset_directory, zip_directory
from .validation import ValidationError, validate_output_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end Viettel submission packaging.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--zip_path", default="output.zip")
    parser.add_argument("--config", required=True)
    parser.add_argument("--keep_existing_output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    if not args.keep_existing_output:
        reset_directory(output_dir)
    run_inference(args.input_dir, output_dir, args.config)
    try:
        validate_output_directory(args.input_dir, output_dir)
    except ValidationError as exc:
        parser.error(str(exc))
    zip_directory(output_dir, args.zip_path)
    print(f"Submission package created at {Path(args.zip_path).resolve()}")
    return 0
