from __future__ import annotations

import argparse
from pathlib import Path

from .audit_report import main as audit_main
from .submission import main as submission_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference, validate, package, and audit in one workflow.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--zip_path", default="output.zip")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--config", required=True)
    parser.add_argument("--keep_existing_output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    submission_args = [
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--zip_path",
        args.zip_path,
        "--config",
        args.config,
    ]
    if args.keep_existing_output:
        submission_args.append("--keep_existing_output")
    submission_main(submission_args)

    audit_main(
        [
            "--input_dir",
            args.input_dir,
            "--output_dir",
            args.output_dir,
            "--report_dir",
            args.report_dir,
        ]
    )
    print(
        f"Workflow completed: output={Path(args.output_dir).resolve()} zip={Path(args.zip_path).resolve()} "
        f"reports={Path(args.report_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
