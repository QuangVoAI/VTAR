from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait for a Modal subset output directory to reach the expected file count, then download and analyze it."
    )
    parser.add_argument("--volume_name", required=True)
    parser.add_argument("--remote_subdir", required=True)
    parser.add_argument("--expected_files", type=int, required=True)
    parser.add_argument("--local_output_root", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--span_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--max_polls", type=int, default=120)
    parser.add_argument("--analysis_report_path", required=True)
    parser.add_argument("--sanity_report_path", required=True)
    return parser


def _run(command: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)


def _remote_file_count(volume_name: str, remote_subdir: str) -> int:
    result = _run(["modal", "volume", "ls", volume_name, f"/{remote_subdir}", "--json"])
    payload = json.loads(result.stdout)
    return sum(1 for row in payload if row.get("type") == "file" and str(row.get("filename", "")).endswith(".json"))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    count = 0
    for _ in range(args.max_polls):
        count = _remote_file_count(args.volume_name, args.remote_subdir)
        print(json.dumps({"remote_subdir": args.remote_subdir, "count": count}, ensure_ascii=False))
        if count >= args.expected_files:
            break
        time.sleep(args.poll_seconds)

    if count < args.expected_files:
        raise RuntimeError(
            f"Remote directory {args.remote_subdir} only has {count}/{args.expected_files} json files after polling."
        )

    local_root = Path(args.local_output_root)
    local_root.mkdir(parents=True, exist_ok=True)
    _run(["modal", "volume", "get", args.volume_name, args.remote_subdir, str(local_root)])

    downloaded_dir = local_root / args.remote_subdir
    if not downloaded_dir.exists():
        raise RuntimeError(f"Downloaded output directory not found: {downloaded_dir}")

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    subprocess.run(
        [
            "python3",
            "-m",
            "vtr_ai.sanity_check_subset_output",
            "--input_dir",
            args.input_dir,
            "--output_dir",
            str(downloaded_dir),
            "--report_path",
            args.sanity_report_path,
        ],
        check=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
    subprocess.run(
        [
            "python3",
            "-m",
            "vtr_ai.analyze_fixed_span_subset",
            "--input_dir",
            args.input_dir,
            "--span_dir",
            args.span_dir,
            "--output_dir",
            str(downloaded_dir),
            "--config",
            args.config,
            "--report_path",
            args.analysis_report_path,
            "--top_k",
            "30",
        ],
        check=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "downloaded_dir": str(downloaded_dir.resolve()),
                "sanity_report_path": str(Path(args.sanity_report_path).resolve()),
                "analysis_report_path": str(Path(args.analysis_report_path).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
