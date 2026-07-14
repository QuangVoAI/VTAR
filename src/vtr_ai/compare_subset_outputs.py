from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .validation import validate_output_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two Viettel subset output directories such as baseline vs improved 68-100 outputs."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--report_path")
    parser.add_argument("--top_k", type=int, default=50)
    return parser


def _entity_key(entity: dict) -> tuple[int, int, str, str]:
    start, end = [int(value) for value in entity["position"]]
    return start, end, str(entity["type"]), str(entity["text"])


def compare_outputs(
    input_dir: str | Path,
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    top_k: int,
) -> dict:
    validate_output_directory(input_dir, baseline_dir)
    validate_output_directory(input_dir, candidate_dir)

    changed_candidates = 0
    changed_assertions = 0
    became_nonempty = 0
    became_empty = 0
    multi_code_gain = 0
    by_type = Counter()
    changed_rows: list[dict] = []

    for input_path in sorted(Path(input_dir).glob("*.txt"), key=lambda path: path.name):
        baseline_rows = json.loads((Path(baseline_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        candidate_rows = json.loads((Path(candidate_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        baseline_map = {_entity_key(row): row for row in baseline_rows}
        candidate_map = {_entity_key(row): row for row in candidate_rows}

        for key, baseline_row in baseline_map.items():
            candidate_row = candidate_map[key]
            baseline_candidates = [str(value) for value in baseline_row.get("candidates", [])]
            candidate_candidates = [str(value) for value in candidate_row.get("candidates", [])]
            baseline_assertions = sorted(str(value) for value in baseline_row.get("assertions", []))
            candidate_assertions = sorted(str(value) for value in candidate_row.get("assertions", []))
            entity_type = str(baseline_row["type"])

            reasons: list[str] = []
            if baseline_candidates != candidate_candidates:
                changed_candidates += 1
                by_type[entity_type] += 1
                reasons.append("candidates_changed")
                if not baseline_candidates and candidate_candidates:
                    became_nonempty += 1
                    reasons.append("became_nonempty")
                if baseline_candidates and not candidate_candidates:
                    became_empty += 1
                    reasons.append("became_empty")
                if len(candidate_candidates) > len(baseline_candidates):
                    multi_code_gain += 1
                    reasons.append("candidate_count_increased")
            if baseline_assertions != candidate_assertions:
                changed_assertions += 1
                by_type[entity_type] += 1
                reasons.append("assertions_changed")

            if reasons:
                changed_rows.append(
                    {
                        "file_name": input_path.name,
                        "text": baseline_row["text"],
                        "type": entity_type,
                        "position": baseline_row["position"],
                        "baseline_candidates": baseline_candidates,
                        "candidate_candidates": candidate_candidates,
                        "baseline_assertions": baseline_assertions,
                        "candidate_assertions": candidate_assertions,
                        "reasons": reasons,
                    }
                )

    changed_rows.sort(
        key=lambda row: (
            -len(row["reasons"]),
            row["type"],
            row["file_name"],
            row["position"][0],
        )
    )
    return {
        "baseline_dir": str(Path(baseline_dir).resolve()),
        "candidate_dir": str(Path(candidate_dir).resolve()),
        "changed_candidates": changed_candidates,
        "changed_assertions": changed_assertions,
        "became_nonempty": became_nonempty,
        "became_empty": became_empty,
        "candidate_count_increased": multi_code_gain,
        "changed_by_type": dict(by_type),
        "top_changes": changed_rows[:top_k],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = compare_outputs(
        input_dir=args.input_dir,
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        top_k=args.top_k,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
