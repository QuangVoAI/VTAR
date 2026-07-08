from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .analysis_report import collect_unmatched_records, summarize_unmatched
from .validation import validate_output_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate outputs and generate audit summaries/reports.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--context_radius", type=int, default=80)
    parser.add_argument("--limit", type=int, default=50)
    return parser


def build_entity_summary(output_dir: str | Path) -> dict:
    counts = Counter()
    files = sorted(Path(output_dir).glob("*.json"), key=lambda path: path.name)
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            counts[str(item["type"])] += 1
    return {
        "files": len(files),
        "entity_counts": dict(counts),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_output_directory(args.input_dir, args.output_dir)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = build_entity_summary(args.output_dir)
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    diagnosis_records = collect_unmatched_records(
        args.input_dir,
        args.output_dir,
        entity_types={"CHẨN_ĐOÁN"},
        context_radius=args.context_radius,
    )
    drug_records = collect_unmatched_records(
        args.input_dir,
        args.output_dir,
        entity_types={"THUỐC"},
        context_radius=args.context_radius,
    )

    diagnosis_payload = summarize_unmatched(diagnosis_records)
    diagnosis_payload["records"] = diagnosis_payload["records"][: args.limit]
    diagnosis_path = report_dir / "unmatched_diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    drug_payload = summarize_unmatched(drug_records)
    drug_payload["records"] = drug_payload["records"][: args.limit]
    drug_path = report_dir / "unmatched_drugs.json"
    drug_path.write_text(json.dumps(drug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rendered = {
        "summary_path": str(summary_path),
        "unmatched_diagnosis_path": str(diagnosis_path),
        "unmatched_drugs_path": str(drug_path),
        "summary": summary,
        "unmatched_diagnosis_total": diagnosis_payload["total"],
        "unmatched_drugs_total": drug_payload["total"],
    }
    print(json.dumps(rendered, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
