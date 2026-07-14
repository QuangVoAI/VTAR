from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit_records(path: str | Path, kind: str) -> dict[str, object]:
    source = Path(path)
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{source}: expected a JSON list")

    codes = [str(item.get("code", "")).strip() for item in records]
    labels = [str(item.get("label", "")).strip() for item in records]
    alias_counts = [len(item.get("aliases", [])) for item in records]
    duplicate_codes = sorted(code for code, count in Counter(codes).items() if code and count > 1)
    invalid_rows = [
        index
        for index, item in enumerate(records)
        if not str(item.get("code", "")).strip()
        or not str(item.get("label", "")).strip()
        or not isinstance(item.get("aliases", []), list)
    ]
    numeric_code_ratio = sum(code.isdigit() for code in codes if code) / max(sum(bool(code) for code in codes), 1)
    return {
        "path": str(source.resolve()),
        "kind": kind,
        "records": len(records),
        "unique_codes": len(set(code for code in codes if code)),
        "records_with_aliases": sum(count > 0 for count in alias_counts),
        "total_aliases": sum(alias_counts),
        "average_aliases": round(sum(alias_counts) / max(len(alias_counts), 1), 4),
        "numeric_code_ratio": round(numeric_code_ratio, 4),
        "duplicate_codes": duplicate_codes[:20],
        "invalid_rows": invalid_rows[:20],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ICD-10 and RxNorm JSON knowledge bases before inference.")
    parser.add_argument("--icd10", required=True)
    parser.add_argument("--rxnorm", required=True)
    parser.add_argument("--min-icd10-records", type=int, default=1000)
    parser.add_argument("--min-rxnorm-records", type=int, default=1000)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = {
        "icd10": audit_records(args.icd10, "icd10"),
        "rxnorm": audit_records(args.rxnorm, "rxnorm"),
    }
    failures: list[str] = []
    if payload["icd10"]["records"] < args.min_icd10_records:
        failures.append(f"ICD-10 has only {payload['icd10']['records']} records")
    if payload["rxnorm"]["records"] < args.min_rxnorm_records:
        failures.append(f"RxNorm has only {payload['rxnorm']['records']} records")
    for kind in ("icd10", "rxnorm"):
        if payload[kind]["duplicate_codes"]:
            failures.append(f"{kind} contains duplicate codes")
        if payload[kind]["invalid_rows"]:
            failures.append(f"{kind} contains invalid rows")
    payload["status"] = "error" if failures else "ok"
    payload["failures"] = failures
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
