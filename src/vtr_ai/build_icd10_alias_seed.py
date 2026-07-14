from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _clean_alias(value: object) -> str:
    alias = re.sub(r"\s+", " ", str(value)).strip()
    if not alias or len(alias) > 80:
        return ""
    return alias


def build_alias_seed(
    annotations_path: str | Path,
    kb_path: str | Path,
    *,
    allowed_codes_path: str | Path | None = None,
) -> list[dict[str, object]]:
    kb_records = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    kb_codes = {str(item["code"]): item for item in kb_records}
    if allowed_codes_path:
        allowed_records = json.loads(Path(allowed_codes_path).read_text(encoding="utf-8"))
        allowed_codes = {str(item["code"]) for item in allowed_records}
    else:
        allowed_codes = set(kb_codes)

    aliases_by_code: dict[str, set[str]] = defaultdict(set)
    lines = Path(annotations_path).read_text(encoding="utf-8").splitlines()
    used_gold = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        gold_entities = payload.get("gold_entities")
        if not isinstance(gold_entities, list):
            raise ValueError(
                f"{annotations_path}:{line_number} has no gold_entities; refusing to learn aliases from predictions"
            )
        used_gold += 1
        for entity in gold_entities:
            if entity.get("type") != "CHẨN_ĐOÁN":
                continue
            alias = _clean_alias(entity.get("text", ""))
            if not alias:
                continue
            for raw_code in entity.get("candidates", []):
                code = str(raw_code).split(":", 1)[0].strip()
                if code in kb_codes and code in allowed_codes:
                    aliases_by_code[code].add(alias)

    result: list[dict[str, object]] = []
    for code in sorted(aliases_by_code):
        record = kb_codes[code]
        existing = {str(alias) for alias in record.get("aliases", [])}
        aliases = sorted(alias for alias in aliases_by_code[code] | existing if alias != record["label"])
        result.append({"code": code, "label": record["label"], "aliases": aliases})
    if used_gold == 0:
        raise ValueError("No approved gold_entities records found")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a conservative ICD-10 alias seed from approved annotations.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--allowed-codes")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = build_alias_seed(args.annotations, args.kb, allowed_codes_path=args.allowed_codes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "records": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
