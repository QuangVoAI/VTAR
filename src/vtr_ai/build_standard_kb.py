from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


DEFAULT_ICD10_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/"
    "icd10cm-Code%20Descriptions-2026.zip"
)
DEFAULT_RXNORM_URL = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_prescribe_current.zip"
RXNORM_TTYS = {"IN", "PIN", "BN", "SCD", "SBD", "SCDG", "SBDG", "GPCK", "BPCK"}


def _format_icd10cm_code(code: str) -> str:
    cleaned = code.strip().upper()
    if len(cleaned) > 3 and "." not in cleaned:
        return f"{cleaned[:3]}.{cleaned[3:]}"
    return cleaned


def build_icd10_records_from_zip(zip_path: str | Path) -> list[dict[str, object]]:
    path = Path(zip_path)
    with zipfile.ZipFile(path) as archive:
        member_name = next(
            (name for name in archive.namelist() if name.lower().startswith("icd10cm-codes-") and name.lower().endswith(".txt")),
            None,
        )
        if member_name is None:
            raise ValueError(f"Could not find ICD-10 code descriptions txt in {path}")
        content = archive.read(member_name).decode("utf-8", errors="replace")

    records: list[dict[str, object]] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        code, label = _format_icd10cm_code(parts[0]), parts[1].strip()
        if not code or not label:
            continue
        records.append({"code": code, "label": label, "aliases": []})
    return records


def build_rxnorm_records_from_zip(zip_path: str | Path) -> list[dict[str, object]]:
    path = Path(zip_path)
    with zipfile.ZipFile(path) as archive:
        conso_name = next((name for name in archive.namelist() if name.endswith("RXNCONSO.RRF")), None)
        if conso_name is None:
            raise ValueError(f"Could not find RXNCONSO.RRF in {path}")
        rows = archive.read(conso_name).decode("utf-8", errors="replace").splitlines()

    grouped: dict[str, dict[str, object]] = {}
    aliases_by_rxcui: dict[str, set[str]] = defaultdict(set)

    for line in rows:
        fields = line.split("|")
        if len(fields) < 17:
            continue
        rxcui = fields[0].strip()
        lat = fields[1].strip()
        tty = fields[12].strip()
        code = fields[13].strip()
        string_value = fields[14].strip()
        suppress = fields[16].strip()
        if lat != "ENG" or suppress != "N":
            continue
        if not rxcui or not code or not string_value:
            continue

        normalized_alias = _normalize_rxnorm_alias(string_value)
        if not normalized_alias:
            continue
        aliases_by_rxcui[rxcui].add(normalized_alias)
        if tty not in RXNORM_TTYS:
            continue
        grouped.setdefault(
            rxcui,
            {
                "code": rxcui,
                "label": normalized_alias,
                "tty": tty,
                "canonical_code": code,
            },
        )
        grouped[rxcui]["label"] = _prefer_label(str(grouped[rxcui]["label"]), normalized_alias, tty)

    records: list[dict[str, object]] = []
    for rxcui, payload in grouped.items():
        aliases = sorted(alias for alias in aliases_by_rxcui[rxcui] if alias != payload["label"])
        records.append(
            {
                "code": payload["code"],
                "label": payload["label"],
                "aliases": aliases,
            }
        )
    records.sort(key=lambda item: item["code"])
    return records


def merge_seed_aliases(
    base_records: list[dict[str, object]],
    seed_path: str | Path | None,
) -> list[dict[str, object]]:
    if seed_path is None:
        return base_records
    seed_records = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    seed_by_code = {str(item["code"]): item for item in seed_records}
    merged: list[dict[str, object]] = []
    covered_codes: set[str] = set()

    for record in base_records:
        code = str(record["code"])
        covered_codes.add(code)
        seed = seed_by_code.get(code)
        aliases = set(record.get("aliases", []))
        label = str(record["label"])
        if seed is not None:
            aliases.update(str(alias) for alias in seed.get("aliases", []))
            seed_label = str(seed.get("label", "")).strip()
            if seed_label and seed_label != label:
                aliases.add(seed_label)
        merged.append(
            {
                "code": code,
                "label": label,
                "aliases": sorted(alias for alias in aliases if alias and alias != label),
            }
        )

    for code, seed in seed_by_code.items():
        if code in covered_codes:
            continue
        merged.append(
            {
                "code": code,
                "label": str(seed["label"]),
                "aliases": sorted(str(alias) for alias in seed.get("aliases", []) if str(alias)),
            }
        )

    merged.sort(key=lambda item: item["code"])
    return merged


def _normalize_rxnorm_alias(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _prefer_label(current: str, candidate: str, tty: str) -> str:
    current_len = len(current)
    candidate_len = len(candidate)
    if tty in {"SCD", "SBD", "IN", "PIN", "BN"} and candidate_len >= current_len:
        return candidate
    if current_len == 0:
        return candidate
    return current


def write_records(records: list[dict[str, object]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build standard ICD-10 / RxNorm knowledge-base files for the pipeline.")
    parser.add_argument("--icd10_zip", help=f"Official ICD-10-CM zip. Suggested source: {DEFAULT_ICD10_URL}")
    parser.add_argument("--rxnorm_zip", help=f"Official RxNorm Prescribable zip. Suggested source: {DEFAULT_RXNORM_URL}")
    parser.add_argument("--icd10_output", default="src/vtr_ai/data/icd10_standard.json")
    parser.add_argument("--rxnorm_output", default="src/vtr_ai/data/rxnorm_standard.json")
    parser.add_argument("--icd10_seed_aliases", help="Optional existing ICD JSON to merge in local aliases")
    parser.add_argument("--rxnorm_seed_aliases", help="Optional existing RxNorm JSON to merge in local aliases")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.icd10_zip and not args.rxnorm_zip:
        parser.error("Provide at least one of --icd10_zip or --rxnorm_zip")

    result: dict[str, object] = {}
    if args.icd10_zip:
        icd_records = build_icd10_records_from_zip(args.icd10_zip)
        icd_records = merge_seed_aliases(icd_records, args.icd10_seed_aliases)
        write_records(icd_records, args.icd10_output)
        result["icd10_output"] = str(Path(args.icd10_output).resolve())
        result["icd10_records"] = len(icd_records)

    if args.rxnorm_zip:
        rx_records = build_rxnorm_records_from_zip(args.rxnorm_zip)
        rx_records = merge_seed_aliases(rx_records, args.rxnorm_seed_aliases)
        write_records(rx_records, args.rxnorm_output)
        result["rxnorm_output"] = str(Path(args.rxnorm_output).resolve())
        result["rxnorm_records"] = len(rx_records)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
