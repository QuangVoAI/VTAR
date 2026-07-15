from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


DEFAULT_ICD10_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/"
    "icd10cm-Code%20Descriptions-2026.zip"
)
DEFAULT_RXNORM_URL = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_prescribe_current.zip"
RXNORM_TTYS = {"IN", "PIN", "BN", "SCD", "SBD", "SCDG", "SBDG", "GPCK", "BPCK"}
XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


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
        rel_name = next((name for name in archive.namelist() if name.endswith("RXNREL.RRF")), None)
        relation_rows = (
            archive.read(rel_name).decode("utf-8", errors="replace").splitlines()
            if rel_name is not None
            else []
        )

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
                "ingredient_codes": set(),
            },
        )
        grouped[rxcui]["label"] = _prefer_label(str(grouped[rxcui]["label"]), normalized_alias, tty)

    # RXNREL links clinical/ branded products to ingredient concepts without
    # relying on challenge-specific code lists. Keep only the generic
    # ingredient relation and normalize both directions for release variants.
    for line in relation_rows:
        fields = line.split("|")
        if len(fields) < 8:
            continue
        left = fields[0].strip()
        relation = fields[7].strip().lower()
        right = fields[4].strip()
        if not left or not right:
            continue
        if relation in {"has_ingredient", "has_precise_ingredient"}:
            if left in grouped and right in grouped:
                grouped[left]["ingredient_codes"].add(right)
        elif relation in {"ingredient_of", "precise_ingredient_of"}:
            if right in grouped and left in grouped:
                grouped[right]["ingredient_codes"].add(left)

    records: list[dict[str, object]] = []
    for rxcui, payload in grouped.items():
        aliases = sorted(alias for alias in aliases_by_rxcui[rxcui] if alias != payload["label"])
        records.append(
            {
                "code": payload["code"],
                "label": payload["label"],
                "aliases": aliases,
                "tty": payload["tty"],
                "ingredient_codes": sorted(payload["ingredient_codes"]),
            }
        )
    records.sort(key=lambda item: item["code"])
    return records


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    namespace = f"{{{XLSX_NS}}}"
    return ["".join(node.text or "" for node in item.iter(namespace + "t")) for item in root.findall(namespace + "si")]


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    namespace = f"{{{XLSX_NS}}}"
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(namespace + "t")).strip()
    value = cell.find(namespace + "v")
    raw = "" if value is None else (value.text or "").strip()
    if cell_type == "s" and raw.isdigit() and int(raw) < len(shared_strings):
        return shared_strings[int(raw)].strip()
    return raw


def _iter_xlsx_code_labels(xlsx_path: str | Path):
    namespace = f"{{{XLSX_NS}}}"
    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != namespace + "row":
                    continue
                cells = row.findall(namespace + "c")
                if len(cells) >= 2 and row.attrib.get("r") != "1":
                    code = _xlsx_cell_value(cells[0], shared_strings)
                    label = _xlsx_cell_value(cells[1], shared_strings)
                    if code and label:
                        yield code, label
                row.clear()


def build_records_from_xlsx(xlsx_path: str | Path, *, kind: str) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for raw_code, label in _iter_xlsx_code_labels(xlsx_path):
        code = _format_icd10cm_code(raw_code) if kind == "icd10" else raw_code.strip()
        current = grouped.get(code)
        if current is None:
            grouped[code] = {"code": code, "label": label.strip(), "aliases": []}
            continue
        if label.strip() != current["label"] and label.strip() not in current["aliases"]:
            current["aliases"].append(label.strip())
    records = list(grouped.values())
    for record in records:
        record["aliases"] = sorted(str(alias) for alias in record["aliases"] if alias != record["label"])
    records.sort(key=lambda item: str(item["code"]))
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
                **({key: record[key] for key in ("tty", "ingredient_codes") if key in record}),
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
                **({key: seed[key] for key in ("tty", "ingredient_codes") if key in seed}),
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
    parser.add_argument("--icd10_xlsx", help="ICD workbook with columns Mã and Tên bệnh")
    parser.add_argument("--rxnorm_xlsx", help="RxNorm workbook with columns Mã and Tên thuốc")
    parser.add_argument("--icd10_output", default="src/vtr_ai/data/icd10_standard.json")
    parser.add_argument("--rxnorm_output", default="src/vtr_ai/data/rxnorm_standard.json")
    parser.add_argument("--icd10_seed_aliases", help="Optional existing ICD JSON to merge in local aliases")
    parser.add_argument("--rxnorm_seed_aliases", help="Optional existing RxNorm JSON to merge in local aliases")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not any((args.icd10_zip, args.rxnorm_zip, args.icd10_xlsx, args.rxnorm_xlsx)):
        parser.error("Provide at least one zip or xlsx source")

    result: dict[str, object] = {}
    if args.icd10_zip or args.icd10_xlsx:
        icd_records = (
            build_icd10_records_from_zip(args.icd10_zip)
            if args.icd10_zip
            else build_records_from_xlsx(args.icd10_xlsx, kind="icd10")
        )
        icd_records = merge_seed_aliases(icd_records, args.icd10_seed_aliases)
        write_records(icd_records, args.icd10_output)
        result["icd10_output"] = str(Path(args.icd10_output).resolve())
        result["icd10_records"] = len(icd_records)

    if args.rxnorm_zip or args.rxnorm_xlsx:
        rx_records = (
            build_rxnorm_records_from_zip(args.rxnorm_zip)
            if args.rxnorm_zip
            else build_records_from_xlsx(args.rxnorm_xlsx, kind="rxnorm")
        )
        rx_records = merge_seed_aliases(rx_records, args.rxnorm_seed_aliases)
        write_records(rx_records, args.rxnorm_output)
        result["rxnorm_output"] = str(Path(args.rxnorm_output).resolve())
        result["rxnorm_records"] = len(rx_records)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
