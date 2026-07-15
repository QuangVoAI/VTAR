from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from vtr_ai.build_standard_kb import (
    build_icd10_records_from_zip,
    build_records_from_xlsx,
    build_rxnorm_records_from_zip,
    merge_seed_aliases,
    write_records,
)


class BuildStandardKbTests(unittest.TestCase):
    @staticmethod
    def _rrf_row(rxcui: str, tty: str, code: str, value: str, suppress: str = "N") -> str:
        fields = [""] * 18
        fields[0] = rxcui
        fields[1] = "ENG"
        fields[12] = tty
        fields[13] = code
        fields[14] = value
        fields[16] = suppress
        return "|".join(fields) + "|\n"

    def test_build_icd10_records_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "icd10.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "icd10cm-codes-2026.txt",
                    "I10 Hypertension\nE119 Type 2 diabetes mellitus without complications\n",
                )
            records = build_icd10_records_from_zip(zip_path)
            self.assertEqual(records[0]["code"], "I10")
            self.assertEqual(records[0]["label"], "Hypertension")
            self.assertEqual(records[1]["code"], "E11.9")

    def test_build_rxnorm_records_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "rxnorm.zip"
            row_1 = self._rrf_row("1191", "SCD", "1191", "aspirin 325 MG Oral Tablet")
            row_2 = self._rrf_row("1191", "SY", "1191", "aspirin 325 MG")
            row_3 = self._rrf_row("6918", "IN", "6918", "atenolol")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("rrf/RXNCONSO.RRF", row_1 + row_2 + row_3)
            records = build_rxnorm_records_from_zip(zip_path)
            by_code = {item["code"]: item for item in records}
            self.assertIn("1191", by_code)
            self.assertEqual(by_code["1191"]["label"], "aspirin 325 MG Oral Tablet")
            self.assertIn("aspirin 325 MG", by_code["1191"]["aliases"])
            self.assertEqual(by_code["6918"]["label"], "atenolol")

    def test_build_rxnorm_records_keeps_tty_and_ingredient_relation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "rxnorm-rel.zip"
            ingredient = self._rrf_row("100", "IN", "100", "aspirin")
            product = self._rrf_row("200", "SCD", "200", "aspirin 325 MG Oral Tablet")
            relation_fields = ["200", "", "", "", "100", "", "", "has_ingredient"]
            relation = "|".join(relation_fields + [""] * 8) + "|\n"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("rrf/RXNCONSO.RRF", ingredient + product)
                archive.writestr("rrf/RXNREL.RRF", relation)
            records = build_rxnorm_records_from_zip(zip_path)
            by_code = {item["code"]: item for item in records}
            self.assertEqual(by_code["100"]["tty"], "IN")
            self.assertEqual(by_code["200"]["tty"], "SCD")
            self.assertEqual(by_code["200"]["ingredient_codes"], ["100"])

    def test_build_records_from_xlsx_groups_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            xlsx_path = Path(tmp_dir) / "records.xlsx"
            workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c t="inlineStr"><is><t>Ma</t></is></c><c t="inlineStr"><is><t>Ten</t></is></c></row>
    <row r="2"><c t="inlineStr"><is><t>E119</t></is></c><c t="inlineStr"><is><t>Type 2 diabetes</t></is></c></row>
    <row r="3"><c t="inlineStr"><is><t>E119</t></is></c><c t="inlineStr"><is><t>Diabetes type 2</t></is></c></row>
  </sheetData>
</workbook>"""
            with zipfile.ZipFile(xlsx_path, "w") as archive:
                archive.writestr("xl/worksheets/sheet1.xml", workbook_xml)
            records = build_records_from_xlsx(xlsx_path, kind="icd10")
            self.assertEqual(records, [{
                "code": "E11.9",
                "label": "Type 2 diabetes",
                "aliases": ["Diabetes type 2"],
            }])

    def test_write_records_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "records.json"
            written = write_records([{"code": "I10", "label": "Hypertension", "aliases": []}], output_path)
            self.assertEqual(written, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["code"], "I10")

    def test_merge_seed_aliases_keeps_standard_label_and_adds_local_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            seed_path = tmp_path / "seed.json"
            seed_path.write_text(
                json.dumps(
                    [
                        {"code": "I10", "label": "Tăng huyết áp", "aliases": ["tha"]},
                        {"code": "E11.9", "label": "ĐTĐ típ 2", "aliases": ["đái tháo đường"]},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            merged = merge_seed_aliases(
                [{"code": "I10", "label": "Essential (primary) hypertension", "aliases": []}],
                seed_path,
            )
            by_code = {item["code"]: item for item in merged}
            self.assertEqual(by_code["I10"]["label"], "Essential (primary) hypertension")
            self.assertIn("Tăng huyết áp", by_code["I10"]["aliases"])
            self.assertIn("tha", by_code["I10"]["aliases"])
            self.assertIn("E11.9", by_code)


if __name__ == "__main__":
    unittest.main()
