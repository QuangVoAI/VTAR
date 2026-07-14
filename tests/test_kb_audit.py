from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vtr_ai.audit_knowledge_base import audit_records
from vtr_ai.build_icd10_alias_seed import build_alias_seed


class KnowledgeBaseAuditTests(unittest.TestCase):
    def test_audit_reports_record_and_alias_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.json"
            path.write_text(
                json.dumps(
                    [
                        {"code": "I10", "label": "Hypertension", "aliases": ["tăng huyết áp"]},
                        {"code": "I11", "label": "Hypertensive heart disease", "aliases": []},
                    ]
                ),
                encoding="utf-8",
            )
            report = audit_records(path, "icd10")
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["records_with_aliases"], 1)
        self.assertEqual(report["duplicate_codes"], [])

    def test_alias_seed_refuses_predictions_without_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.jsonl"
            kb = root / "kb.json"
            annotations.write_text(
                json.dumps({"predicted_entities": [{"type": "CHẨN_ĐOÁN"}]}) + "\n",
                encoding="utf-8",
            )
            kb.write_text(json.dumps([{"code": "I10", "label": "Hypertension", "aliases": []}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_alias_seed(annotations, kb)

    def test_alias_seed_uses_only_approved_gold_entities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.jsonl"
            kb = root / "kb.json"
            annotations.write_text(
                json.dumps(
                    {
                        "gold_entities": [
                            {"type": "CHẨN_ĐOÁN", "text": "tăng huyết áp", "candidates": ["I10"]}
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            kb.write_text(json.dumps([{"code": "I10", "label": "Hypertension", "aliases": []}]), encoding="utf-8")
            records = build_alias_seed(annotations, kb)
        self.assertEqual(records[0]["aliases"], ["tăng huyết áp"])


if __name__ == "__main__":
    unittest.main()
