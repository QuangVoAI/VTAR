from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from vtr_ai.cli import main
from vtr_ai.config import load_config
from vtr_ai.knowledge_base import load_knowledge_base
from vtr_ai.pipeline import ClinicalNlpPipeline
from vtr_ai.preprocess import normalize_with_mapping


ROOT = Path(__file__).resolve().parents[1]


class PreprocessTests(unittest.TestCase):
    def test_normalize_preserves_offset_lookup(self) -> None:
        text = "  Bệnh   nhân\tkhông ho  "
        normalized, mapping = normalize_with_mapping(text)
        self.assertEqual(normalized, "Bệnh nhân không ho")
        raw_slice = text[mapping[0] : mapping[-1] + 1]
        self.assertIn("Bệnh", raw_slice)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "config.yaml")
        kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
        cls.pipeline = ClinicalNlpPipeline(config, kb)

    def test_pipeline_extracts_core_entities(self) -> None:
        text = (
            "Bệnh nhân nam 70 tuổi ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, "
            "được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. "
            "Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML. "
            "WBC:14,43; NEUT%:76,4;"
        )
        entities = self.pipeline.process_text(text)
        by_type = {}
        for entity in entities:
            by_type.setdefault(entity.entity_type, []).append(entity)

        self.assertTrue(any("trào ngược" in item.text.lower() for item in by_type["CHẨN_ĐOÁN"]))
        self.assertTrue(any(item.candidates for item in by_type["CHẨN_ĐOÁN"]))
        self.assertTrue(any(item.text == "Chlorpheniramine 0.4 MG/ML" for item in by_type["THUỐC"]))
        self.assertTrue(any("isHistorical" in item.assertions for item in by_type["THUỐC"]))
        self.assertTrue(any(item.text == "WBC" for item in by_type["TÊN_XÉT_NGHIỆM"]))
        self.assertTrue(any(item.text == "14,43" for item in by_type["KẾT_QUẢ_XÉT_NGHIỆM"]))

    def test_pipeline_builds_case_relations_and_facts(self) -> None:
        text = "Thuốc trước khi nhập viện: Chlorpheniramine 0.4 MG/ML. WBC:14,43;"
        patient_case = self.pipeline.process_case(text)
        self.assertTrue(any(item.text == "Chlorpheniramine 0.4 MG/ML" for item in patient_case.entities))
        self.assertTrue(any("isHistorical" in item.assertions for item in patient_case.entities if item.entity_type == "THUỐC"))
        self.assertTrue(any(rel.relation_type == "has_result" for rel in patient_case.relations))
        self.assertTrue(any(fact.predicate == "mapped_to_rxnorm" for fact in patient_case.facts))

    def test_negation_detection_is_local(self) -> None:
        text = "Bệnh nhân không ho nhưng đau bụng kéo dài."
        entities = self.pipeline.process_text(text)
        assertion_map = {entity.text: entity.assertions for entity in entities if entity.entity_type == "TRIỆU_CHỨNG"}
        self.assertIn("isNegated", assertion_map["ho"])
        self.assertNotIn("isNegated", assertion_map["đau bụng"])

    def test_cli_writes_json_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            sample = input_dir / "1.txt"
            sample.write_text("Bệnh nhân có tiền sử hen suyễn và dùng Salbutamol 2 MG.", encoding="utf-8")
            exit_code = main(
                [
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--config",
                    str(ROOT / "config.yaml"),
                ]
            )
            self.assertEqual(exit_code, 0)
            output_file = output_dir / "1.json"
            self.assertTrue(output_file.exists())
            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(any(item["type"] == "THUỐC" for item in data))


if __name__ == "__main__":
    unittest.main()
