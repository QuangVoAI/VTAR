from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest
import zipfile

from vtr_ai.cli import main
from vtr_ai.config import load_config
from vtr_ai.io_utils import zip_directory
from vtr_ai.knowledge_base import load_knowledge_base
from vtr_ai.model_ner import (
    load_ner_runtime,
    load_transformer_label_map,
    normalize_transformer_label,
    resolve_transformer_metadata_path,
    validate_ner_checkpoint,
)
from vtr_ai.ner_labels import LABEL_TO_ID
from vtr_ai.ner_training import AnnotatedEntity, align_entities_to_offsets, load_annotated_examples, split_annotated_examples, validate_annotated_example
from vtr_ai.pipeline import ClinicalNlpPipeline
from vtr_ai.preprocess import normalize_with_mapping
from vtr_ai.abbreviation import AbbreviationExpander
from vtr_ai.analysis_report import collect_unmatched_records, summarize_unmatched
from vtr_ai.audit_report import build_entity_summary
from vtr_ai.bootstrap_annotations import export_bootstrap_annotations
from vtr_ai.evaluation import evaluate_output_directories, evaluate_entity_lists
from vtr_ai.entity_recognizer import build_entity_recognizer
from vtr_ai.gold_to_ner_jsonl import export_gold_dir_to_ner_jsonl
from vtr_ai.graph_retriever import GraphRetriever
from vtr_ai.knowledge_graph import KnowledgeGraphIndex
from vtr_ai.knowledge_base import KnowledgeRecord
from vtr_ai.patient_graph import build_patient_graph
from vtr_ai.preprocess import build_document
from vtr_ai.prepare_dev_subset import main as prepare_dev_subset_main
from vtr_ai.review_to_gold import export_review_jsonl_to_gold_dir
from vtr_ai.review_packet import export_review_packet
from vtr_ai.review_subset import rank_files_for_review
from vtr_ai.review_to_train import main as review_to_train_main
from vtr_ai.prepare_semantic_datasets import collect_semantic_examples, export_semantic_dataset_bundle
from vtr_ai.qwen_fixed_span import (
    apply_prediction_to_entity,
    collect_fixed_span_examples,
    export_qwen_fixed_span_dataset,
    parse_qwen_json_response,
)
from vtr_ai.schemas import Entity
from vtr_ai.select_review_subset import export_selected_review_subset
from vtr_ai.submission import main as submission_main
from vtr_ai.validate_review_jsonl import ReviewValidationError, validate_review_jsonl
from vtr_ai.workflow import main as workflow_main
from vtr_ai.validation import validate_output_directory


ROOT = Path(__file__).resolve().parents[1]


def _write_lexical_test_config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "knowledge_base:",
                f"  icd10_path: {ROOT / 'src/vtr_ai/data/icd10_sample.json'}",
                f"  rxnorm_path: {ROOT / 'src/vtr_ai/data/rxnorm_sample.json'}",
                f"  abbreviations_path: {ROOT / 'src/vtr_ai/data/vi_abbreviations.json'}",
                "",
                "matching:",
                "  max_candidates: 3",
                "  min_confidence: 0.55",
                "",
                "rules:",
                "  symptom_window: 80",
                "  assertion_window: 60",
                "",
                "output:",
                "  pretty: true",
                "",
                "ner:",
                "  backend: hybrid",
                "  enable_model_backend: true",
                "  provider: lexical",
                f"  checkpoint_path: {ROOT / 'src/vtr_ai/data/checkpoints/lexical_ner.json'}",
                f"  metadata_path: {ROOT / 'src/vtr_ai/data/checkpoints/transformers_ner_metadata.json'}",
                "  fallback_to_rules: true",
                "  min_score: 0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class PreprocessTests(unittest.TestCase):
    def test_normalize_preserves_offset_lookup(self) -> None:
        text = "  Bệnh   nhân\tkhông ho  "
        normalized, mapping = normalize_with_mapping(text)
        self.assertEqual(normalized, "Bệnh nhân không ho")
        raw_slice = text[mapping[0] : mapping[-1] + 1]
        self.assertIn("Bệnh", raw_slice)

    def test_abbreviation_expansion_preserves_raw_mapping(self) -> None:
        expander = AbbreviationExpander.from_path(ROOT / "src/vtr_ai/data/vi_abbreviations.json")
        expanded, mapping = expander.expand_text_with_mapping("BN bị THA và SOB")
        self.assertIn("tăng huyết áp", expanded.lower())
        self.assertIn("khó thở", expanded.lower())
        self.assertEqual(mapping[expanded.lower().find("tăng")], 6)

    def test_align_entities_to_offsets_builds_bio_labels(self) -> None:
        text = "WBC:14,43"
        offsets = [(0, 3), (3, 4), (4, 9)]
        labels = align_entities_to_offsets(
            text,
            offsets,
            [
                AnnotatedEntity(start=0, end=3, label="TÊN_XÉT_NGHIỆM"),
                AnnotatedEntity(start=4, end=9, label="KẾT_QUẢ_XÉT_NGHIỆM"),
            ],
        )
        self.assertEqual(labels[0], "B-TÊN_XÉT_NGHIỆM")
        self.assertEqual(labels[2], "B-KẾT_QUẢ_XÉT_NGHIỆM")

    def test_load_annotated_examples_from_jsonl(self) -> None:
        examples = load_annotated_examples(ROOT / "src/vtr_ai/data/examples/ner_train_sample.jsonl")
        self.assertGreaterEqual(len(examples), 2)
        validate_annotated_example(examples[0])

    def test_split_annotated_examples_builds_non_empty_dev_split(self) -> None:
        examples = load_annotated_examples(ROOT / "src/vtr_ai/data/examples/ner_train_sample.jsonl")
        train_examples, eval_examples = split_annotated_examples(examples, 0.5, seed=7)
        self.assertGreaterEqual(len(train_examples), 1)
        self.assertGreaterEqual(len(eval_examples), 1)
        self.assertEqual(len(train_examples) + len(eval_examples), len(examples))

    def test_build_document_extracts_section_clause_and_anchor_structure(self) -> None:
        document = build_document(
            "1. Tiền sử bệnh\nBệnh lý mãn tính: tăng huyết áp.\n2. Thuốc trước khi nhập viện: aspirin 81mg.",
            AbbreviationExpander.from_path(ROOT / "src/vtr_ai/data/vi_abbreviations.json"),
        )
        self.assertTrue(any(section.kind == "diagnosis_history" for section in document.sections))
        self.assertTrue(any(section.kind == "drug_history" for section in document.sections))
        self.assertTrue(any(clause.anchor_text.lower() == "bệnh lý mãn tính" for clause in document.clauses))
        self.assertTrue(any(anchor.kind == "drug_history" for anchor in document.anchors))

    def test_patient_graph_builder_creates_section_clause_anchor_entity_links(self) -> None:
        document = build_document(
            "Bệnh lý mãn tính: tăng huyết áp. Thuốc trước khi nhập viện: aspirin 81mg.",
            AbbreviationExpander.from_path(ROOT / "src/vtr_ai/data/vi_abbreviations.json"),
        )
        clinical_entities = [
            Entity(text=text, start=start, end=end, entity_type=entity_type)
            for text, start, end, entity_type in
            [
                ("tăng huyết áp", 17, 30, "CHẨN_ĐOÁN"),
                ("aspirin 81mg", 57, 69, "THUỐC"),
            ]
        ]
        graph = build_patient_graph(document, clinical_entities)
        self.assertTrue(any(node.node_type == "section" for node in graph.nodes))
        self.assertTrue(any(node.node_type == "clause" for node in graph.nodes))
        self.assertTrue(any(node.node_type == "anchor" for node in graph.nodes))
        self.assertTrue(any(node.node_type == "entity" for node in graph.nodes))
        self.assertTrue(any(edge.relation == "mentions_entity" for edge in graph.edges))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "config.yaml")
        kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
        cls.pipeline = ClinicalNlpPipeline(config, kb)
        cls.knowledge_base = kb

    @staticmethod
    def _to_runtime_entities(rows: list[tuple[str, int, int, str]]) -> list:
        return [Entity(text=text, start=start, end=end, entity_type=entity_type) for text, start, end, entity_type in rows]

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

    def test_pipeline_uses_abbreviation_expansion_for_diagnosis(self) -> None:
        text = "Bệnh nhân có tiền sử THA và AF."
        entities = self.pipeline.process_text(text)
        texts = {entity.text for entity in entities if entity.entity_type == "CHẨN_ĐOÁN"}
        self.assertTrue(any("THA" in text or "AF" in text for text in texts))
        self.assertTrue(self.pipeline.entity_recognizer_status.startswith("hybrid"))

    def test_section_anchor_marks_historical_entities_by_default(self) -> None:
        text = "Bệnh lý mãn tính: tăng huyết áp. Thuốc trước khi nhập viện: aspirin 81mg."
        entities = self.pipeline.process_text(text)
        entity_map = {(entity.text, entity.entity_type): entity.assertions for entity in entities}
        self.assertIn("isHistorical", entity_map[("tăng huyết áp", "CHẨN_ĐOÁN")])
        self.assertIn("isHistorical", entity_map[("aspirin 81mg", "THUỐC")])

    def test_rule_backend_uses_abbreviation_expansion_for_mentions(self) -> None:
        config = load_config(ROOT / "config.yaml")
        config.ner.backend = "rule"
        kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
        recognizer = build_entity_recognizer(config, kb)
        expander = AbbreviationExpander.from_path(config.knowledge_base.abbreviations_path)
        document = build_document("BN có THA và SOB.", expander)
        entities = recognizer.extract(document)
        diagnosis_texts = {entity.text for entity in entities if entity.entity_type == "CHẨN_ĐOÁN"}
        symptom_texts = {entity.text for entity in entities if entity.entity_type == "TRIỆU_CHỨNG"}
        self.assertIn("THA", diagnosis_texts)
        self.assertIn("SOB", symptom_texts)

    def test_pipeline_extracts_common_clinical_abbreviations_and_drugs_without_dose(self) -> None:
        text = "Bệnh nhân có ĐTĐ, COPD, XHTH. Hiện đang dùng atenolol và aspirin."
        entities = self.pipeline.process_text(text)
        diagnoses = [entity for entity in entities if entity.entity_type == "CHẨN_ĐOÁN"]
        drugs = [entity for entity in entities if entity.entity_type == "THUỐC"]
        diagnosis_texts = {entity.text for entity in diagnoses}
        drug_texts = {entity.text for entity in drugs}
        self.assertIn("ĐTĐ", diagnosis_texts)
        self.assertIn("COPD", diagnosis_texts)
        self.assertIn("XHTH", diagnosis_texts)
        self.assertIn("atenolol", drug_texts)
        self.assertIn("aspirin", drug_texts)
        self.assertTrue(any("E11.9" in entity.candidates for entity in diagnoses if entity.text == "ĐTĐ"))
        self.assertTrue(any("J44.9" in entity.candidates for entity in diagnoses if entity.text == "COPD"))
        self.assertTrue(any("6918" in entity.candidates for entity in drugs if entity.text == "atenolol"))

    def test_pipeline_handles_realistic_viettel_sections_for_drug_and_lab(self) -> None:
        text = (
            "2.  Tiền sử bệnh hiện tại\n"
            "    Triệu chứng hiện tại\n"
            "    - ho\n"
            "    - mệt mỏi\n"
            "    - ho ra máu cỡ đồng xu x3 đêm qua\n"
            "    - fever (trở thành sốt đến 38.8°C)\n"
            "    - Đau bụng âm ỉ trong tháng qua\n"
            "    - Được cho dùng levofloxacin vì nghi ngờ viêm phế quản, cùng tylenol\n"
            "3.  Đánh giá tại bệnh viện\n"
            "    Kết quả xét nghiệm: công thức máu (cbc) nâng cao lên 11.3\n"
        )
        entities = self.pipeline.process_text(text)
        symptom_texts = {entity.text for entity in entities if entity.entity_type == "TRIỆU_CHỨNG"}
        drug_map = {entity.text: entity.candidates for entity in entities if entity.entity_type == "THUỐC"}
        lab_names = {entity.text for entity in entities if entity.entity_type == "TÊN_XÉT_NGHIỆM"}
        lab_values = {entity.text for entity in entities if entity.entity_type == "KẾT_QUẢ_XÉT_NGHIỆM"}
        self.assertIn("ho", symptom_texts)
        self.assertIn("ho ra máu", symptom_texts)
        self.assertIn("mệt mỏi", symptom_texts)
        self.assertIn("levofloxacin", drug_map)
        self.assertIn("tylenol", drug_map)
        self.assertIn("82122", drug_map["levofloxacin"])
        self.assertIn("161", drug_map["tylenol"])
        self.assertTrue(any("công thức máu" in text.lower() or "cbc" in text.lower() for text in lab_names))
        self.assertIn("11.3", lab_values)

    def test_assertion_ignores_non_scoping_khong_phrases(self) -> None:
        text = (
            "Lý do nhập viện: tụt huyết áp không rõ nguyên nhân và mệt mỏi. "
            "Bệnh phổi tắc nghẽn mạn tính, không xác định."
        )
        entities = self.pipeline.process_text(text)
        by_key = {(entity.text, entity.entity_type): entity.assertions for entity in entities}
        self.assertNotIn("isNegated", by_key[("mệt mỏi", "TRIỆU_CHỨNG")])
        self.assertNotIn("isNegated", by_key[("Bệnh phổi tắc nghẽn mạn tính", "CHẨN_ĐOÁN")])

    def test_pipeline_cleans_drug_prefix_and_decimal_strength(self) -> None:
        text = "Thuốc trước khi nhập viện lần này\n- lisinopril 2.5mg daily (bắt đầu)\n- Tự điều trị bằng liều cao acetaminophen 500mg"
        entities = self.pipeline.process_text(text)
        drug_texts = {entity.text for entity in entities if entity.entity_type == "THUỐC"}
        self.assertIn("lisinopril 2.5mg", drug_texts)
        self.assertIn("acetaminophen 500mg", drug_texts)

    def test_pipeline_maps_targeted_viettel_candidates(self) -> None:
        text = "Bệnh lý mãn tính: bệnh mạch máu. Sinh thiết chỉ cho thấy một u tuyến."
        entities = self.pipeline.process_text(text)
        candidate_map = {(entity.text, entity.entity_type): entity.candidates for entity in entities}
        self.assertIn("I99.9", candidate_map[("bệnh mạch máu", "CHẨN_ĐOÁN")])
        self.assertIn("D12.8", candidate_map[("u tuyến", "CHẨN_ĐOÁN")])

    def test_normalizer_cleans_route_frequency_and_prefix_noise(self) -> None:
        text = (
            "Thuốc trước khi nhập viện\n"
            "- metoprolol 25mg po bid\n"
            "- được cho aspirin 325mg x 1\n"
            "- iv lasix 40 mg once\n"
            "- prograf dose decreased from 5mg bid to 1mg bid\n"
        )
        entities = self.pipeline.process_text(text)
        drug_map = {entity.text: entity.candidates for entity in entities if entity.entity_type == "THUỐC"}
        self.assertTrue(any("6916" in cands for key, cands in drug_map.items() if "metoprolol" in key.lower()))
        self.assertIn("1191", drug_map["aspirin 325mg x 1"])
        self.assertTrue(any("4603" in cands for key, cands in drug_map.items() if "lasix" in key.lower()))
        self.assertTrue(any("42347" in cands for key, cands in drug_map.items() if "prograf" in key.lower()))

    def test_diagnosis_normalizer_cleans_common_prefix_noise(self) -> None:
        text = "Các phát hiện chẩn đoán khác\n- nghi ngờ xơ gan do rượu\n- tiền sử lâu dài của tăng huyết áp\n- viêm mô tế bào"
        entities = self.pipeline.process_text(text)
        diagnosis_map = {entity.text: entity.candidates for entity in entities if entity.entity_type == "CHẨN_ĐOÁN"}
        self.assertIn("K70.30", diagnosis_map["xơ gan do rượu"])
        self.assertTrue(any("I10" in cands for key, cands in diagnosis_map.items() if "tăng huyết áp" in key.lower()))
        self.assertTrue(any("L03.90" in cands for key, cands in diagnosis_map.items() if "viêm mô tế bào" in key.lower()))

    def test_diagnosis_normalizer_maps_additional_real_world_cases(self) -> None:
        text = (
            "Các bệnh lý mãn tính\n"
            "- u cơ trơn tử cung, không đặc hiệu\n"
            "- rối loạn lo âu, không biệt định nghiêm trọng\n"
            "- não úng thuỷ khác từ thời kỳ sơ sinh\n"
            "- bàn chân vẹo bẩm sinh\n"
            "Chẩn đoán sơ bộ : Bệnh động mạch vành mạn tính có thiếu máu cơ tim, hẹp 60% đoạn gần động mạch\n"
            "- được chẩn đoán nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin, nghi liên quan đến đường truyền\n"
        )
        entities = self.pipeline.process_text(text)
        diagnosis_map = {entity.text: entity.candidates for entity in entities if entity.entity_type == "CHẨN_ĐOÁN"}
        self.assertTrue(any("D25.9" in cands for key, cands in diagnosis_map.items() if "u cơ trơn tử cung" in key.lower()))
        self.assertTrue(any("F41.9" in cands for key, cands in diagnosis_map.items() if "rối loạn lo âu" in key.lower()))
        self.assertTrue(any("Q03.8" in cands for key, cands in diagnosis_map.items() if "não úng" in key.lower()))
        self.assertTrue(any("Q66.0" in cands for key, cands in diagnosis_map.items() if "bàn chân vẹo" in key.lower()))
        self.assertTrue(any("I25.10" in cands for key, cands in diagnosis_map.items() if "bệnh động mạch vành" in key.lower()))
        self.assertTrue(any("A41.01" in cands for key, cands in diagnosis_map.items() if "nhiễm khuẩn huyết" in key.lower()))

    def test_healthy_without_disease_phrase_is_not_extracted_as_diagnosis(self) -> None:
        text = "Bệnh lý mãn tính: nam giới khỏe mạnh, không có bệnh lý khác"
        entities = self.pipeline.process_text(text)
        self.assertFalse(any(entity.entity_type == "CHẨN_ĐOÁN" for entity in entities))

    def test_normalizer_handles_brand_and_shorthand_drug_queries(self) -> None:
        text = (
            "Xử trí thuốc\n"
            "- 1mg dilaudid\n"
            "- coumadin 3.0 mg\n"
            "- cipro\n"
            "- bactrim\n"
            "- z-pack\n"
            "- mucinex d\n"
        )
        entities = self.pipeline.process_text(text)
        drug_map = {entity.text: entity.candidates for entity in entities if entity.entity_type == "THUỐC"}
        self.assertTrue(any("3423" in cands for key, cands in drug_map.items() if "dilaudid" in key.lower()))
        self.assertTrue(any("29046" in cands for key, cands in drug_map.items() if "coumadin" in key.lower()))
        self.assertTrue(any("2551" in cands for key, cands in drug_map.items() if "cipro" in key.lower()))
        self.assertTrue(any("36437" in cands for key, cands in drug_map.items() if "bactrim" in key.lower()))
        self.assertTrue(any("855332" in cands for key, cands in drug_map.items() if "z-pack" in key.lower()))
        self.assertTrue(any("214182" in cands for key, cands in drug_map.items() if "mucinex" in key.lower()))

    def test_drug_normalizer_prioritizes_exact_ingredient_and_strength(self) -> None:
        text = "Thuốc trước khi nhập viện: salbutamol 2 MG."
        entities = self.pipeline.process_text(text)
        drug_map = {entity.text: entity.candidates for entity in entities if entity.entity_type == "THUỐC"}
        self.assertIn("313782", drug_map["salbutamol 2 MG"])

    def test_graph_retriever_returns_neighbor_drug_candidates(self) -> None:
        retriever = GraphRetriever(KnowledgeGraphIndex.from_knowledge_base(self.knowledge_base))
        entity = self._to_runtime_entities([("aspirin 81mg", 0, 12, "THUỐC")])[0]
        result = retriever.retrieve(entity, "drug_history")
        self.assertIn("1191", result.candidate_codes)

    def test_indexed_candidate_retrieval_scopes_large_icd_search(self) -> None:
        from vtr_ai.candidate_generation import rank_candidates_indexed
        from vtr_ai.config import MatchingConfig

        large_kb = load_knowledge_base(ROOT / "src/vtr_ai/data/icd10_standard.json", ROOT / "src/vtr_ai/data/rxnorm_standard.json")
        ranked = rank_candidates_indexed(
            "tăng huyết áp",
            large_kb.icd10,
            MatchingConfig(max_candidates=3, min_confidence=0.1),
            large_kb,
            "CHẨN_ĐOÁN",
        )
        self.assertTrue(any(candidate.code == "I10" for candidate in ranked))

    def test_ner_runtime_loader_returns_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_lexical_test_config(Path(tmp_dir) / "config.yaml")
            config = load_config(config_path)
            kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
            runtime = load_ner_runtime(config.ner, kb)
            predictions = runtime.predict("Bệnh nhân khó thở và rung nhĩ.")
            self.assertTrue(any(item.label == "TRIỆU_CHỨNG" for item in predictions))
            self.assertTrue(any(item.label == "CHẨN_ĐOÁN" for item in predictions))

    def test_transformer_label_map_normalization(self) -> None:
        label_map = load_transformer_label_map(
            ROOT / "src/vtr_ai/data/checkpoints/transformers_ner_metadata.json"
        )
        self.assertEqual(normalize_transformer_label("B-SYMPTOM", label_map), "TRIỆU_CHỨNG")
        self.assertEqual(normalize_transformer_label("I-DIAGNOSIS", label_map), "CHẨN_ĐOÁN")
        self.assertEqual(normalize_transformer_label("LABEL_DRUG", label_map), "THUỐC")

    def test_transformer_metadata_path_can_be_resolved_from_checkpoint_dir(self) -> None:
        checkpoint_dir = ROOT / "src/vtr_ai/data/checkpoints"
        resolved = resolve_transformer_metadata_path(checkpoint_dir, None)
        self.assertEqual(resolved, checkpoint_dir / "transformers_ner_metadata.json")

    def test_validate_transformer_checkpoint_requires_core_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = load_config(ROOT / "config.yaml")
            config.ner.provider = "transformers"
            config.ner.checkpoint_path = tmp_path
            with self.assertRaises(RuntimeError):
                validate_ner_checkpoint(config.ner)

    def test_model_backend_falls_back_when_checkpoint_missing(self) -> None:
        config = load_config(ROOT / "config.yaml")
        config.ner.checkpoint_path = ROOT / "missing-checkpoint.json"
        kb = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
        pipeline = ClinicalNlpPipeline(config, kb)
        self.assertIn("fallback", pipeline.entity_recognizer_status)

    def test_check_ner_runtime_cli_reports_runtime_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = _write_lexical_test_config(tmp_path / "config.yaml")
            output_path = tmp_path / "runtime.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            with output_path.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [
                        "python3",
                        "-m",
                        "vtr_ai.check_ner_runtime",
                        "--config",
                        str(config_path),
                        "--text",
                        "Bệnh nhân khó thở và rung nhĩ.",
                    ],
                    check=True,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["provider"], "lexical")
            self.assertEqual(payload["runtime_class"], "LexicalNerRuntime")
            self.assertTrue(any(item["type"] == "TRIỆU_CHỨNG" for item in payload["predictions"]))

    def test_negation_detection_is_local(self) -> None:
        text = "Bệnh nhân không ho nhưng đau bụng kéo dài."
        entities = self.pipeline.process_text(text)
        assertion_map = {entity.text: entity.assertions for entity in entities if entity.entity_type == "TRIỆU_CHỨNG"}
        self.assertIn("isNegated", assertion_map["ho"])
        self.assertNotIn("isNegated", assertion_map["đau bụng"])

    def test_double_negation_does_not_mark_entity_negated(self) -> None:
        text = "Không loại trừ viêm phổi."
        entities = self.pipeline.process_text(text)
        diagnosis_entities = [entity for entity in entities if entity.entity_type == "CHẨN_ĐOÁN"]
        self.assertTrue(any("viêm phổi" in entity.text.lower() for entity in diagnosis_entities))
        self.assertTrue(all("isNegated" not in entity.assertions for entity in diagnosis_entities))

    def test_family_and_negation_do_not_leak_across_clauses(self) -> None:
        text = "Bố bệnh nhân bị tăng huyết áp; bệnh nhân không sốt."
        entities = self.pipeline.process_text(text)
        entity_map = {entity.text: entity.assertions for entity in entities if entity.entity_type in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}}
        self.assertIn("isFamily", entity_map["tăng huyết áp"])
        self.assertNotIn("isFamily", entity_map["sốt"])
        self.assertIn("isNegated", entity_map["sốt"])

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

    def test_submission_builds_zip_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            zip_path = tmp_path / "output.zip"
            input_dir.mkdir()
            (input_dir / "1.txt").write_text(
                "Thuốc trước khi nhập viện: Chlorpheniramine 0.4 MG/ML. WBC:14,43;",
                encoding="utf-8",
            )
            exit_code = submission_main(
                [
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--zip_path",
                    str(zip_path),
                    "--config",
                    str(ROOT / "config.yaml"),
                ]
            )
            self.assertEqual(exit_code, 0)
            validated = validate_output_directory(input_dir, output_dir)
            self.assertEqual(len(validated), 1)
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["output/1.json"])

    def test_evaluation_scores_entities_assertions_and_candidates(self) -> None:
        gold = [
            {
                "text": "ho",
                "position": [12, 14],
                "type": "TRIỆU_CHỨNG",
                "assertions": ["isNegated"],
                "candidates": [],
            },
            {
                "text": "THA",
                "position": [26, 29],
                "type": "CHẨN_ĐOÁN",
                "assertions": ["isHistorical"],
                "candidates": ["I10"],
            },
        ]
        predicted = [
            {
                "text": "ho",
                "position": [12, 14],
                "type": "TRIỆU_CHỨNG",
                "assertions": ["isNegated"],
                "candidates": [],
            },
            {
                "text": "THA",
                "position": [26, 29],
                "type": "CHẨN_ĐOÁN",
                "assertions": [],
                "candidates": ["I10", "I11.9"],
            },
        ]
        summary = evaluate_entity_lists(predicted, gold)
        self.assertEqual(summary["entity_span"].true_positive, 2)
        self.assertEqual(summary["entity_span_type"].true_positive, 2)
        self.assertEqual(summary["entity_record"].true_positive, 1)
        self.assertEqual(summary["assertions"].true_positive, 1)
        self.assertEqual(summary["assertions"].false_negative, 1)
        self.assertEqual(summary["candidates"].true_positive, 1)
        self.assertEqual(summary["candidates"].false_positive, 1)

    def test_evaluation_cli_supports_directory_level_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            gold_dir = tmp_path / "gold"
            pred_dir = tmp_path / "pred"
            input_dir.mkdir()
            gold_dir.mkdir()
            pred_dir.mkdir()

            raw_text = "Bệnh nhân không ho. Tiền sử THA."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (gold_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "ho",
                            "position": [16, 18],
                            "type": "TRIỆU_CHỨNG",
                            "assertions": ["isNegated"],
                            "candidates": [],
                        },
                        {
                            "text": "THA",
                            "position": [28, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                            "candidates": ["I10"],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (pred_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "ho",
                            "position": [16, 18],
                            "type": "TRIỆU_CHỨNG",
                            "assertions": ["isNegated"],
                            "candidates": [],
                        },
                        {
                            "text": "THA",
                            "position": [28, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                            "candidates": [],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            summary = evaluate_output_directories(input_dir, pred_dir, gold_dir)
            self.assertEqual(summary.files, 1)
            self.assertEqual(summary.entity_record.true_positive, 1)
            self.assertEqual(summary.entity_record.false_negative, 1)
            self.assertEqual(summary.assertions.true_positive, 2)
            self.assertEqual(summary.candidates.false_negative, 1)

    def test_analysis_report_collects_unmatched_candidates_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            raw_text = "Bệnh lý mãn tính: bệnh mạch máu. Thuốc trước khi nhập viện: aspirin 81mg."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "bệnh mạch máu",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": [],
                        },
                        {
                            "text": "aspirin 81mg",
                            "position": [60, 72],
                            "type": "THUỐC",
                            "assertions": ["isHistorical"],
                            "candidates": ["1191"],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            records = collect_unmatched_records(input_dir, output_dir, entity_types={"CHẨN_ĐOÁN"})
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].text, "bệnh mạch máu")
            self.assertIn("bệnh mạch máu", records[0].context)
            summary = summarize_unmatched(records)
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["by_type"]["CHẨN_ĐOÁN"], 1)

    def test_analysis_report_cli_can_write_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            report_path = tmp_path / "report.json"
            input_dir.mkdir()
            output_dir.mkdir()
            raw_text = "Bệnh lý mãn tính: bệnh mạch máu."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "bệnh mạch máu",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": [],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "vtr_ai.analysis_report",
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--output_path",
                    str(report_path),
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total"], 1)
            self.assertIn("bệnh mạch máu", completed.stdout)

    def test_audit_report_cli_writes_summary_and_unmatched_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            report_dir = tmp_path / "reports"
            input_dir.mkdir()
            output_dir.mkdir()
            raw_text = "Bệnh lý mãn tính: bệnh mạch máu. Thuốc trước khi nhập viện: aspirin 81mg."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "bệnh mạch máu",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": [],
                        },
                        {
                            "text": "aspirin 81mg",
                            "position": [60, 72],
                            "type": "THUỐC",
                            "assertions": ["isHistorical"],
                            "candidates": ["1191"],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "vtr_ai.audit_report",
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--report_dir",
                    str(report_dir),
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((report_dir / "summary.json").exists())
            self.assertTrue((report_dir / "unmatched_diagnosis.json").exists())
            self.assertTrue((report_dir / "unmatched_drugs.json").exists())
            summary = build_entity_summary(output_dir)
            self.assertEqual(summary["entity_counts"]["CHẨN_ĐOÁN"], 1)
            self.assertIn("unmatched_diagnosis_total", completed.stdout)

    def test_workflow_runs_submission_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            report_dir = tmp_path / "reports"
            zip_path = tmp_path / "output.zip"
            input_dir.mkdir()
            (input_dir / "1.txt").write_text(
                "Thuốc trước khi nhập viện: aspirin 81mg. Bệnh lý mãn tính: tăng huyết áp.",
                encoding="utf-8",
            )
            exit_code = workflow_main(
                [
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--zip_path",
                    str(zip_path),
                    "--report_dir",
                    str(report_dir),
                    "--config",
                    str(ROOT / "config.yaml"),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "1.json").exists())
            self.assertTrue(zip_path.exists())
            self.assertTrue((report_dir / "summary.json").exists())
            self.assertTrue((report_dir / "unmatched_diagnosis.json").exists())

    def test_bootstrap_annotations_exports_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            export_path = tmp_path / "bootstrap.jsonl"
            input_dir.mkdir()
            output_dir.mkdir()
            raw_text = "Bệnh lý mãn tính: tăng huyết áp."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": ["I10"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            written = export_bootstrap_annotations(input_dir, output_dir, export_path)
            self.assertEqual(written, export_path)
            lines = export_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["file_name"], "1.txt")
            self.assertEqual(payload["text"], raw_text)
            self.assertEqual(payload["predicted_entities"][0]["candidates"], ["I10"])

    def test_bootstrap_annotations_can_seed_gold_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            export_path = tmp_path / "bootstrap_seeded.jsonl"
            input_dir.mkdir()
            output_dir.mkdir()
            raw_text = "Bệnh lý mãn tính: tăng huyết áp."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": ["I10"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            export_bootstrap_annotations(input_dir, output_dir, export_path, seed_gold_entities=True)
            payload = json.loads(export_path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["predicted_entities"][0]["text"], "tăng huyết áp")
            self.assertEqual(payload["gold_entities"][0]["candidates"], ["I10"])

    def test_review_jsonl_can_be_converted_to_gold_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            review_path = tmp_path / "review.jsonl"
            gold_dir = tmp_path / "gold"
            review_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "file_name": "1.txt",
                                "text": "Bệnh lý mãn tính: tăng huyết áp.",
                                "gold_entities": [
                                    {
                                        "text": "tăng huyết áp",
                                        "position": [18, 31],
                                        "type": "CHẨN_ĐOÁN",
                                        "assertions": [],
                                        "candidates": ["I10"],
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "file_name": "2.txt",
                                "text": "Thuốc trước khi nhập viện: aspirin 81mg.",
                                "predicted_entities": [
                                    {
                                        "text": "aspirin 81mg",
                                        "position": [28, 40],
                                        "type": "THUỐC",
                                        "assertions": ["isHistorical"],
                                        "candidates": ["1191"],
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            written = export_review_jsonl_to_gold_dir(review_path, gold_dir)
            self.assertEqual(written, gold_dir)
            gold_1 = json.loads((gold_dir / "1.json").read_text(encoding="utf-8"))
            gold_2 = json.loads((gold_dir / "2.json").read_text(encoding="utf-8"))
            self.assertEqual(gold_1[0]["candidates"], ["I10"])
            self.assertEqual(gold_2[0]["type"], "THUỐC")

    def test_gold_dir_can_be_converted_to_ner_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            gold_dir = tmp_path / "gold"
            output_path = tmp_path / "ner_train.jsonl"
            input_dir.mkdir()
            gold_dir.mkdir()
            (input_dir / "1.txt").write_text("Bệnh lý mãn tính: tăng huyết áp.", encoding="utf-8")
            (gold_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": ["I10"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            written = export_gold_dir_to_ner_jsonl(input_dir, gold_dir, output_path)
            self.assertEqual(written, output_path)
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["text"], "Bệnh lý mãn tính: tăng huyết áp.")
            self.assertEqual(payload["entities"][0]["type"], "CHẨN_ĐOÁN")

    def test_review_subset_ranks_files_with_unmatched_candidates_higher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "1.txt").write_text("Bệnh lý mãn tính: bệnh mạch máu.", encoding="utf-8")
            (input_dir / "2.txt").write_text("Thuốc trước khi nhập viện: aspirin 81mg.", encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "bệnh mạch máu",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": [],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (output_dir / "2.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "aspirin 81mg",
                            "position": [27, 39],
                            "type": "THUỐC",
                            "assertions": ["isHistorical"],
                            "candidates": ["1191"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ranked = rank_files_for_review(input_dir, output_dir)
            self.assertEqual(ranked[0]["file_name"], "1.txt")
            self.assertEqual(ranked[0]["unmatched_candidates"], 1)

    def test_select_review_subset_exports_only_ranked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bootstrap_path = tmp_path / "bootstrap.jsonl"
            priority_path = tmp_path / "priority.json"
            output_path = tmp_path / "subset.jsonl"
            bootstrap_path.write_text(
                "\n".join(
                    [
                        json.dumps({"file_name": "1.txt", "text": "A", "predicted_entities": []}, ensure_ascii=False),
                        json.dumps({"file_name": "2.txt", "text": "B", "predicted_entities": []}, ensure_ascii=False),
                        json.dumps({"file_name": "3.txt", "text": "C", "predicted_entities": []}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            priority_path.write_text(
                json.dumps(
                    {
                        "top_files": [
                            {"file_name": "2.txt"},
                            {"file_name": "3.txt"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            written = export_selected_review_subset(bootstrap_path, priority_path, output_path, limit=1)
            self.assertEqual(written, output_path)
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["file_name"], "2.txt")

    def test_prepare_dev_subset_builds_seeded_subset_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            report_dir = tmp_path / "reports"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "1.txt").write_text("Bệnh lý mãn tính: bệnh mạch máu.", encoding="utf-8")
            (input_dir / "2.txt").write_text("Bệnh lý mãn tính: tăng huyết áp.", encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "bệnh mạch máu",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": [],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (output_dir / "2.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": ["I10"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            exit_code = prepare_dev_subset_main(
                [
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--report_dir",
                    str(report_dir),
                    "--limit",
                    "1",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((report_dir / "bootstrap_seeded.jsonl").exists())
            self.assertTrue((report_dir / "review_priority.json").exists())
            self.assertTrue((report_dir / "dev_subset.jsonl").exists())
            subset_lines = (report_dir / "dev_subset.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(subset_lines), 1)
            payload = json.loads(subset_lines[0])
            self.assertEqual(payload["file_name"], "1.txt")
            self.assertIn("gold_entities", payload)

    def test_review_packet_exports_txt_json_and_jsonl_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            packet_dir = tmp_path / "packet"
            review_jsonl = tmp_path / "dev_subset.jsonl"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "1.txt").write_text("Bệnh lý mãn tính: tăng huyết áp.", encoding="utf-8")
            (output_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": [],
                            "candidates": ["I10"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            review_jsonl.write_text(
                json.dumps(
                    {
                        "file_name": "1.txt",
                        "text": "Bệnh lý mãn tính: tăng huyết áp.",
                        "predicted_entities": [],
                        "gold_entities": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            written = export_review_packet(input_dir, output_dir, review_jsonl, packet_dir)
            self.assertEqual(written, packet_dir)
            self.assertTrue((packet_dir / "1.txt").exists())
            self.assertTrue((packet_dir / "1.json").exists())
            self.assertTrue((packet_dir / "review_subset.jsonl").exists())
            manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["count"], 1)

    def test_review_to_train_builds_gold_dir_and_ner_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            (input_dir / "1.txt").write_text("Bệnh lý mãn tính: tăng huyết áp.", encoding="utf-8")
            review_jsonl = tmp_path / "review.jsonl"
            review_jsonl.write_text(
                json.dumps(
                    {
                        "file_name": "1.txt",
                        "text": "Bệnh lý mãn tính: tăng huyết áp.",
                        "gold_entities": [
                            {
                                "text": "tăng huyết áp",
                                "position": [18, 31],
                                "type": "CHẨN_ĐOÁN",
                                "assertions": [],
                                "candidates": ["I10"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            gold_dir = tmp_path / "gold"
            ner_path = tmp_path / "ner_train.jsonl"
            exit_code = review_to_train_main(
                [
                    "--review_jsonl",
                    str(review_jsonl),
                    "--input_dir",
                    str(input_dir),
                    "--gold_dir",
                    str(gold_dir),
                    "--ner_output_path",
                    str(ner_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((gold_dir / "1.json").exists())
            self.assertTrue(ner_path.exists())
            payload = json.loads(ner_path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["entities"][0]["type"], "CHẨN_ĐOÁN")

    def test_validate_review_jsonl_accepts_seeded_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            review_jsonl = tmp_path / "review.jsonl"
            review_jsonl.write_text(
                json.dumps(
                    {
                        "file_name": "1.txt",
                        "text": "Bệnh lý mãn tính: tăng huyết áp.",
                        "predicted_entities": [
                            {
                                "text": "tăng huyết áp",
                                "position": [18, 31],
                                "type": "CHẨN_ĐOÁN",
                                "assertions": [],
                                "candidates": ["I10"],
                            }
                        ],
                        "gold_entities": [
                            {
                                "text": "tăng huyết áp",
                                "position": [18, 31],
                                "type": "CHẨN_ĐOÁN",
                                "assertions": [],
                                "candidates": ["I10"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_review_jsonl(review_jsonl), 1)

    def test_validate_review_jsonl_rejects_bad_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            review_jsonl = tmp_path / "review.jsonl"
            review_jsonl.write_text(
                json.dumps(
                    {
                        "file_name": "1.txt",
                        "text": "Bệnh lý mãn tính: tăng huyết áp.",
                        "gold_entities": [
                            {
                                "text": "tăng huyết áp",
                                "position": [17, 31],
                                "type": "CHẨN_ĐOÁN",
                                "assertions": [],
                                "candidates": ["I10"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ReviewValidationError):
                validate_review_jsonl(review_jsonl)

    def test_collect_semantic_examples_builds_shortlist_with_gold_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            gold_dir = tmp_path / "gold"
            input_dir.mkdir()
            gold_dir.mkdir()
            config_path = _write_lexical_test_config(tmp_path / "config.yaml")

            raw_text = "Bệnh lý mãn tính: tăng huyết áp. Thuốc trước khi nhập viện: aspirin 81mg."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (gold_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                            "candidates": ["I10"],
                        },
                        {
                            "text": "aspirin 81mg",
                            "position": [60, 72],
                            "type": "THUỐC",
                            "assertions": ["isHistorical"],
                            "candidates": ["1191"],
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            examples, unmatched = collect_semantic_examples(input_dir, gold_dir, config_path, shortlist_size=5)
            self.assertEqual(unmatched, [])
            self.assertEqual(len(examples), 2)
            by_type = {example.entity_type: example for example in examples}
            self.assertIn("I10", [item["code"] for item in by_type["CHẨN_ĐOÁN"].shortlist])
            self.assertIn("1191", [item["code"] for item in by_type["THUỐC"].shortlist])

    def test_export_semantic_dataset_bundle_writes_prompt_and_sft_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            gold_dir = tmp_path / "gold"
            output_dir = tmp_path / "semantic_bundle"
            input_dir.mkdir()
            gold_dir.mkdir()
            config_path = _write_lexical_test_config(tmp_path / "config.yaml")

            samples = {
                "1": (
                    "Bệnh lý mãn tính: tăng huyết áp.",
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                            "candidates": ["I10"],
                        }
                    ],
                ),
                "2": (
                    "Thuốc trước khi nhập viện: aspirin 81mg.",
                    [
                        {
                            "text": "aspirin 81mg",
                            "position": [27, 39],
                            "type": "THUỐC",
                            "assertions": ["isHistorical"],
                            "candidates": ["1191"],
                        }
                    ],
                ),
                "3": (
                    "Bệnh lý mãn tính: COPD.",
                    [
                        {
                            "text": "COPD",
                            "position": [18, 22],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                            "candidates": ["J44.9"],
                        }
                    ],
                ),
            }
            for stem, (raw_text, entities) in samples.items():
                (input_dir / f"{stem}.txt").write_text(raw_text, encoding="utf-8")
                (gold_dir / f"{stem}.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")

            written = export_semantic_dataset_bundle(
                input_dir=input_dir,
                gold_dir=gold_dir,
                config_path=config_path,
                output_dir=output_dir,
                shortlist_size=5,
                support_size_per_type=1,
                shots_per_query=1,
                seed=13,
            )
            self.assertEqual(written, output_dir)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "support.jsonl").exists())
            self.assertTrue((output_dir / "zero_shot_eval.jsonl").exists())
            self.assertTrue((output_dir / "few_shot_eval.jsonl").exists())
            self.assertTrue((output_dir / "sft_train.jsonl").exists())
            self.assertTrue((output_dir / "sft_dev.jsonl").exists())

            zero_shot_lines = (output_dir / "zero_shot_eval.jsonl").read_text(encoding="utf-8").strip().splitlines()
            if zero_shot_lines:
                payload = json.loads(zero_shot_lines[0])
                self.assertEqual(payload["messages"][0]["role"], "system")
                self.assertIn("candidates", payload["expected"])
            sft_lines = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            if sft_lines:
                payload = json.loads(sft_lines[0])
                self.assertEqual(payload["messages"][-1]["role"], "assistant")
                self.assertIn("candidates", payload["messages"][-1]["content"])

    def test_collect_fixed_span_examples_keeps_gold_assertions_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            gold_dir = tmp_path / "gold"
            input_dir.mkdir()
            gold_dir.mkdir()
            config_path = _write_lexical_test_config(tmp_path / "config.yaml")

            raw_text = "Bệnh lý mãn tính: tăng huyết áp. Triệu chứng hiện tại: khó thở. Thuốc trước khi nhập viện: aspirin 81mg."
            (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
            (gold_dir / "1.json").write_text(
                json.dumps(
                    [
                        {
                            "text": "tăng huyết áp",
                            "position": [18, 31],
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                            "candidates": ["I10"],
                        },
                        {
                            "text": "khó thở",
                            "position": [55, 62],
                            "type": "TRIỆU_CHỨNG",
                            "assertions": ["isNegated", "isHistorical"],
                            "candidates": [],
                        },
                        {
                            "text": "aspirin 81mg",
                            "position": [91, 103],
                            "type": "THUỐC",
                            "assertions": ["isHistorical"],
                            "candidates": ["1191"],
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            examples = collect_fixed_span_examples(input_dir, gold_dir, config_path, shortlist_size=5)
            self.assertEqual(len(examples), 3)
            by_type = {example.entity_type: example for example in examples}
            self.assertEqual(by_type["CHẨN_ĐOÁN"].gold_candidates, ["I10"])
            self.assertEqual(by_type["TRIỆU_CHỨNG"].gold_assertions, ["isHistorical", "isNegated"])
            self.assertEqual(by_type["TRIỆU_CHỨNG"].gold_candidates, [])
            self.assertIn("1191", [item["code"] for item in by_type["THUỐC"].shortlist])

    def test_export_qwen_fixed_span_dataset_writes_sft_and_eval_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            gold_dir = tmp_path / "gold"
            output_dir = tmp_path / "qwen_fixed_span"
            input_dir.mkdir()
            gold_dir.mkdir()
            config_path = _write_lexical_test_config(tmp_path / "config.yaml")

            samples = {
                "1": (
                    "Bệnh lý mãn tính: tăng huyết áp.",
                    [{"text": "tăng huyết áp", "position": [18, 31], "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"], "candidates": ["I10"]}],
                ),
                "2": (
                    "Thuốc trước khi nhập viện: aspirin 81mg.",
                    [{"text": "aspirin 81mg", "position": [27, 39], "type": "THUỐC", "assertions": ["isHistorical"], "candidates": ["1191"]}],
                ),
                "3": (
                    "Triệu chứng hiện tại: khó thở.",
                    [{"text": "khó thở", "position": [22, 29], "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"], "candidates": []}],
                ),
            }
            for stem, (raw_text, entities) in samples.items():
                (input_dir / f"{stem}.txt").write_text(raw_text, encoding="utf-8")
                (gold_dir / f"{stem}.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")

            written = export_qwen_fixed_span_dataset(
                input_dir=input_dir,
                gold_dir=gold_dir,
                config_path=config_path,
                output_dir=output_dir,
                shortlist_size=5,
                seed=7,
            )
            self.assertEqual(written, output_dir)
            self.assertTrue((output_dir / "sft_train.jsonl").exists())
            self.assertTrue((output_dir / "eval_test.jsonl").exists())
            train_lines = (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            if train_lines:
                payload = json.loads(train_lines[0])
                self.assertEqual(payload["messages"][-1]["role"], "assistant")
                self.assertIn("assertions", payload["messages"][-1]["content"])

    def test_parse_qwen_json_response_and_apply_prediction_to_entity(self) -> None:
        parsed = parse_qwen_json_response(
            "```json\n{\"assertions\":[\"isNegated\",\"bad\"],\"candidates\":[\"I10\",\"I10\"]}\n```"
        )
        self.assertEqual(parsed["assertions"], ["isNegated"])
        self.assertEqual(parsed["candidates"], ["I10"])

        parsed_with_labels = parse_qwen_json_response(
            "```json\n{\"assertions\":[],\"candidates\":[\"I10: Essential (primary) hypertension\",\"1191: aspirin 325 MG\"]}\n```"
        )
        self.assertEqual(parsed_with_labels["candidates"], ["I10", "1191"])

        symptom = {
            "text": "khó thở",
            "position": [1, 8],
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": ["should-clear"],
        }
        updated = apply_prediction_to_entity(symptom, {"assertions": ["isFamily"], "candidates": ["X"]})
        self.assertEqual(updated["assertions"], ["isFamily"])
        self.assertEqual(updated["candidates"], [])

    def test_mention_proposal_handles_empty_or_newline_text_gracefully(self) -> None:
        from vtr_ai.mention_proposal import _iter_lines_with_offsets, propose_mentions
        from vtr_ai.config import RulesConfig
        self.assertEqual(_iter_lines_with_offsets(""), [])
        self.assertEqual(_iter_lines_with_offsets("\n"), [(0, "")])
        
        # Test propose_mentions works without error
        config = RulesConfig()
        proposals = propose_mentions("", config)
        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()
