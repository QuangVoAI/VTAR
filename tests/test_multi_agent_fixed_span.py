from __future__ import annotations

import unittest
from pathlib import Path

from vtr_ai.multi_agent_fixed_span import (
    CandidateJudgeAgent,
    FixedSpanAgent,
    FixedSpanMultiAgentOrchestrator,
    RetrievalResult,
    SemanticSelectionResult,
)


ROOT = Path(__file__).resolve().parents[1]


class MultiAgentFixedSpanTests(unittest.TestCase):
    def test_fixed_span_agent_rejects_offset_changes(self) -> None:
        agent = FixedSpanAgent()
        text = "Tăng huyết áp"
        entity = {
            "text": text,
            "position": [0, len(text)],
            "type": "CHẨN_ĐOÁN",
            "assertions": [],
            "candidates": [],
        }
        self.assertEqual(agent.run(text, [entity]), [entity])
        with self.assertRaises(ValueError):
            agent.run(text, [{**entity, "position": [1, len(text)]}])

    def test_mapped_candidates_do_not_bypass_semantic_judging(self) -> None:
        judge = CandidateJudgeAgent()
        retrieval = RetrievalResult(
            entity={
                "text": "bệnh động mạch vành",
                "type": "CHẨN_ĐOÁN",
                "candidates": ["I25.10", "I25.110"],
            },
            example=None,
            shortlist_codes=["I25.10", "I25.110", "I10"],
            shortlist=[
                {"code": "I25.10", "score": 0.8, "source": "mapped"},
                {"code": "I25.110", "score": 0.7, "source": "mapped"},
                {"code": "I10", "score": 1.1, "source": "exact"},
            ],
        )
        semantic = SemanticSelectionResult(
            entity=retrieval.entity,
            candidates=["I10"],
            raw_prediction={"assertions": [], "candidates": ["I10"]},
        )
        judged = judge.run(retrieval, assertion=None, semantic=semantic)
        self.assertEqual(judged.candidates, ["I10"])

    def test_orchestrator_splits_assertion_and_semantic_roles(self) -> None:
        text = (
            "Tiền sử tăng huyết áp.\n"
            "Bệnh nhân phủ nhận khó thở.\n"
            "Thuốc trước khi nhập viện cipro.\n"
        )
        diagnosis = "tăng huyết áp"
        symptom = "khó thở"
        drug = "cipro"
        entities = [
            {
                "text": diagnosis,
                "position": [text.index(diagnosis), text.index(diagnosis) + len(diagnosis)],
                "type": "CHẨN_ĐOÁN",
                "assertions": [],
                "candidates": [],
            },
            {
                "text": symptom,
                "position": [text.index(symptom), text.index(symptom) + len(symptom)],
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
            },
            {
                "text": drug,
                "position": [text.index(drug), text.index(drug) + len(drug)],
                "type": "THUỐC",
                "assertions": [],
                "candidates": [],
            },
        ]

        def fake_generator(messages: list[dict]) -> dict[str, list[str]]:
            prompt = messages[-1]["content"]
            if "Mention: cipro" in prompt:
                return {"assertions": [], "candidates": ["2551", "OUTSIDE"]}
            if "Mention: tăng huyết áp" in prompt:
                return {"assertions": ["isNegated"], "candidates": ["I10", "BADCODE"]}
            return {"assertions": [], "candidates": []}

        orchestrator = FixedSpanMultiAgentOrchestrator(
            config_path=ROOT / "config.yaml",
            context_window=220,
            shortlist_size=10,
            shortlist_min_confidence=0.1,
            generator=fake_generator,
        )
        merged = orchestrator.process_entities(text, "sample.txt", entities)
        by_key = {(item["text"], item["type"]): item for item in merged}

        self.assertEqual(by_key[("tăng huyết áp", "CHẨN_ĐOÁN")]["assertions"], ["isHistorical"])
        self.assertEqual(by_key[("tăng huyết áp", "CHẨN_ĐOÁN")]["candidates"], ["I10"])

        self.assertEqual(by_key[("khó thở", "TRIỆU_CHỨNG")]["assertions"], ["isNegated"])
        self.assertEqual(by_key[("khó thở", "TRIỆU_CHỨNG")]["candidates"], [])

        self.assertEqual(by_key[("cipro", "THUỐC")]["assertions"], ["isHistorical"])
        self.assertEqual(by_key[("cipro", "THUỐC")]["candidates"], ["2551"])

    def test_candidate_judge_drops_extra_drug_codes(self) -> None:
        judge = CandidateJudgeAgent()
        retrieval = RetrievalResult(
            entity={"text": "aspirin 81mg", "type": "THUỐC"},
            example=None,
            shortlist_codes=["1191", "30131", "6851"],
            shortlist=[
                {"code": "1191", "score": 1.2, "source": "exact"},
                {"code": "30131", "score": 0.8, "source": "embedding"},
                {"code": "6851", "score": 0.78, "source": "embedding"},
            ],
        )
        semantic = SemanticSelectionResult(
            entity=retrieval.entity,
            candidates=["1191", "30131", "6851"],
            raw_prediction={"assertions": [], "candidates": ["1191", "30131", "6851"]},
        )
        judged = judge.run(retrieval, assertion=None, semantic=semantic)
        self.assertEqual(judged.candidates, ["1191"])

    def test_candidate_judge_falls_back_to_strong_exact_shortlist(self) -> None:
        judge = CandidateJudgeAgent()
        retrieval = RetrievalResult(
            entity={"text": "cipro", "type": "THUỐC"},
            example=None,
            shortlist_codes=["2551", "6916"],
            shortlist=[
                {"code": "2551", "score": 1.15, "source": "exact"},
                {"code": "6916", "score": 0.21, "source": "embedding"},
            ],
        )
        semantic = SemanticSelectionResult(
            entity=retrieval.entity,
            candidates=[],
            raw_prediction={"assertions": [], "candidates": []},
        )
        judged = judge.run(retrieval, assertion=None, semantic=semantic)
        self.assertEqual(judged.candidates, ["2551"])

    def test_candidate_judge_allows_multicode_only_with_strong_signal(self) -> None:
        judge = CandidateJudgeAgent()
        retrieval = RetrievalResult(
            entity={"text": "ER+/HER2−", "type": "CHẨN_ĐOÁN"},
            example=None,
            shortlist_codes=["Z17.0", "Z17.1", "C50.912"],
            shortlist=[
                {"code": "Z17.0", "score": 1.02, "source": "exact"},
                {"code": "Z17.1", "score": 0.96, "source": "exact"},
                {"code": "C50.912", "score": 0.70, "source": "embedding"},
            ],
        )
        semantic = SemanticSelectionResult(
            entity=retrieval.entity,
            candidates=["Z17.0", "Z17.1", "C50.912"],
            raw_prediction={"assertions": [], "candidates": ["Z17.0", "Z17.1", "C50.912"]},
        )
        judged = judge.run(retrieval, assertion=None, semantic=semantic)
        self.assertEqual(judged.candidates, ["Z17.0", "Z17.1"])


if __name__ == "__main__":
    unittest.main()
