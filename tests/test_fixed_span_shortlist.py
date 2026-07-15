from __future__ import annotations

import unittest
from pathlib import Path

from vtr_ai.fixed_span_shortlist import _bare_drug_preference_score, rank_fixed_span_shortlist
from vtr_ai.knowledge_base import KnowledgeRecord, load_knowledge_base


ROOT = Path(__file__).resolve().parents[1]


class FixedSpanShortlistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = load_knowledge_base(
            ROOT / "src/vtr_ai/data/icd10_standard.json",
            ROOT / "src/vtr_ai/data/rxnorm_standard.json",
        )

    def test_diagnosis_exact_alias_stays_in_shortlist(self) -> None:
        ranked = rank_fixed_span_shortlist("tăng huyết áp", "CHẨN_ĐOÁN", self.kb, shortlist_size=10, min_confidence=0.1)
        self.assertIn("I10", ranked)

    def test_drug_strength_match_beats_wrong_strength(self) -> None:
        ranked = rank_fixed_span_shortlist(
            "Chlorpheniramine 0.4 MG/ML",
            "THUỐC",
            self.kb,
            shortlist_size=10,
            min_confidence=0.1,
        )
        self.assertTrue(ranked)

    def test_drug_without_strength_keeps_base_ingredient_candidates(self) -> None:
        ranked = rank_fixed_span_shortlist("atenolol", "THUỐC", self.kb, shortlist_size=10, min_confidence=0.1)
        self.assertIn("6918", ranked)

    def test_bare_drug_prefers_approved_alias_over_product_label(self) -> None:
        ingredient = KnowledgeRecord(code="x", label="Ingredient", aliases=["brand"])
        product = KnowledgeRecord(code="y", label="brand", aliases=[])
        self.assertGreater(_bare_drug_preference_score("brand", ingredient), 0.0)
        self.assertLess(_bare_drug_preference_score("brand", product), 0.0)


if __name__ == "__main__":
    unittest.main()
