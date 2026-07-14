from __future__ import annotations

import unittest
from pathlib import Path

from vtr_ai.knowledge_base import load_knowledge_base
from vtr_ai.semantic_embedding_retriever import rank_by_embedding


ROOT = Path(__file__).resolve().parents[1]


class SemanticEmbeddingRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = load_knowledge_base(
            ROOT / "src/vtr_ai/data/icd10_standard.json",
            ROOT / "src/vtr_ai/data/rxnorm_standard.json",
        )

    def test_embedding_prefers_hypertension_alias(self) -> None:
        subset = [record for record in self.kb.icd10 if record.code in {"I10", "I60.9", "K92.2", "A41.01"}]
        ranked = rank_by_embedding("tăng huyết áp", subset, top_k=4)
        self.assertEqual(ranked[0].code, "I10")

    def test_embedding_prefers_exact_drug_surface(self) -> None:
        subset = [record for record in self.kb.rxnorm if record.code in {"6918", "1191", "4603"}]
        ranked = rank_by_embedding("atenolol 50mg", subset, top_k=3)
        self.assertEqual(ranked[0].code, "6918")


if __name__ == "__main__":
    unittest.main()
