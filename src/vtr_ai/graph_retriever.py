from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from .knowledge_graph import KnowledgeGraphIndex
from .schemas import Entity


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-zà-ỹđ0-9]+", " ", text.lower(), flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


@dataclass(slots=True)
class GraphRetrievalResult:
    candidate_codes: list[str]
    support: dict[str, float]


class GraphRetriever:
    def __init__(self, index: KnowledgeGraphIndex) -> None:
        self.index = index

    def retrieve(self, entity: Entity, clause_kind: str | None = None) -> GraphRetrievalResult:
        if entity.entity_type == "CHẨN_ĐOÁN":
            return self._retrieve_diagnosis(entity, clause_kind)
        if entity.entity_type == "THUỐC":
            return self._retrieve_drug(entity, clause_kind)
        return GraphRetrievalResult(candidate_codes=[], support={})

    def _retrieve_diagnosis(self, entity: Entity, clause_kind: str | None) -> GraphRetrievalResult:
        query = _normalize(entity.text)
        direct = self.index.alias_to_diagnosis_codes.get(query, set())
        support: dict[str, float] = {}
        for code in direct:
            support[code] = max(support.get(code, 0.0), 1.0)
            for neighbor in self.index.diagnosis_nodes[code].neighbors:
                support[neighbor] = max(support.get(neighbor, 0.0), 0.35)
        if not support:
            query_tokens = set(query.split())
            pool_codes = {
                code
                for token in query_tokens
                for code in self.index.diagnosis_token_to_codes.get(token, set())
            }
            for code in pool_codes:
                node = self.index.diagnosis_nodes[code]
                similarity = max(
                    SequenceMatcher(None, query, _normalize(alias)).ratio()
                    for alias in [node.label, *node.aliases]
                )
                if similarity >= 0.72:
                    support[code] = max(support.get(code, 0.0), similarity)
        if clause_kind in {"diagnosis_history", "diagnosis_current"}:
            support = {code: score + 0.05 for code, score in support.items()}
        ordered = sorted(support.items(), key=lambda item: (-item[1], item[0]))
        return GraphRetrievalResult(candidate_codes=[code for code, _ in ordered[:6]], support=support)

    def _retrieve_drug(self, entity: Entity, clause_kind: str | None) -> GraphRetrievalResult:
        query = _normalize(entity.text)
        direct = self.index.alias_to_drug_codes.get(query, set())
        support: dict[str, float] = {}
        for code in direct:
            support[code] = max(support.get(code, 0.0), 1.0)
            for neighbor in self.index.drug_nodes[code].neighbors:
                support[neighbor] = max(support.get(neighbor, 0.0), 0.45)
        if not support:
            query_ingredient = _normalize(_strip_strength(entity.text))
            query_tokens = set(query_ingredient.split())
            pool_codes = {
                code
                for token in query_tokens
                for code in self.index.drug_token_to_codes.get(token, set())
            }
            for code in pool_codes:
                node = self.index.drug_nodes[code]
                aliases = [node.label, *node.aliases]
                alias_scores = [SequenceMatcher(None, query, _normalize(alias)).ratio() for alias in aliases]
                ingredient_scores = [SequenceMatcher(None, query_ingredient, _normalize(_strip_strength(alias))).ratio() for alias in aliases]
                best = max(max(alias_scores, default=0.0), max(ingredient_scores, default=0.0))
                if best >= 0.72:
                    support[code] = max(support.get(code, 0.0), best)
        if clause_kind == "drug_history":
            support = {code: score + 0.05 for code, score in support.items()}
        ordered = sorted(support.items(), key=lambda item: (-item[1], item[0]))
        return GraphRetrievalResult(candidate_codes=[code for code, _ in ordered[:6]], support=support)


def _strip_strength(text: str) -> str:
    return re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)\b", "", text, flags=re.IGNORECASE).strip()
