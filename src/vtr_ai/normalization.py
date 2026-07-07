from __future__ import annotations

from .candidate_generation import rank_candidates
from .config import MatchingConfig
from .knowledge_base import KnowledgeBase
from .schemas import Entity


class DiagnosisNormalizer:
    def __init__(self, knowledge_base: KnowledgeBase, config: MatchingConfig) -> None:
        self.knowledge_base = knowledge_base
        self.config = config

    def normalize(self, entity: Entity) -> list[str]:
        return [candidate.code for candidate in rank_candidates(entity.text, self.knowledge_base.icd10, self.config)]


class DrugNormalizer:
    def __init__(self, knowledge_base: KnowledgeBase, config: MatchingConfig) -> None:
        self.knowledge_base = knowledge_base
        self.config = config

    def normalize(self, entity: Entity) -> list[str]:
        return [candidate.code for candidate in rank_candidates(entity.text, self.knowledge_base.rxnorm, self.config)]


class ConceptNormalizer:
    def __init__(self, knowledge_base: KnowledgeBase, config: MatchingConfig) -> None:
        self.diagnosis = DiagnosisNormalizer(knowledge_base, config)
        self.drug = DrugNormalizer(knowledge_base, config)

    def apply(self, entities: list[Entity]) -> list[Entity]:
        for entity in entities:
            if entity.entity_type == "CHẨN_ĐOÁN":
                entity.candidates = self.diagnosis.normalize(entity)
            elif entity.entity_type == "THUỐC":
                entity.candidates = self.drug.normalize(entity)
            else:
                entity.candidates = []
        return entities

