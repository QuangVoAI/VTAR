from __future__ import annotations

from .assertion import infer_assertions
from .config import AppConfig
from .knowledge_base import KnowledgeBase
from .ner import extract_entities
from .normalization import ConceptNormalizer
from .ontology import build_patient_case
from .postprocess import deduplicate_entities
from .preprocess import build_document
from .relation import link_relations
from .rule_engine import RuleEngine
from .schemas import Entity, PatientCase


class ClinicalNlpPipeline:
    def __init__(self, config: AppConfig, knowledge_base: KnowledgeBase) -> None:
        self.config = config
        self.knowledge_base = knowledge_base
        self.normalizer = ConceptNormalizer(knowledge_base, config.matching)
        self.rule_engine = RuleEngine()

    def process_text(self, text: str) -> list[Entity]:
        return self.process_case(text).entities

    def process_case(self, text: str) -> PatientCase:
        document = build_document(text)
        entities = extract_entities(document.text, self.config.rules)
        entities = deduplicate_entities(entities)
        entities = self.normalizer.apply(entities)
        for entity in entities:
            entity.assertions = infer_assertions(document.text, entity, self.config.rules.assertion_window)
        relations = link_relations(entities)
        patient_case = build_patient_case(document.text, entities, relations)
        patient_case = self.rule_engine.apply(patient_case)
        patient_case.entities = deduplicate_entities(patient_case.entities)
        return patient_case
