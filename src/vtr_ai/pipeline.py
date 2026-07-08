from __future__ import annotations

from .abbreviation import AbbreviationExpander
from .assertion import infer_assertions
from .config import AppConfig
from .entity_recognizer import build_entity_recognizer
from .graph_reasoner import GraphReasoner
from .graph_retriever import GraphRetriever
from .knowledge_base import KnowledgeBase
from .knowledge_graph import KnowledgeGraphIndex
from .normalization import ConceptNormalizer
from .ontology import build_patient_case
from .patient_graph import build_patient_graph
from .postprocess import deduplicate_entities
from .preprocess import build_document
from .relation import link_relations
from .rule_engine import RuleEngine
from .schemas import Entity, PatientCase


class ClinicalNlpPipeline:
    def __init__(self, config: AppConfig, knowledge_base: KnowledgeBase) -> None:
        self.config = config
        self.knowledge_base = knowledge_base
        self.abbreviation_expander = AbbreviationExpander.from_path(config.knowledge_base.abbreviations_path)
        self.entity_recognizer = build_entity_recognizer(config, knowledge_base)
        self.entity_recognizer_status = self._describe_entity_recognizer()
        self.normalizer = ConceptNormalizer(knowledge_base, config.matching, self.abbreviation_expander)
        self.knowledge_graph = KnowledgeGraphIndex.from_knowledge_base(knowledge_base)
        self.graph_retriever = GraphRetriever(self.knowledge_graph)
        self.graph_reasoner = GraphReasoner(self.graph_retriever)
        self.rule_engine = RuleEngine()

    def _describe_entity_recognizer(self) -> str:
        recognizer = self.entity_recognizer
        if hasattr(recognizer, "model_backend"):
            model_backend = recognizer.model_backend
            if getattr(model_backend, "load_error", None):
                return f"hybrid(rule+fallback): {model_backend.load_error}"
            return "hybrid(rule+model)"
        if hasattr(recognizer, "load_error"):
            if getattr(recognizer, "load_error", None):
                return f"model-fallback: {recognizer.load_error}"
            return "model"
        return "rule"

    def process_text(self, text: str) -> list[Entity]:
        return self.process_case(text).entities

    def process_case(self, text: str) -> PatientCase:
        document = build_document(text, self.abbreviation_expander)
        entities = self.entity_recognizer.extract(document)
        entities = deduplicate_entities(entities)
        patient_graph = build_patient_graph(document, entities)
        entities = self.normalizer.apply(entities, document)
        entities = self.graph_reasoner.apply(entities, document, patient_graph)
        for entity in entities:
            entity.assertions = infer_assertions(document, entity, self.config.rules.assertion_window)
        relations = link_relations(entities)
        patient_case = build_patient_case(document.text, entities, relations)
        patient_case = self.rule_engine.apply(patient_case)
        patient_case.entities = deduplicate_entities(patient_case.entities)
        return patient_case
