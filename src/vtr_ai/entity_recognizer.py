from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .extractor import extract_entities_from_mentions
from .knowledge_base import KnowledgeBase
from .mention_proposal import propose_mentions
from .model_ner import BaseNerRuntime, load_ner_runtime
from .postprocess import deduplicate_entities
from .schemas import Document, Entity, MentionProposal


def _map_mentions_to_raw(document: Document, mentions: list[MentionProposal]) -> list[MentionProposal]:
    if document.expanded_text == document.text:
        return mentions
    mapped: list[MentionProposal] = []
    for mention in mentions:
        if mention.end <= mention.start or mention.end > len(document.expanded_to_raw):
            continue
        raw_start = document.expanded_to_raw[mention.start]
        raw_end = document.expanded_to_raw[mention.end - 1] + 1
        mapped.append(
            MentionProposal(
                text=document.text[raw_start:raw_end],
                start=raw_start,
                end=raw_end,
                proposed_type=mention.proposed_type,
                source=mention.source,
            )
        )
    return mapped


def _predictions_to_mentions(document: Document, runtime: BaseNerRuntime) -> list[MentionProposal]:
    predictions = runtime.predict(document.expanded_text)
    mentions: list[MentionProposal] = []
    for prediction in predictions:
        mentions.append(
            MentionProposal(
                text=document.expanded_text[prediction.start : prediction.end],
                start=prediction.start,
                end=prediction.end,
                proposed_type=prediction.label,
                source="model-runtime",
            )
        )
    return _map_mentions_to_raw(document, mentions)


@dataclass
class BaseEntityRecognizer:
    config: AppConfig
    knowledge_base: KnowledgeBase

    def extract(self, document: Document) -> list[Entity]:
        raise NotImplementedError


class RuleBasedEntityRecognizer(BaseEntityRecognizer):
    def extract(self, document: Document) -> list[Entity]:
        mentions = propose_mentions(document.text, self.config.rules, self.knowledge_base)
        if document.expanded_text != document.text:
            expanded_mentions = propose_mentions(document.expanded_text, self.config.rules, self.knowledge_base)
            mentions.extend(_map_mentions_to_raw(document, expanded_mentions))
        return extract_entities_from_mentions(document.text, mentions)


class ModelBackedEntityRecognizer(BaseEntityRecognizer):
    def __init__(self, config: AppConfig, knowledge_base: KnowledgeBase) -> None:
        super().__init__(config, knowledge_base)
        self.runtime: BaseNerRuntime | None = None
        self.load_error: str | None = None
        try:
            self.runtime = load_ner_runtime(config.ner, knowledge_base)
        except RuntimeError as exc:
            self.load_error = str(exc)
            if not config.ner.fallback_to_rules:
                raise

    def extract(self, document: Document) -> list[Entity]:
        if self.runtime is None:
            return []
        mentions = _predictions_to_mentions(document, self.runtime)
        entities = extract_entities_from_mentions(document.text, mentions)
        return entities

    @property
    def is_available(self) -> bool:
        return self.runtime is not None


class HybridEntityRecognizer(BaseEntityRecognizer):
    def __init__(self, config: AppConfig, knowledge_base: KnowledgeBase) -> None:
        super().__init__(config, knowledge_base)
        self.rule_backend = RuleBasedEntityRecognizer(config, knowledge_base)
        self.model_backend = ModelBackedEntityRecognizer(config, knowledge_base)

    def extract(self, document: Document) -> list[Entity]:
        entities = self.rule_backend.extract(document)
        if self.config.ner.enable_model_backend:
            entities.extend(self.model_backend.extract(document))
        return deduplicate_entities(entities)


def build_entity_recognizer(config: AppConfig, knowledge_base: KnowledgeBase) -> BaseEntityRecognizer:
    backend = config.ner.backend.lower()
    if backend == "rule":
        return RuleBasedEntityRecognizer(config, knowledge_base)
    if backend == "model":
        return ModelBackedEntityRecognizer(config, knowledge_base)
    return HybridEntityRecognizer(config, knowledge_base)
