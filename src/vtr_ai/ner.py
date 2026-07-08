from __future__ import annotations

from .config import RulesConfig
from .knowledge_base import KnowledgeBase
from .schemas import Document, Entity


def extract_entities(document: Document, config: RulesConfig, knowledge_base: KnowledgeBase) -> list[Entity]:
    from .entity_recognizer import RuleBasedEntityRecognizer

    recognizer = RuleBasedEntityRecognizer(
        config=type("Tmp", (), {"rules": config, "ner": type("N", (), {"enable_model_backend": False})()})(),  # type: ignore[arg-type]
        knowledge_base=knowledge_base,
    )
    return recognizer.extract(document)
