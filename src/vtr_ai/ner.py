from __future__ import annotations

from .config import RulesConfig
from .extractor import extract_entities_from_mentions
from .mention_proposal import propose_mentions
from .schemas import Entity


def extract_entities(text: str, config: RulesConfig) -> list[Entity]:
    return extract_entities_from_mentions(text, propose_mentions(text, config))
