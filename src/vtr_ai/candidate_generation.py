from __future__ import annotations

from difflib import SequenceMatcher

from .config import MatchingConfig
from .knowledge_base import KnowledgeRecord
from .schemas import Candidate


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in cleaned.split() if token}


def _score(query: str, record: KnowledgeRecord) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & record.tokens) / max(len(query_tokens), 1)
    fuzzy = SequenceMatcher(None, query.lower(), record.searchable_text).ratio()
    exact_bonus = 0.2 if query.lower() in record.searchable_text else 0.0
    return min(1.0, overlap * 0.55 + fuzzy * 0.45 + exact_bonus)


def rank_candidates(
    query: str,
    records: list[KnowledgeRecord],
    config: MatchingConfig,
) -> list[Candidate]:
    scored = [
        Candidate(code=record.code, label=record.label, score=_score(query, record))
        for record in records
    ]
    filtered = [candidate for candidate in scored if candidate.score >= config.min_confidence]
    filtered.sort(key=lambda item: (-item.score, item.code))
    return filtered[: config.max_candidates]

