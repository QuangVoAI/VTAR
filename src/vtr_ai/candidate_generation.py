from __future__ import annotations

from difflib import SequenceMatcher

from .config import MatchingConfig
from .knowledge_base import KnowledgeBase, KnowledgeRecord
from .schemas import Candidate


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in cleaned.split() if token}


def score_record(query: str, record: KnowledgeRecord) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & record.tokens) / max(len(query_tokens), 1)
    fuzzy = SequenceMatcher(None, query.lower(), record.searchable_text).ratio()
    exact_bonus = 0.2 if query.lower() in record.searchable_text else 0.0
    # Clamp the final score to [0.0, 1.0] since overlap * 0.55 + fuzzy * 0.45 + exact_bonus 
    # can exceed 1.0 (e.g. when both ratios are 1.0 and exact_bonus is applied).
    return min(1.0, overlap * 0.55 + fuzzy * 0.45 + exact_bonus)


def rank_candidates(
    query: str,
    records: list[KnowledgeRecord],
    config: MatchingConfig,
) -> list[Candidate]:
    scored = [
        Candidate(code=record.code, label=record.label, score=score_record(query, record))
        for record in records
    ]
    filtered = [candidate for candidate in scored if candidate.score >= config.min_confidence]
    filtered.sort(key=lambda item: (-item.score, item.code))
    return filtered[: config.max_candidates]


def rank_candidates_indexed(
    query: str,
    records: list[KnowledgeRecord],
    config: MatchingConfig,
    knowledge_base: KnowledgeBase,
    entity_type: str,
) -> list[Candidate]:
    indexed_records = _select_indexed_records(query, records, knowledge_base, entity_type)
    return rank_candidates(query, indexed_records, config)


def _select_indexed_records(
    query: str,
    records: list[KnowledgeRecord],
    knowledge_base: KnowledgeBase,
    entity_type: str,
) -> list[KnowledgeRecord]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return records

    if entity_type == "CHẨN_ĐOÁN":
        token_index = knowledge_base.diagnosis_token_index or {}
        alias_index = knowledge_base.diagnosis_alias_index or {}
    else:
        token_index = knowledge_base.drug_token_index or {}
        alias_index = knowledge_base.drug_alias_index or {}

    normalized_query = " ".join(sorted(query_tokens))
    candidates: dict[str, KnowledgeRecord] = {}

    for record in alias_index.get(normalized_query, []):
        candidates[record.code] = record

    for token in query_tokens:
        for record in token_index.get(token, []):
            candidates[record.code] = record

    if candidates:
        candidate_list = list(candidates.values())
        if len(candidate_list) > 400:
            candidate_list.sort(
                key=lambda record: (
                    -len(query_tokens & record.tokens),
                    -SequenceMatcher(None, query.lower(), record.searchable_text).ratio(),
                    record.code,
                )
            )
            return candidate_list[:400]
        return candidate_list
    return records
