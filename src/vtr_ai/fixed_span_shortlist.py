from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .candidate_generation import rank_candidates_indexed, score_record
from .config import MatchingConfig
from .knowledge_base import KnowledgeBase, KnowledgeRecord
from .semantic_embedding_retriever import rank_by_embedding


STRENGTH_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg/ml|mcg/ml|mg|mcg|g|ml)\b", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"\b(po|iv|im|sc|subq|bid|tid|qid|daily|once|prn|nebs?|nebulizer)\b", re.IGNORECASE)


@dataclass(slots=True)
class ShortlistCandidate:
    code: str
    score: float
    source: str
    matched_alias: str


def _normalize_match_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-zà-ỹđ0-9]+", " ", lowered, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", lowered).strip()


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    folded = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def _normalize_cache_key(text: str) -> str:
    normalized = _normalize_match_text(text)
    return re.sub(r"\s{2,}", " ", _ascii_fold(normalized)).strip()


def _strip_terminal_strength(text: str) -> str:
    return re.sub(
        r"\s+\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,-:")


def _extract_strength_signature(text: str) -> tuple[str, str] | None:
    match = STRENGTH_PATTERN.search(text.lower())
    if not match:
        return None
    return match.group(1).replace(",", "."), match.group(2).lower()


def _record_aliases(record: KnowledgeRecord) -> list[str]:
    return [record.label, *record.aliases]


def _alias_similarity(query_norm: str, alias_norm: str) -> float:
    if not query_norm or not alias_norm:
        return 0.0
    if query_norm == alias_norm:
        return 1.0
    if query_norm in alias_norm or alias_norm in query_norm:
        return 0.94
    query_tokens = set(query_norm.split())
    alias_tokens = set(alias_norm.split())
    token_overlap = 0.0
    if query_tokens and alias_tokens:
        token_overlap = len(query_tokens & alias_tokens) / max(len(query_tokens), len(alias_tokens))
    fuzzy = SequenceMatcher(None, query_norm, alias_norm).ratio()
    return max(token_overlap, fuzzy * 0.9)


def _semantic_similarity(query: str, alias: str) -> float:
    return _alias_similarity(_normalize_cache_key(query), _normalize_cache_key(alias))


def _structured_match_score(query: str, alias: str, entity_type: str) -> tuple[float | None, bool]:
    if entity_type != "THUỐC":
        return None, False

    ingredient_query = _normalize_cache_key(_strip_terminal_strength(query))
    ingredient_alias = _normalize_cache_key(_strip_terminal_strength(alias))
    query_strength = _extract_strength_signature(query)
    alias_strength = _extract_strength_signature(alias)

    score = 0.0
    if ingredient_query and ingredient_alias:
        if ingredient_query == ingredient_alias:
            score += 0.7
        elif ingredient_query in ingredient_alias or ingredient_alias in ingredient_query:
            score += 0.45

    mismatch = False
    if query_strength and alias_strength:
        if query_strength == alias_strength:
            score += 0.3
        else:
            mismatch = True
            score -= 0.25
    elif query_strength and not alias_strength:
        score -= 0.05

    route_match = len(set(ROUTE_PATTERN.findall(query.lower())) & set(ROUTE_PATTERN.findall(alias.lower())))
    if route_match:
        score += 0.05

    return max(0.0, min(1.0, score)), mismatch


def _exact_match_codes(query: str, records: list[KnowledgeRecord], entity_type: str) -> list[str]:
    normalized_query = _normalize_cache_key(query)
    if not normalized_query:
        return []

    exact_codes: list[str] = []
    seen: set[str] = set()
    normalized_queries = {normalized_query}
    if entity_type == "THUỐC":
        stripped_norm = _normalize_cache_key(_strip_terminal_strength(query))
        if stripped_norm:
            normalized_queries.add(stripped_norm)

    for record in records:
        for alias in _record_aliases(record):
            alias_norm = _normalize_cache_key(alias)
            stripped_alias_norm = _normalize_cache_key(_strip_terminal_strength(alias))
            if alias_norm not in normalized_queries and stripped_alias_norm not in normalized_queries:
                continue
            if record.code not in seen:
                seen.add(record.code)
                exact_codes.append(record.code)
            break
    return exact_codes


def _gather_candidate_pool(
    mention: str,
    entity_type: str,
    knowledge_base: KnowledgeBase,
    pool_size: int,
    min_confidence: float,
) -> tuple[list[KnowledgeRecord], list[KnowledgeRecord], list[KnowledgeRecord]]:
    records = knowledge_base.icd10 if entity_type == "CHẨN_ĐOÁN" else knowledge_base.rxnorm
    exact_codes = _exact_match_codes(mention, records, entity_type)
    exact_pool = [record for record in records if record.code in set(exact_codes)]
    indexed = rank_candidates_indexed(
        mention,
        records,
        MatchingConfig(max_candidates=pool_size, min_confidence=min_confidence),
        knowledge_base,
        entity_type,
    )
    by_code = {record.code: record for record in records}
    indexed_pool = [by_code[candidate.code] for candidate in indexed if candidate.code in by_code]
    embedding_base = records
    embedding_pool = [
        by_code[item.code]
        for item in rank_by_embedding(mention, embedding_base, top_k=max(pool_size, 30))
        if item.code in by_code
    ] if embedding_base else []
    return exact_pool, indexed_pool, embedding_pool


def rank_fixed_span_shortlist_records(
    mention: str,
    entity_type: str,
    knowledge_base: KnowledgeBase,
    shortlist_size: int,
    min_confidence: float,
) -> list[ShortlistCandidate]:
    records = knowledge_base.icd10 if entity_type == "CHẨN_ĐOÁN" else knowledge_base.rxnorm
    if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"} or not records:
        return []

    exact_pool, indexed_pool, embedding_pool = _gather_candidate_pool(
        mention=mention,
        entity_type=entity_type,
        knowledge_base=knowledge_base,
        pool_size=max(shortlist_size * 6, 30),
        min_confidence=min_confidence,
    )

    candidate_pool: dict[str, KnowledgeRecord] = {}
    for record in exact_pool + indexed_pool + embedding_pool:
        candidate_pool[record.code] = record
    if not candidate_pool:
        return []

    exact_codes = {record.code for record in exact_pool}
    embedding_scores = {
        item.code: item
        for item in rank_by_embedding(mention, list(candidate_pool.values()), top_k=len(candidate_pool))
    }
    scored: list[ShortlistCandidate] = []
    for record in candidate_pool.values():
        best_semantic = 0.0
        best_structured: float | None = None
        structured_mismatch = False
        for alias in _record_aliases(record):
            semantic = _semantic_similarity(mention, alias)
            structured_score, mismatch = _structured_match_score(mention, alias, entity_type)
            if semantic > best_semantic:
                best_semantic = semantic
            if structured_score is not None and (best_structured is None or structured_score > best_structured):
                best_structured = structured_score
            structured_mismatch = structured_mismatch or mismatch

        lexical = max(score_record(mention, record), score_record(_strip_terminal_strength(mention), record))
        embedding_match = embedding_scores.get(record.code)
        embedding_score = embedding_match.score if embedding_match else 0.0
        structured = best_structured or 0.0
        final_score = lexical * 0.4 + best_semantic * 0.2 + embedding_score * 0.2 + structured * 0.2
        if record.code in exact_codes:
            final_score += 0.35
        if entity_type == "THUỐC" and structured_mismatch:
            final_score -= 0.12
        if record.code in exact_codes:
            source = "exact"
        else:
            source = "embedding" if embedding_score >= lexical else "indexed"
        scored.append(
            ShortlistCandidate(
                code=record.code,
                score=final_score,
                source=source,
                matched_alias=embedding_match.matched_alias if embedding_match else record.label,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.code))
    return scored[:shortlist_size]


def rank_fixed_span_shortlist(
    mention: str,
    entity_type: str,
    knowledge_base: KnowledgeBase,
    shortlist_size: int,
    min_confidence: float,
) -> list[str]:
    return [
        candidate.code
        for candidate in rank_fixed_span_shortlist_records(
            mention=mention,
            entity_type=entity_type,
            knowledge_base=knowledge_base,
            shortlist_size=shortlist_size,
            min_confidence=min_confidence,
        )
    ]
