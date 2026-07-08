from __future__ import annotations

import re

from .abbreviation import AbbreviationExpander
from .candidate_generation import rank_candidates_indexed, score_record
from .config import MatchingConfig
from .knowledge_base import KnowledgeBase, KnowledgeRecord
from .section_parser import find_enclosing_clause
from .schemas import Document, Entity


STRENGTH_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg/ml|mcg/ml|mg|mcg|g|ml)\b", re.IGNORECASE)


def _clean_diagnosis_query(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^[\s:;-]+", "", cleaned)
    cleaned = re.sub(r"^(?:nghi ngờ|tiền sử lâu dài của|tiền sử|mới được chẩn đoán|là)\s+", "", cleaned, flags=re.IGNORECASE)
    lowered = cleaned.lower()
    stop_positions = [
        lowered.find(stop_token)
        for stop_token in (",", ";", " nghi ", " cách đây ", " điều trị ", " nhưng ", " biến chứng ", " có ", " và ")
        if lowered.find(stop_token) != -1
    ]
    if stop_positions:
        cleaned = cleaned[: min(stop_positions)]
    cleaned = re.sub(r"\([^)]*$", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _clean_drug_query(text: str) -> str:
    cleaned = text.strip()
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.sub(r"^\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)\s+", "", cleaned, flags=re.IGNORECASE)
    for stop_token in (" và ", ", cùng ", " cùng ", ";"):
        lowered = cleaned.lower()
        pos = lowered.find(stop_token)
        if pos != -1:
            cleaned = cleaned[:pos].strip()
            break
    cleaned = re.sub(r"^(?:được cho dùng|được cho|nhận|cho po|iv|po|uống)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:bằng liều cao|liều cao)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:x\s*\d+|po|iv|im|bid|tid|qid|daily|once|nebs?|nebulizer)\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,-:")
    return cleaned


def _strip_terminal_strength(text: str) -> str:
    return re.sub(
        r"\s+\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,-:")


def _normalize_match_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-zà-ỹđ0-9]+", " ", lowered, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", lowered).strip()


def _extract_strength_signature(text: str) -> tuple[str, str] | None:
    match = STRENGTH_PATTERN.search(text.lower())
    if not match:
        return None
    return match.group(1).replace(",", "."), match.group(2).lower()


def _record_aliases(record: KnowledgeRecord) -> list[str]:
    return [record.label, *record.aliases]


def _best_normalized_alias(record: KnowledgeRecord, query: str) -> str:
    query_norm = _normalize_match_text(query)
    aliases = _record_aliases(record)
    return max(aliases, key=lambda item: _alias_similarity(query_norm, _normalize_match_text(item)))


def _alias_similarity(query_norm: str, alias_norm: str) -> float:
    if not query_norm or not alias_norm:
        return 0.0
    if query_norm == alias_norm:
        return 1.0
    if query_norm in alias_norm or alias_norm in query_norm:
        return 0.9
    query_tokens = set(query_norm.split())
    alias_tokens = set(alias_norm.split())
    if not query_tokens or not alias_tokens:
        return 0.0
    return len(query_tokens & alias_tokens) / max(len(query_tokens), len(alias_tokens))


def _contextual_score_bonus(
    query: str,
    record: KnowledgeRecord,
    entity_type: str,
    clause_kind: str | None,
) -> float:
    query_norm = _normalize_match_text(query)
    best_alias = _best_normalized_alias(record, query)
    best_alias_norm = _normalize_match_text(best_alias)
    bonus = 0.0

    alias_similarity = _alias_similarity(query_norm, best_alias_norm)
    if alias_similarity >= 0.999:
        bonus += 0.28
    elif alias_similarity >= 0.9:
        bonus += 0.18
    elif alias_similarity >= 0.6:
        bonus += 0.08

    if entity_type == "THUỐC":
        query_ingredient = _normalize_match_text(_strip_terminal_strength(query))
        record_ingredient = _normalize_match_text(_strip_terminal_strength(best_alias))
        if query_ingredient and query_ingredient == record_ingredient:
            bonus += 0.22
        elif query_ingredient and record_ingredient and (
            query_ingredient in record_ingredient or record_ingredient in query_ingredient
        ):
            bonus += 0.12

        query_strength = _extract_strength_signature(query)
        record_strength = _extract_strength_signature(best_alias)
        if query_strength and record_strength:
            if query_strength == record_strength:
                bonus += 0.2
            else:
                bonus -= 0.18
        elif query_strength and not record_strength:
            bonus -= 0.04

        if clause_kind == "drug_history":
            bonus += 0.04

    if entity_type == "CHẨN_ĐOÁN":
        if clause_kind in {"diagnosis_history", "diagnosis_current"}:
            bonus += 0.04
        if len(query_norm.split()) <= 2 and alias_similarity < 0.9:
            bonus -= 0.06

    return bonus


def _rank_candidates_with_context(
    query: str,
    records: list[KnowledgeRecord],
    config: MatchingConfig,
    entity_type: str,
    clause_kind: str | None,
    knowledge_base: KnowledgeBase,
) -> list[str]:
    base_ranked = rank_candidates_indexed(query, records, config, knowledge_base, entity_type)
    allowed_codes = {candidate.code for candidate in base_ranked}
    rescored = []
    for record in records:
        if allowed_codes and record.code not in allowed_codes:
            continue
        base_score = score_record(query, record)
        final_score = max(0.0, min(1.0, base_score + _contextual_score_bonus(query, record, entity_type, clause_kind)))
        if final_score >= config.min_confidence:
            rescored.append((record.code, final_score))
    rescored.sort(key=lambda item: (-item[1], item[0]))
    return [code for code, _score in rescored[: config.max_candidates]]


class DiagnosisNormalizer:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: MatchingConfig,
        abbreviation_expander: AbbreviationExpander | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.config = config
        self.abbreviation_expander = abbreviation_expander

    def normalize(self, entity: Entity, clause_kind: str | None = None) -> list[str]:
        query = entity.text
        if self.abbreviation_expander is not None:
            query, _ = self.abbreviation_expander.expand_text_with_mapping(entity.text)
        query = _clean_diagnosis_query(query)
        if not query:
            return []
        return _rank_candidates_with_context(
            query,
            self.knowledge_base.icd10,
            self.config,
            "CHẨN_ĐOÁN",
            clause_kind,
            self.knowledge_base,
        )


class DrugNormalizer:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: MatchingConfig,
        abbreviation_expander: AbbreviationExpander | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.config = config
        self.abbreviation_expander = abbreviation_expander

    def normalize(self, entity: Entity, clause_kind: str | None = None) -> list[str]:
        query = entity.text
        if self.abbreviation_expander is not None:
            query, _ = self.abbreviation_expander.expand_text_with_mapping(entity.text)
        query = _clean_drug_query(query)
        variants: list[str] = [query]
        stripped = _strip_terminal_strength(query)
        if stripped and stripped != query:
            variants.append(stripped)

        merged: dict[str, float] = {}
        for variant in variants:
            if not variant:
                continue
            ranked_codes = _rank_candidates_with_context(
                variant,
                self.knowledge_base.rxnorm,
                self.config,
                "THUỐC",
                clause_kind,
                self.knowledge_base,
            )
            for rank, code in enumerate(ranked_codes):
                merged[code] = max(merged.get(code, -1.0), float(len(ranked_codes) - rank))
        return [code for code, _score in sorted(merged.items(), key=lambda item: (-item[1], item[0]))[: self.config.max_candidates]]


class ConceptNormalizer:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: MatchingConfig,
        abbreviation_expander: AbbreviationExpander | None = None,
    ) -> None:
        self.diagnosis = DiagnosisNormalizer(knowledge_base, config, abbreviation_expander)
        self.drug = DrugNormalizer(knowledge_base, config, abbreviation_expander)

    def apply(self, entities: list[Entity], document: Document | None = None) -> list[Entity]:
        for entity in entities:
            clause_kind = None
            if document is not None:
                clause = find_enclosing_clause(document.clauses, entity.start, entity.end)
                clause_kind = clause.anchor_kind if clause is not None else None
            if document is not None and not _context_allows_mapping(document, entity):
                entity.candidates = []
                continue
            if entity.entity_type == "CHẨN_ĐOÁN":
                entity.candidates = self.diagnosis.normalize(entity, clause_kind)
            elif entity.entity_type == "THUỐC":
                entity.candidates = self.drug.normalize(entity, clause_kind)
            else:
                entity.candidates = []
        return entities


def _context_allows_mapping(document: Document, entity: Entity) -> bool:
    clause = find_enclosing_clause(document.clauses, entity.start, entity.end)
    if clause is None:
        return True
    if clause.anchor_kind == "lab_results" and entity.entity_type in {"CHẨN_ĐOÁN", "THUỐC"}:
        return False
    return True
