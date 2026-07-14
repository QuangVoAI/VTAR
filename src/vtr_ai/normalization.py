from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .abbreviation import AbbreviationExpander
from .candidate_generation import rank_candidates_indexed, score_record
from .config import MatchingConfig
from .knowledge_base import KnowledgeBase, KnowledgeRecord
from .section_parser import find_enclosing_clause
from .schemas import Document, Entity


STRENGTH_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg/ml|mcg/ml|mg|mcg|g|ml)\b", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"\b(po|iv|im|sc|subq|bid|tid|qid|daily|once|prn|nebs?|nebulizer)\b", re.IGNORECASE)


@dataclass(slots=True)
class MappingCandidateScore:
    code: str
    candidate_text: str
    lexical_score: float
    semantic_score: float
    structured_score: float | None
    strength_mismatch: bool = False
    final_score: float = 0.0


@dataclass(slots=True)
class MappingDecision:
    selected_codes: list[str]
    confidence: float
    margin: float | None
    match_mode: str
    top_candidates: list[MappingCandidateScore] = field(default_factory=list)


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


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    folded = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def _normalize_cache_key(text: str) -> str:
    normalized = _normalize_match_text(text)
    return re.sub(r"\s{2,}", " ", _ascii_fold(normalized)).strip()


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


def _semantic_similarity(query: str, alias: str) -> float:
    query_norm = _normalize_cache_key(query)
    alias_norm = _normalize_cache_key(alias)
    if not query_norm or not alias_norm:
        return 0.0
    if query_norm == alias_norm:
        return 1.0
    if query_norm in alias_norm or alias_norm in query_norm:
        return 0.94
    return _alias_similarity(query_norm, alias_norm)


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
    for record in records:
        aliases = [record.label, *record.aliases]
        for alias in aliases:
            alias_norm = _normalize_cache_key(alias)
            if alias_norm != normalized_query:
                continue
            if record.code not in seen:
                seen.add(record.code)
                exact_codes.append(record.code)
            break

    if entity_type == "THUỐC":
        stripped_query = _strip_terminal_strength(query)
        stripped_norm = _normalize_cache_key(stripped_query)
        if stripped_norm and stripped_norm != normalized_query:
            for record in records:
                aliases = [record.label, *record.aliases]
                for alias in aliases:
                    if _normalize_cache_key(_strip_terminal_strength(alias)) != stripped_norm:
                        continue
                    if record.code not in seen:
                        seen.add(record.code)
                        exact_codes.append(record.code)
                    break
    return exact_codes


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


def _query_too_long_for_mapping(query: str, entity_type: str) -> bool:
    normalized = _normalize_match_text(query)
    token_count = len(normalized.split())
    if entity_type == "CHẨN_ĐOÁN":
        return len(normalized) > 48 or token_count > 8
    if entity_type == "THUỐC":
        return len(normalized) > 64 or token_count > 10
    return False


def _score_mapping_candidates(
    query: str,
    records: list[KnowledgeRecord],
    config: MatchingConfig,
    entity_type: str,
    clause_kind: str | None,
    knowledge_base: KnowledgeBase,
) -> list[MappingCandidateScore]:
    base_ranked = rank_candidates_indexed(query, records, config, knowledge_base, entity_type)
    allowed_codes = {candidate.code for candidate in base_ranked}
    rescored: list[MappingCandidateScore] = []
    for record in records:
        if allowed_codes and record.code not in allowed_codes:
            continue
        lexical_score = score_record(query, record)
        best_alias = _best_normalized_alias(record, query)
        semantic_score = _semantic_similarity(query, best_alias)
        structured_score, strength_mismatch = _structured_match_score(query, best_alias, entity_type)

        final_score = lexical_score * 0.45 + semantic_score * 0.45
        if structured_score is not None:
            final_score += structured_score * 0.1
        final_score = max(0.0, min(1.0, final_score + _contextual_score_bonus(query, record, entity_type, clause_kind)))
        if final_score >= config.min_confidence:
            rescored.append(
                MappingCandidateScore(
                    code=record.code,
                    candidate_text=best_alias,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    structured_score=structured_score,
                    strength_mismatch=strength_mismatch,
                    final_score=final_score,
                )
            )
    rescored.sort(key=lambda item: (-item.final_score, item.code))
    return rescored[: config.max_candidates]


def _select_codes_from_scores(
    scored_candidates: list[MappingCandidateScore],
    config: MatchingConfig,
    entity_type: str,
) -> MappingDecision:
    if not scored_candidates:
        return MappingDecision(selected_codes=[], confidence=0.0, margin=None, match_mode="none")

    best = scored_candidates[0]
    second = scored_candidates[1] if len(scored_candidates) > 1 else None
    margin = None if second is None else max(0.0, best.final_score - second.final_score)

    keep_threshold = max(config.min_confidence, best.final_score - (0.03 if entity_type == "CHẨN_ĐOÁN" else 0.05))
    selected: list[str] = []
    for candidate in scored_candidates:
        if candidate.final_score < keep_threshold:
            continue
        if candidate.code not in selected:
            selected.append(candidate.code)
    selected = selected[: config.max_candidates]
    return MappingDecision(
        selected_codes=selected,
        confidence=best.final_score,
        margin=margin,
        match_mode="hybrid",
        top_candidates=scored_candidates,
    )


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
        self.cache: dict[str, MappingDecision] = {}

    def normalize(self, entity: Entity, clause_kind: str | None = None) -> list[str]:
        query = entity.text
        if self.abbreviation_expander is not None:
            query, _ = self.abbreviation_expander.expand_text_with_mapping(entity.text)
        query = _clean_diagnosis_query(query)
        if not query or _query_too_long_for_mapping(query, "CHẨN_ĐOÁN"):
            return []
        cache_key = f"CHẨN_ĐOÁN||{_normalize_cache_key(query)}"
        if cache_key in self.cache:
            return list(self.cache[cache_key].selected_codes)

        exact_codes = _exact_match_codes(query, self.knowledge_base.icd10, "CHẨN_ĐOÁN")
        if exact_codes:
            decision = MappingDecision(
                selected_codes=exact_codes[: self.config.max_candidates],
                confidence=1.0,
                margin=None,
                match_mode="exact",
            )
            self.cache[cache_key] = decision
            return list(decision.selected_codes)

        scored = _score_mapping_candidates(
            query,
            self.knowledge_base.icd10,
            self.config,
            "CHẨN_ĐOÁN",
            clause_kind,
            self.knowledge_base,
        )
        decision = _select_codes_from_scores(scored, self.config, "CHẨN_ĐOÁN")
        self.cache[cache_key] = decision
        return list(decision.selected_codes)


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
        self.cache: dict[str, MappingDecision] = {}

    def normalize(self, entity: Entity, clause_kind: str | None = None) -> list[str]:
        query = entity.text
        if self.abbreviation_expander is not None:
            query, _ = self.abbreviation_expander.expand_text_with_mapping(entity.text)
        query = _clean_drug_query(query)
        if _query_too_long_for_mapping(query, "THUỐC"):
            return []
        variants: list[str] = [query]
        stripped = _strip_terminal_strength(query)
        if stripped and stripped != query:
            variants.append(stripped)

        cache_key = f"THUỐC||{_normalize_cache_key(query)}"
        if cache_key in self.cache:
            return list(self.cache[cache_key].selected_codes)

        exact_codes = _exact_match_codes(query, self.knowledge_base.rxnorm, "THUỐC")
        if exact_codes:
            decision = MappingDecision(
                selected_codes=exact_codes[: self.config.max_candidates],
                confidence=1.0,
                margin=None,
                match_mode="exact",
            )
            self.cache[cache_key] = decision
            return list(decision.selected_codes)

        merged: dict[str, MappingCandidateScore] = {}
        for variant in variants:
            if not variant:
                continue
            ranked_scores = _score_mapping_candidates(
                variant,
                self.knowledge_base.rxnorm,
                self.config,
                "THUỐC",
                clause_kind,
                self.knowledge_base,
            )
            for candidate in ranked_scores:
                current = merged.get(candidate.code)
                if current is None or candidate.final_score > current.final_score:
                    merged[candidate.code] = candidate
        decision = _select_codes_from_scores(
            sorted(merged.values(), key=lambda item: (-item.final_score, item.code)),
            self.config,
            "THUỐC",
        )
        self.cache[cache_key] = decision
        return list(decision.selected_codes)


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
