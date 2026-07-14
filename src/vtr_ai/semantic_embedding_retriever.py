from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .knowledge_base import KnowledgeRecord


@dataclass(slots=True)
class RetrievedCandidate:
    code: str
    score: float
    label: str
    matched_alias: str


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    folded = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d")
    cleaned = re.sub(r"[^a-z0-9]+", " ", folded)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _token_features(text: str) -> Counter[str]:
    normalized = _normalize(text)
    if not normalized:
        return Counter()
    return Counter(f"tok:{token}" for token in normalized.split())


def _char_ngram_features(text: str, min_n: int = 3, max_n: int = 5) -> Counter[str]:
    normalized = _normalize(text).replace(" ", "_")
    if not normalized:
        return Counter()
    counts: Counter[str] = Counter()
    for n in range(min_n, max_n + 1):
        if len(normalized) < n:
            continue
        for idx in range(len(normalized) - n + 1):
            counts[f"chr:{normalized[idx:idx+n]}"] += 1
    return counts


def _vectorize(text: str) -> Counter[str]:
    vector = _token_features(text)
    vector.update(_char_ngram_features(text))
    return vector


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right.get(key, 0.0) for key in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def best_embedding_match(query: str, record: KnowledgeRecord) -> tuple[float, str]:
    query_vector = _vectorize(query)
    best_score = 0.0
    best_alias = record.label
    for alias in [record.label, *record.aliases]:
        score = _cosine_similarity(query_vector, _vectorize(alias))
        if score > best_score:
            best_score = score
            best_alias = alias
    return best_score, best_alias


def rank_by_embedding(query: str, records: list[KnowledgeRecord], top_k: int) -> list[RetrievedCandidate]:
    ranked: list[RetrievedCandidate] = []
    for record in records:
        score, matched_alias = best_embedding_match(query, record)
        if score <= 0.0:
            continue
        ranked.append(
            RetrievedCandidate(
                code=record.code,
                score=score,
                label=record.label,
                matched_alias=matched_alias,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.code))
    return ranked[:top_k]
