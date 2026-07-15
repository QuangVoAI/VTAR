from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

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


class _TransformerEmbeddingBackend:
    def __init__(self, model_name: str) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Transformer embedding requires torch and transformers. "
                "Unset VTR_EMBEDDING_MODEL to use the lexical fallback."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        requested_device = os.getenv("VTR_EMBEDDING_DEVICE", "").strip()
        if requested_device:
            self.device = torch.device(requested_device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.detach().cpu().tolist())
        return vectors


@lru_cache(maxsize=1)
def _get_transformer_backend() -> _TransformerEmbeddingBackend | None:
    model_name = os.getenv("VTR_EMBEDDING_MODEL", "").strip()
    if not model_name:
        return None
    return _TransformerEmbeddingBackend(model_name)


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(left_value * right_value for left_value, right_value in zip(left, right)))


def _rank_by_transformer_embedding(
    query: str,
    records: list[KnowledgeRecord],
    top_k: int,
    backend: _TransformerEmbeddingBackend,
) -> list[RetrievedCandidate]:
    aliases: list[tuple[KnowledgeRecord, str]] = []
    for record in records:
        aliases.append((record, record.label))
        aliases.extend((record, alias) for alias in record.aliases)
    if not aliases:
        return []

    batch_size = int(os.getenv("VTR_EMBEDDING_BATCH_SIZE", "32"))
    texts = [query, *(alias for _, alias in aliases)]
    vectors = backend.encode(texts, batch_size=max(1, batch_size))
    query_vector = vectors[0]
    alias_vectors = vectors[1:]

    best_by_code: dict[str, RetrievedCandidate] = {}
    for (record, alias), alias_vector in zip(aliases, alias_vectors):
        score = _dot(query_vector, alias_vector)
        current = best_by_code.get(record.code)
        if current is None or score > current.score:
            best_by_code[record.code] = RetrievedCandidate(
                code=record.code,
                score=score,
                label=record.label,
                matched_alias=alias,
            )

    ranked = list(best_by_code.values())
    ranked.sort(key=lambda item: (-item.score, item.code))
    return ranked[:top_k]


def rank_by_embedding(query: str, records: list[KnowledgeRecord], top_k: int) -> list[RetrievedCandidate]:
    backend = _get_transformer_backend()
    if backend is not None:
        return _rank_by_transformer_embedding(query, records, top_k, backend)

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
