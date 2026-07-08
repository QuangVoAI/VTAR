from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

from .config import NerConfig
from .knowledge_base import KnowledgeBase
from .schemas import ENTITY_TYPES


@dataclass(slots=True)
class NerPrediction:
    start: int
    end: int
    label: str
    score: float


@dataclass(slots=True)
class BaseNerRuntime:
    def predict(self, text: str) -> list[NerPrediction]:
        raise NotImplementedError


DEFAULT_TRANSFORMER_LABEL_MAP = {
    "SYMPTOM": "TRIỆU_CHỨNG",
    "TRIEU_CHUNG": "TRIỆU_CHỨNG",
    "TRIỆU_CHỨNG": "TRIỆU_CHỨNG",
    "DIAGNOSIS": "CHẨN_ĐOÁN",
    "CHAN_DOAN": "CHẨN_ĐOÁN",
    "CHẨN_ĐOÁN": "CHẨN_ĐOÁN",
    "DRUG": "THUỐC",
    "THUOC": "THUỐC",
    "THUỐC": "THUỐC",
    "LAB_NAME": "TÊN_XÉT_NGHIỆM",
    "TEST_NAME": "TÊN_XÉT_NGHIỆM",
    "TÊN_XÉT_NGHIỆM": "TÊN_XÉT_NGHIỆM",
    "LAB_VALUE": "KẾT_QUẢ_XÉT_NGHIỆM",
    "TEST_VALUE": "KẾT_QUẢ_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM": "KẾT_QUẢ_XÉT_NGHIỆM",
}


@dataclass(slots=True)
class LexicalPattern:
    label: str
    pattern: re.Pattern[str]
    score: float


@dataclass(slots=True)
class LexicalNerRuntime(BaseNerRuntime):
    patterns: list[LexicalPattern]

    def predict(self, text: str) -> list[NerPrediction]:
        predictions: list[NerPrediction] = []
        lowered = text.lower()
        for lexical_pattern in self.patterns:
            for match in lexical_pattern.pattern.finditer(lowered):
                predictions.append(
                    NerPrediction(
                        start=match.start(),
                        end=match.end(),
                        label=lexical_pattern.label,
                        score=lexical_pattern.score,
                    )
                )
        predictions.sort(key=lambda item: (item.start, item.end, item.label))
        return predictions


@dataclass
class TransformersTokenClassifierRuntime(BaseNerRuntime):
    checkpoint_path: Path
    metadata_path: Path | None = None
    aggregation_strategy: str = "simple"
    min_score: float = 0.5

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install transformers/torch or switch ner.provider to lexical."
            ) from exc
        self.label_map = load_transformer_label_map(self.metadata_path)
        try:
            self._pipeline = pipeline(
                "token-classification",
                model=str(self.checkpoint_path),
                tokenizer=str(self.checkpoint_path),
                aggregation_strategy=self.aggregation_strategy,
            )
        except Exception as exc:  # pragma: no cover - depends on local HF checkpoint state
            raise RuntimeError(f"Failed to load transformers NER checkpoint from {self.checkpoint_path}: {exc}") from exc

    def predict(self, text: str) -> list[NerPrediction]:
        predictions: list[NerPrediction] = []
        for item in self._pipeline(text):
            if float(item.get("score", 1.0)) < self.min_score:
                continue
            label = normalize_transformer_label(item.get("entity_group") or item.get("entity"), self.label_map)
            if not label:
                continue
            predictions.append(
                NerPrediction(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    label=label,
                    score=float(item.get("score", 1.0)),
                )
            )
        return predictions


def normalize_transformer_label(raw_label: str | None, label_map: dict[str, str] | None = None) -> str | None:
    if not raw_label:
        return None
    cleaned = str(raw_label).strip()
    if not cleaned:
        return None
    if "-" in cleaned:
        cleaned = cleaned.split("-")[-1]
    cleaned = cleaned.replace("I-", "").replace("B-", "").replace("LABEL_", "")
    normalized_key = cleaned.upper().replace(" ", "_")
    resolved = (label_map or DEFAULT_TRANSFORMER_LABEL_MAP).get(normalized_key)
    if resolved in ENTITY_TYPES:
        return resolved
    if cleaned in ENTITY_TYPES:
        return cleaned
    return None


def load_transformer_label_map(metadata_path: Path | None) -> dict[str, str]:
    if metadata_path is None or not metadata_path.exists():
        return DEFAULT_TRANSFORMER_LABEL_MAP
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_map = data.get("label_map", {})
    label_map: dict[str, str] = {}
    for raw_label, target_label in raw_map.items():
        normalized_key = str(raw_label).strip().upper().replace(" ", "_")
        if str(target_label) in ENTITY_TYPES:
            label_map[normalized_key] = str(target_label)
            simplified_key = normalized_key
            if "-" in simplified_key:
                simplified_key = simplified_key.split("-")[-1]
            simplified_key = simplified_key.replace("I_", "").replace("B_", "").replace("LABEL_", "")
            label_map[simplified_key] = str(target_label)
    return label_map or DEFAULT_TRANSFORMER_LABEL_MAP


def resolve_transformer_metadata_path(checkpoint_path: Path, metadata_path: Path | None) -> Path | None:
    if metadata_path is not None:
        return metadata_path
    candidate = checkpoint_path / "transformers_ner_metadata.json"
    if candidate.exists():
        return candidate
    return None


def validate_ner_checkpoint(config: NerConfig) -> None:
    if config.checkpoint_path is None:
        raise RuntimeError("ner.checkpoint_path is not configured")
    if not config.checkpoint_path.exists():
        raise RuntimeError(f"NER checkpoint not found: {config.checkpoint_path}")
    provider = config.provider.lower()
    if provider == "transformers":
        required_files = ["config.json", "tokenizer_config.json"]
        missing = [name for name in required_files if not (config.checkpoint_path / name).exists()]
        if missing:
            raise RuntimeError(
                "Transformers checkpoint is incomplete. Missing files in "
                f"{config.checkpoint_path}: {', '.join(missing)}"
            )


def _regex_for_term(term: str) -> re.Pattern[str]:
    pattern = rf"\b{re.escape(term)}\b" if " " not in term else re.escape(term)
    return re.compile(pattern, re.IGNORECASE)


def _load_lexical_patterns(checkpoint_path: Path, knowledge_base: KnowledgeBase) -> list[LexicalPattern]:
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    patterns: list[LexicalPattern] = []
    for entry in data.get("patterns", []):
        score = float(entry.get("score", 0.8))
        for term in entry.get("terms", []):
            patterns.append(
                LexicalPattern(
                    label=str(entry["label"]),
                    pattern=_regex_for_term(str(term).lower()),
                    score=score,
                )
            )

    for term in knowledge_base.diagnosis_terms:
        if len(term) >= 4:
            patterns.append(LexicalPattern(label="CHẨN_ĐOÁN", pattern=_regex_for_term(term), score=0.72))
    for term in knowledge_base.drug_terms:
        if len(term) >= 5:
            patterns.append(LexicalPattern(label="THUỐC", pattern=_regex_for_term(term), score=0.7))
    return patterns


def load_ner_runtime(config: NerConfig, knowledge_base: KnowledgeBase) -> BaseNerRuntime:
    validate_ner_checkpoint(config)
    assert config.checkpoint_path is not None

    provider = config.provider.lower()
    if provider == "transformers":
        return TransformersTokenClassifierRuntime(
            checkpoint_path=config.checkpoint_path,
            metadata_path=resolve_transformer_metadata_path(config.checkpoint_path, config.metadata_path),
            min_score=config.min_score,
        )
    return LexicalNerRuntime(_load_lexical_patterns(config.checkpoint_path, knowledge_base))
