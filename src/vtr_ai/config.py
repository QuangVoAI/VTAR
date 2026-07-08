from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class KnowledgeBaseConfig:
    icd10_path: Path
    rxnorm_path: Path
    abbreviations_path: Path


@dataclass(slots=True)
class MatchingConfig:
    max_candidates: int = 3
    min_confidence: float = 0.55


@dataclass(slots=True)
class RulesConfig:
    symptom_window: int = 80
    assertion_window: int = 60


@dataclass(slots=True)
class OutputConfig:
    pretty: bool = True


@dataclass(slots=True)
class NerConfig:
    backend: str = "hybrid"
    enable_model_backend: bool = True
    checkpoint_path: Path | None = None
    metadata_path: Path | None = None
    provider: str = "lexical"
    fallback_to_rules: bool = True
    min_score: float = 0.5


@dataclass(slots=True)
class AppConfig:
    knowledge_base: KnowledgeBaseConfig
    matching: MatchingConfig
    rules: RulesConfig
    output: OutputConfig
    ner: NerConfig


def _parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            nested: dict = {}
            current[key] = nested
            stack.append((indent, nested))
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed_value = lowered == "true"
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value.strip("'\"")
        current[key] = parsed_value
    return root


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        data = _parse_simple_yaml(raw_text)

    base_dir = path.parent
    kb = data["knowledge_base"]
    matching = data.get("matching", {})
    rules = data.get("rules", {})
    output = data.get("output", {})
    return AppConfig(
        knowledge_base=KnowledgeBaseConfig(
            icd10_path=(base_dir / kb["icd10_path"]).resolve(),
            rxnorm_path=(base_dir / kb["rxnorm_path"]).resolve(),
            abbreviations_path=(base_dir / kb["abbreviations_path"]).resolve(),
        ),
        matching=MatchingConfig(
            max_candidates=int(matching.get("max_candidates", 3)),
            min_confidence=float(matching.get("min_confidence", 0.55)),
        ),
        rules=RulesConfig(
            symptom_window=int(rules.get("symptom_window", 80)),
            assertion_window=int(rules.get("assertion_window", 60)),
        ),
        output=OutputConfig(pretty=bool(output.get("pretty", True))),
        ner=NerConfig(
            backend=str(data.get("ner", {}).get("backend", "hybrid")),
            enable_model_backend=bool(
                data.get("ner", {}).get(
                    "enable_model_backend",
                    # Backward compatibility for older configs before the real model runtime landed.
                    data.get("ner", {}).get("enable_model_stub", True),
                )
            ),
            checkpoint_path=(
                (base_dir / data.get("ner", {}).get("checkpoint_path")).resolve()
                if data.get("ner", {}).get("checkpoint_path")
                else None
            ),
            metadata_path=(
                (base_dir / data.get("ner", {}).get("metadata_path")).resolve()
                if data.get("ner", {}).get("metadata_path")
                else None
            ),
            provider=str(data.get("ner", {}).get("provider", "lexical")),
            fallback_to_rules=bool(data.get("ner", {}).get("fallback_to_rules", True)),
            min_score=float(data.get("ner", {}).get("min_score", 0.5)),
        ),
    )
