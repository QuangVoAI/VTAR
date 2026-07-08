from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

from .ner_labels import ENTITY_LABELS, ID_TO_LABEL, LABEL_TO_ID


@dataclass(slots=True)
class AnnotatedEntity:
    start: int
    end: int
    label: str


@dataclass(slots=True)
class AnnotatedExample:
    text: str
    entities: list[AnnotatedEntity]


def load_annotated_examples(path: str | Path) -> list[AnnotatedExample]:
    examples: list[AnnotatedExample] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        entities = [
            AnnotatedEntity(
                start=int(entity["position"][0]),
                end=int(entity["position"][1]),
                label=str(entity["type"]),
            )
            for entity in item.get("entities", [])
        ]
        examples.append(AnnotatedExample(text=str(item["text"]), entities=entities))
    return examples


def validate_annotated_example(example: AnnotatedExample) -> None:
    for entity in example.entities:
        if entity.label not in ENTITY_LABELS:
            raise ValueError(f"Unsupported label: {entity.label}")
        if entity.start < 0 or entity.end <= entity.start or entity.end > len(example.text):
            raise ValueError(f"Invalid span: {(entity.start, entity.end)}")


def align_entities_to_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    entities: list[AnnotatedEntity],
) -> list[str]:
    labels = ["O"] * len(offsets)
    ordered_entities = sorted(entities, key=lambda item: (item.start, item.end))
    for entity in ordered_entities:
        covered_token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and not (token_end <= entity.start or token_start >= entity.end)
        ]
        if not covered_token_indices:
            continue
        labels[covered_token_indices[0]] = f"B-{entity.label}"
        for token_index in covered_token_indices[1:]:
            labels[token_index] = f"I-{entity.label}"
    return labels


def export_label_metadata(output_dir: str | Path) -> Path:
    target = Path(output_dir) / "transformers_ner_metadata.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "label_map": ID_TO_LABEL,
        "id_to_label": ID_TO_LABEL,
        "label_to_id": LABEL_TO_ID,
    }
    target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def split_annotated_examples(
    examples: list[AnnotatedExample],
    validation_ratio: float,
    seed: int = 42,
) -> tuple[list[AnnotatedExample], list[AnnotatedExample]]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0.0, 1.0)")
    if validation_ratio == 0.0 or len(examples) < 2:
        return examples, []
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(round(len(shuffled) * validation_ratio)))
    validation_size = min(validation_size, len(shuffled) - 1)
    eval_examples = shuffled[:validation_size]
    train_examples = shuffled[validation_size:]
    return train_examples, eval_examples
