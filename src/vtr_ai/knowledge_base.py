from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]


@dataclass(slots=True)
class KnowledgeRecord:
    code: str
    label: str
    aliases: list[str]

    @property
    def searchable_text(self) -> str:
        return " ".join([self.label, *self.aliases]).lower()

    @property
    def tokens(self) -> set[str]:
        return set(_tokenize(self.searchable_text))


@dataclass(slots=True)
class KnowledgeBase:
    icd10: list[KnowledgeRecord]
    rxnorm: list[KnowledgeRecord]


def _load_records(path: Path) -> list[KnowledgeRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        KnowledgeRecord(
            code=item["code"],
            label=item["label"],
            aliases=item.get("aliases", []),
        )
        for item in data
    ]


def load_knowledge_base(icd10_path: Path, rxnorm_path: Path) -> KnowledgeBase:
    return KnowledgeBase(
        icd10=_load_records(icd10_path),
        rxnorm=_load_records(rxnorm_path),
    )

