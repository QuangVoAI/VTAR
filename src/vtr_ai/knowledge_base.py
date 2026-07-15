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
    tty: str | None = None
    ingredient_codes: list[str] | None = None

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
    diagnosis_token_index: dict[str, list[KnowledgeRecord]] | None = None
    drug_token_index: dict[str, list[KnowledgeRecord]] | None = None
    diagnosis_alias_index: dict[str, list[KnowledgeRecord]] | None = None
    drug_alias_index: dict[str, list[KnowledgeRecord]] | None = None

    @property
    def diagnosis_terms(self) -> list[str]:
        terms = {record.label.lower() for record in self.icd10}
        for record in self.icd10:
            terms.update(alias.lower() for alias in record.aliases)
        return sorted(term for term in terms if term)

    @property
    def drug_terms(self) -> list[str]:
        terms = {record.label.lower() for record in self.rxnorm}
        for record in self.rxnorm:
            terms.update(alias.lower() for alias in record.aliases)
        return sorted(term for term in terms if term)


def _load_records(path: Path) -> list[KnowledgeRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        KnowledgeRecord(
            code=item["code"],
            label=item["label"],
            aliases=item.get("aliases", []),
            tty=item.get("tty"),
            ingredient_codes=item.get("ingredient_codes"),
        )
        for item in data
    ]


def load_knowledge_base(icd10_path: Path, rxnorm_path: Path) -> KnowledgeBase:
    icd10 = _load_records(icd10_path)
    rxnorm = _load_records(rxnorm_path)
    return KnowledgeBase(
        icd10=icd10,
        rxnorm=rxnorm,
        diagnosis_token_index=_build_token_index(icd10),
        drug_token_index=_build_token_index(rxnorm),
        diagnosis_alias_index=_build_alias_index(icd10),
        drug_alias_index=_build_alias_index(rxnorm),
    )


def _build_token_index(records: list[KnowledgeRecord]) -> dict[str, list[KnowledgeRecord]]:
    index: dict[str, list[KnowledgeRecord]] = {}
    for record in records:
        for token in record.tokens:
            if len(token) < 2:
                continue
            index.setdefault(token, []).append(record)
    return index


def _build_alias_index(records: list[KnowledgeRecord]) -> dict[str, list[KnowledgeRecord]]:
    index: dict[str, list[KnowledgeRecord]] = {}
    for record in records:
        for alias in [record.label, *record.aliases]:
            normalized = " ".join(_tokenize(alias))
            if not normalized:
                continue
            index.setdefault(normalized, []).append(record)
    return index
