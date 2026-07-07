from __future__ import annotations

from dataclasses import dataclass, field


ENTITY_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}

ASSERTION_TYPES = {"isNegated", "isFamily", "isHistorical"}


@dataclass(slots=True)
class Candidate:
    code: str
    label: str
    score: float


@dataclass(slots=True)
class MentionProposal:
    text: str
    start: int
    end: int
    proposed_type: str
    source: str


@dataclass(slots=True)
class Entity:
    text: str
    start: int
    end: int
    entity_type: str
    assertions: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "position": [self.start, self.end],
            "type": self.entity_type,
            "assertions": self.assertions,
            "candidates": self.candidates,
        }


@dataclass(slots=True)
class Document:
    text: str
    normalized_text: str
    norm_to_raw: list[int]


@dataclass(slots=True)
class Relation:
    source_text: str
    source_type: str
    relation_type: str
    target_text: str
    target_type: str


@dataclass(slots=True)
class Fact:
    subject: str
    predicate: str
    object: str


@dataclass(slots=True)
class PatientCase:
    raw_text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
