from __future__ import annotations

from dataclasses import dataclass, field
import re

from .knowledge_base import KnowledgeBase, KnowledgeRecord


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-zà-ỹđ0-9]+", " ", text.lower(), flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


@dataclass(slots=True)
class KnowledgeGraphNode:
    code: str
    label: str
    node_type: str
    aliases: list[str] = field(default_factory=list)
    neighbors: set[str] = field(default_factory=set)


@dataclass(slots=True)
class KnowledgeGraphIndex:
    diagnosis_nodes: dict[str, KnowledgeGraphNode]
    drug_nodes: dict[str, KnowledgeGraphNode]
    alias_to_diagnosis_codes: dict[str, set[str]]
    alias_to_drug_codes: dict[str, set[str]]

    @classmethod
    def from_knowledge_base(cls, knowledge_base: KnowledgeBase) -> "KnowledgeGraphIndex":
        diagnosis_nodes, alias_to_diagnosis_codes = _build_diagnosis_graph(knowledge_base.icd10)
        drug_nodes, alias_to_drug_codes = _build_drug_graph(knowledge_base.rxnorm)
        return cls(
            diagnosis_nodes=diagnosis_nodes,
            drug_nodes=drug_nodes,
            alias_to_diagnosis_codes=alias_to_diagnosis_codes,
            alias_to_drug_codes=alias_to_drug_codes,
        )


def _build_diagnosis_graph(records: list[KnowledgeRecord]) -> tuple[dict[str, KnowledgeGraphNode], dict[str, set[str]]]:
    nodes: dict[str, KnowledgeGraphNode] = {}
    alias_map: dict[str, set[str]] = {}
    prefix_groups: dict[str, list[str]] = {}
    for record in records:
        node = KnowledgeGraphNode(code=record.code, label=record.label, node_type="CHẨN_ĐOÁN", aliases=list(record.aliases))
        nodes[record.code] = node
        for alias in [record.label, *record.aliases]:
            alias_map.setdefault(_normalize(alias), set()).add(record.code)
        prefix = record.code.split(".")[0]
        prefix_groups.setdefault(prefix, []).append(record.code)
    for codes in prefix_groups.values():
        for code in codes:
            nodes[code].neighbors.update(other for other in codes if other != code)
    return nodes, alias_map


def _build_drug_graph(records: list[KnowledgeRecord]) -> tuple[dict[str, KnowledgeGraphNode], dict[str, set[str]]]:
    nodes: dict[str, KnowledgeGraphNode] = {}
    alias_map: dict[str, set[str]] = {}
    ingredient_groups: dict[str, list[str]] = {}
    for record in records:
        node = KnowledgeGraphNode(code=record.code, label=record.label, node_type="THUỐC", aliases=list(record.aliases))
        nodes[record.code] = node
        for alias in [record.label, *record.aliases]:
            normalized_alias = _normalize(alias)
            alias_map.setdefault(normalized_alias, set()).add(record.code)
            ingredient = _normalize(_strip_strength(alias))
            if ingredient:
                ingredient_groups.setdefault(ingredient, []).append(record.code)
    for codes in ingredient_groups.values():
        for code in codes:
            nodes[code].neighbors.update(other for other in codes if other != code)
    return nodes, alias_map


def _strip_strength(text: str) -> str:
    return re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)\b", "", text, flags=re.IGNORECASE).strip()
