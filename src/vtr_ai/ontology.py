from __future__ import annotations

from .schemas import Entity, Fact, PatientCase, Relation


def build_patient_case(text: str, entities: list[Entity], relations: list[Relation]) -> PatientCase:
    facts: list[Fact] = []
    for entity in entities:
        facts.append(Fact(subject=entity.entity_type, predicate="mention_text", object=entity.text))
        for assertion in entity.assertions:
            facts.append(Fact(subject=entity.text, predicate="has_assertion", object=assertion))
        if entity.entity_type == "CHẨN_ĐOÁN":
            for candidate in entity.candidates:
                facts.append(Fact(subject=entity.text, predicate="mapped_to_icd10", object=candidate))
        elif entity.entity_type == "THUỐC":
            for candidate in entity.candidates:
                facts.append(Fact(subject=entity.text, predicate="mapped_to_rxnorm", object=candidate))
    for relation in relations:
        facts.append(Fact(subject=relation.source_text, predicate=relation.relation_type, object=relation.target_text))
    return PatientCase(raw_text=text, entities=entities, relations=relations, facts=facts)

