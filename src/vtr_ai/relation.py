from __future__ import annotations

from .schemas import Entity, Relation


def link_relations(entities: list[Entity]) -> list[Relation]:
    relations: list[Relation] = []
    sorted_entities = sorted(entities, key=lambda item: (item.start, item.end))
    for index, entity in enumerate(sorted_entities):
        if entity.entity_type == "TÊN_XÉT_NGHIỆM":
            for other in sorted_entities[index + 1 :]:
                if other.start - entity.end > 12:
                    break
                if other.entity_type == "KẾT_QUẢ_XÉT_NGHIỆM":
                    relations.append(
                        Relation(
                            source_text=entity.text,
                            source_type=entity.entity_type,
                            relation_type="has_result",
                            target_text=other.text,
                            target_type=other.entity_type,
                        )
                    )
                    break

    for entity in sorted_entities:
        relation_type = ""
        if entity.entity_type == "TRIỆU_CHỨNG":
            relation_type = "has_symptom"
        elif entity.entity_type == "CHẨN_ĐOÁN":
            relation_type = "has_diagnosis"
        elif entity.entity_type == "THUỐC":
            relation_type = "uses_drug"
        if relation_type:
            relations.append(
                Relation(
                    source_text="PatientCase",
                    source_type="PATIENT_CASE",
                    relation_type=relation_type,
                    target_text=entity.text,
                    target_type=entity.entity_type,
                )
            )
    return relations

