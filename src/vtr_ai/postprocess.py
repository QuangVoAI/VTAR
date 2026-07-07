from __future__ import annotations

from .schemas import Entity


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    unique: list[Entity] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in sorted(entities, key=lambda item: (item.start, item.end, item.entity_type)):
        key = (entity.start, entity.end, entity.entity_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    filtered: list[Entity] = []
    for entity in unique:
        drop = False
        for other in unique:
            if entity is other:
                continue
            if entity.entity_type != other.entity_type:
                continue
            if entity.start >= other.start and entity.end <= other.end and len(entity.text) < len(other.text):
                if not (
                    entity.entity_type == "KẾT_QUẢ_XÉT_NGHIỆM"
                    or other.entity_type == "KẾT_QUẢ_XÉT_NGHIỆM"
                ):
                    drop = True
                    break
        if not drop:
            filtered.append(entity)
    return sorted(filtered, key=lambda item: (item.start, item.end))

