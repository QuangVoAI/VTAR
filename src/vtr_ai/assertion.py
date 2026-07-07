from __future__ import annotations

from .schemas import Entity


import re


NEGATION_CUES = ("không", "chưa", "phủ nhận", "âm tính với", "không có")
FAMILY_CUES = ("bố", "mẹ", "cha", "anh", "chị", "em", "người nhà", "gia đình")
HISTORY_CUES = (
    "tiền sử",
    "đã từng",
    "trước đây",
    "mạn tính",
    "dùng lâu dài",
    "thuốc trước khi nhập viện",
    "trước nhập viện",
    "đã điều trị trước đây",
)
CLAUSE_SEPARATORS = (".", ";", ",", "\n", " nhưng ", " tuy nhiên ")


def _contains_cue(context: str, cues: tuple[str, ...]) -> bool:
    for cue in cues:
        if " " in cue:
            if cue in context:
                return True
            continue
        if re.search(rf"\b{re.escape(cue)}\b", context):
            return True
    return False


def infer_assertions(text: str, entity: Entity, window: int) -> list[str]:
    if entity.entity_type not in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}:
        return []
    left = max(0, entity.start - window)
    right = min(len(text), entity.end + window)
    context = text[left:right].lower()
    relative_start = entity.start - left
    clause_start = max(context.rfind(separator, 0, relative_start) for separator in CLAUSE_SEPARATORS)
    clause_end_candidates = [context.find(separator, relative_start) for separator in CLAUSE_SEPARATORS if context.find(separator, relative_start) != -1]
    clause_end = min(clause_end_candidates) if clause_end_candidates else len(context)
    if clause_start == -1:
        clause_start = 0
    else:
        clause_start += 1
    local_context = context[clause_start:clause_end]
    document_prefix = text[: entity.start].lower()
    assertions: list[str] = []
    if _contains_cue(local_context, NEGATION_CUES):
        assertions.append("isNegated")
    if _contains_cue(local_context, FAMILY_CUES):
        assertions.append("isFamily")
    if _contains_cue(local_context, HISTORY_CUES) or (
        entity.entity_type == "THUỐC"
        and any(cue in document_prefix for cue in ("thuốc trước khi nhập viện", "trước khi nhập viện"))
    ):
        assertions.append("isHistorical")
    return assertions
