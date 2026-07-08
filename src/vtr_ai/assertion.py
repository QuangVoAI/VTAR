from __future__ import annotations

import re

from .section_parser import find_enclosing_clause
from .schemas import Document, Entity


NEGATION_CUES = (
    "không",
    "chưa",
    "phủ nhận",
    "âm tính với",
    "không có",
    "loại trừ",
)
DOUBLE_NEGATION_PATTERNS = (
    "không loại trừ",
    "chưa loại trừ",
    "không thể loại trừ",
    "không hoàn toàn loại trừ",
)
NON_SCOPING_NEGATION_PATTERNS = (
    "không rõ",
    "không đặc hiệu",
    "không xác định",
    "không dung nạp",
)
FAMILY_CUES = ("bố", "mẹ", "cha", "anh", "chị", "em", "người nhà", "gia đình")
HISTORY_CUES = (
    "tiền sử",
    "đã từng",
    "trước đây",
    "dùng lâu dài",
    "thuốc trước khi nhập viện",
    "trước nhập viện",
    "đã điều trị trước đây",
)
HARD_SEPARATORS = (".", ";", "\n")
CONTRAST_CUES = (" nhưng ", " tuy nhiên ", " song ", " dù vậy ")


def _find_clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    clause_start = 0
    clause_end = len(text)

    for separator in HARD_SEPARATORS:
        position = text.rfind(separator, 0, start)
        if position != -1:
            clause_start = max(clause_start, position + 1)
    for separator in HARD_SEPARATORS:
        position = text.find(separator, end)
        if position != -1:
            clause_end = min(clause_end, position)

    for cue in CONTRAST_CUES:
        position = text.rfind(cue, clause_start, start)
        if position != -1:
            clause_start = max(clause_start, position + len(cue))
        position = text.find(cue, end, clause_end)
        if position != -1:
            clause_end = min(clause_end, position)

    return clause_start, clause_end


def _contains_cue(context: str, cues: tuple[str, ...]) -> bool:
    for cue in cues:
        if " " in cue:
            if cue in context:
                return True
            continue
        if re.search(rf"\b{re.escape(cue)}\b", context):
            return True
    return False


def _has_negation(prefix: str) -> bool:
    if not prefix.strip():
        return False
    for pattern in NON_SCOPING_NEGATION_PATTERNS:
        prefix = prefix.replace(pattern, " ")
    if any(pattern in prefix for pattern in DOUBLE_NEGATION_PATTERNS):
        for pattern in DOUBLE_NEGATION_PATTERNS:
            prefix = prefix.replace(pattern, " ")
    return _contains_cue(prefix, NEGATION_CUES)


def infer_assertions(text_or_document: str | Document, entity: Entity, window: int) -> list[str]:
    if entity.entity_type not in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}:
        return []

    if isinstance(text_or_document, Document):
        document = text_or_document
        text = document.text
    else:
        document = None
        text = text_or_document

    left = max(0, entity.start - window)
    right = min(len(text), entity.end + window)
    local_text = text[left:right].lower()
    relative_start = entity.start - left
    relative_end = entity.end - left

    clause_text = ""
    clause_prefix = ""
    default_assertions: list[str] = []
    if document is not None:
        clause = find_enclosing_clause(document.clauses, entity.start, entity.end)
        if clause is not None:
            clause_relative_start = max(0, entity.start - clause.start)
            clause_relative_end = max(clause_relative_start, entity.end - clause.start)
            lowered_clause = clause.text.lower()
            local_start, local_end = _find_clause_bounds(lowered_clause, clause_relative_start, clause_relative_end)
            clause_text = lowered_clause[local_start:local_end]
            clause_prefix = lowered_clause[local_start:clause_relative_start].strip()
            default_assertions = list(clause.default_assertions)
    if not clause_text:
        clause_start, clause_end = _find_clause_bounds(local_text, relative_start, relative_end)
        clause_text = local_text[clause_start:clause_end]
        clause_prefix = local_text[clause_start:relative_start].strip()

    document_prefix = text[: entity.start].lower()

    assertions: list[str] = list(default_assertions)
    if _has_negation(clause_prefix):
        assertions.append("isNegated")
    if _contains_cue(clause_prefix, FAMILY_CUES):
        assertions.append("isFamily")
    if _contains_cue(clause_prefix, HISTORY_CUES) or (
        entity.entity_type == "THUỐC"
        and any(cue in document_prefix for cue in ("thuốc trước khi nhập viện", "trước khi nhập viện"))
    ):
        assertions.append("isHistorical")

    if not assertions and clause_text.startswith("gia đình") and _contains_cue(clause_text, FAMILY_CUES):
        assertions.append("isFamily")

    return sorted(set(assertions))
