from __future__ import annotations

import re

from .schemas import Entity, MentionProposal


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1] in " .,;:-":
        end -= 1
    return start, end


def _clean_drug_span(text: str, start: int, end: int) -> tuple[int, int]:
    candidate = text[start:end]
    candidate = re.sub(
        r"^(?:thuốc trước khi nhập viện(?: lần này)?|trước khi nhập viện|bệnh nhân có tiền sử sử dụng|có tiền sử sử dụng|tiền sử sử dụng|bắt đầu dùng|được chỉ định điều trị|điều trị|sử dụng|dùng)\s*[:\-]?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^(?:[-*]\s*)+", "", candidate)
    leading_trim = len(text[start:end]) - len(candidate)
    return start + leading_trim, start + leading_trim + len(candidate)


def extract_entities_from_mentions(text: str, mentions: list[MentionProposal]) -> list[Entity]:
    entities: list[Entity] = []
    for mention in mentions:
        start, end = _trim_span(text, mention.start, mention.end)
        if mention.proposed_type == "THUỐC":
            start, end = _clean_drug_span(text, start, end)
            start, end = _trim_span(text, start, end)
        if start >= end:
            continue
        entities.append(
            Entity(
                text=text[start:end],
                start=start,
                end=end,
                entity_type=mention.proposed_type,
            )
        )
    return entities
