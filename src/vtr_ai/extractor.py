from __future__ import annotations

import re

from .schemas import Entity, MentionProposal


NON_DRUG_PHRASES = (
    "nước tiểu",
    "ống thông",
    "tiểu cầu",
    "bạch cầu",
    "ure",
    "creatinine",
    "giảm lượng",
    "xuống còn",
    "lên 6.3",
    "sau đó giảm xuống",
    "sau đó 30 mg",
    "total of",
)
NON_DRUG_EXACTS = {"bid", "daily", "once", "tid", "qid", "dose"}
NON_DIAGNOSIS_PREFIXES = ("với ",)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1] in " .,;:-":
        end -= 1
    return start, end


def _is_token_internal(text: str, start: int, end: int) -> bool:
    return (start > 0 and text[start - 1].isalnum()) or (end < len(text) and text[end : end + 1].isalnum())


def _clean_drug_span(text: str, start: int, end: int) -> tuple[int, int]:
    raw_candidate = text[start:end].splitlines()[0]
    candidate = raw_candidate
    for stop_token in (" dose ", " reduced from ", " decreased from "):
        lowered_candidate = candidate.lower()
        pos = lowered_candidate.find(stop_token)
        if pos != -1:
            candidate = candidate[:pos]
            break
    for stop_token in (" và ", ", cùng ", " cùng ", ";"):
        lowered_candidate = candidate.lower()
        pos = lowered_candidate.find(stop_token)
        if pos != -1:
            candidate = candidate[:pos]
            break
    for cue in (" dùng ", " sử dụng ", " điều trị ", " chỉ định "):
        lowered = candidate.lower()
        pos = lowered.rfind(cue)
        if pos != -1:
            candidate = candidate[pos + len(cue) :]
            break
    candidate = re.sub(
        r"^(?:(?:thuốc trước khi nhập viện(?: lần này)?|trước khi nhập viện|bệnh nhân có tiền sử sử dụng|có tiền sử sử dụng|tiền sử sử dụng|bắt đầu dùng|được chỉ định điều trị|được cho dùng|được cho|cho po|điều trị|sử dụng|dùng|nhận|tự điều trị bằng|điều trị bằng|bằng|liều cao)\s*[:\-]?\s*)+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^(?:thêm)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:(?:iv|po|im)\s+)+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(?:x\s*\d+|po|iv|im|bid|tid|qid|daily|once|nebs?|nebulizer)\b.*$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:[-*]\s*)+", "", candidate)
    leading_trim = raw_candidate.lower().find(candidate.lower()) if candidate else 0
    if leading_trim < 0:
        leading_trim = 0
    return start + leading_trim, start + leading_trim + len(candidate)


def _clean_diagnosis_span(text: str, start: int, end: int) -> tuple[int, int]:
    candidate = text[start:end]
    candidate = re.sub(r"^(?:[-*]\s*)+", "", candidate)
    lowered = candidate.lower()
    for prefix in NON_DIAGNOSIS_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    leading_trim = len(text[start:end]) - len(candidate)
    return start + leading_trim, start + leading_trim + len(candidate)


def extract_entities_from_mentions(text: str, mentions: list[MentionProposal]) -> list[Entity]:
    entities: list[Entity] = []
    for mention in mentions:
        start, end = _trim_span(text, mention.start, mention.end)
        if mention.proposed_type == "CHẨN_ĐOÁN":
            start, end = _clean_diagnosis_span(text, start, end)
            start, end = _trim_span(text, start, end)
            if _is_token_internal(text, start, end):
                continue
        if mention.proposed_type == "THUỐC":
            start, end = _clean_drug_span(text, start, end)
            start, end = _trim_span(text, start, end)
            candidate_text = text[start:end].strip().lower()
            if _is_token_internal(text, start, end):
                continue
            if (
                not candidate_text
                or not re.search(r"[a-zà-ỹđ]", candidate_text, re.IGNORECASE)
                or candidate_text in NON_DRUG_EXACTS
                or candidate_text.startswith(("x ", "iv ", "po ", "im "))
                or any(marker in candidate_text for marker in ("decreased from", "reduced from"))
                or any(phrase in candidate_text for phrase in NON_DRUG_PHRASES)
                or re.match(r"^\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)$", candidate_text)
                or re.match(r"^\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)?\s*(?:bid|daily|once|tid|qid)$", candidate_text)
            ):
                continue
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
