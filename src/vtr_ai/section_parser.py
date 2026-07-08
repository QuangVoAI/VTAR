from __future__ import annotations

import re

from .schemas import AnchorNode, ClauseContext, SectionContext


SECTION_RULES = (
    ("drug_history", ("thuốc trước khi nhập viện", "thuốc trước nhập viện", "danh sách thuốc trước nhập viện", "xử trí thuốc"), ["isHistorical"], ["THUỐC"]),
    ("diagnosis_history", ("bệnh lý mãn tính", "các bệnh lý mãn tính", "bệnh mãn tính", "các bệnh lý mạn tính", "tiền sử bệnh", "tiền sử bệnh nội khoa"), ["isHistorical"], ["CHẨN_ĐOÁN"]),
    ("symptom_current", ("triệu chứng hiện tại", "triệu chứng khi nhập viện", "lý do nhập viện", "bệnh sử", "tiền sử bệnh hiện tại"), [], ["TRIỆU_CHỨNG"]),
    ("lab_results", ("kết quả xét nghiệm", "xét nghiệm", "cận lâm sàng", "xét nghiệm máu", "công thức máu"), [], ["TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]),
    ("diagnosis_current", ("chẩn đoán sơ bộ", "chẩn đoán", "đánh giá", "kết quả chẩn đoán hình ảnh", "kết quả hình ảnh"), [], ["CHẨN_ĐOÁN"]),
)
INLINE_ANCHOR_LIMIT = 64
INLINE_ANCHOR_TOKENS = 8
CLAUSE_SEPARATORS = ".;\n"


def _iter_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    offset = 0
    for chunk in text.splitlines(keepends=True):
        line = chunk.rstrip("\n")
        lines.append((offset, line))
        offset += len(chunk)
    if not text.endswith("\n") and text and (not lines or lines[-1][1] != text.splitlines()[-1]):
        last = text.splitlines()[-1]
        lines.append((len(text) - len(last), last))
    return lines


def _normalize_header(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"^[-*]+\s*", "", cleaned)
    cleaned = cleaned.replace(":", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _classify_header(header: str) -> tuple[str, list[str], list[str]]:
    normalized = _normalize_header(header)
    for kind, aliases, default_assertions, expected_types in SECTION_RULES:
        if any(alias in normalized for alias in aliases):
            return kind, list(default_assertions), list(expected_types)
    return "general", [], []


def extract_structure(text: str) -> tuple[list[SectionContext], list[ClauseContext], list[AnchorNode]]:
    explicit_sections: list[SectionContext] = []
    anchors: list[AnchorNode] = []

    for line_start, line in _iter_lines_with_offsets(text):
        stripped = line.strip()
        if not stripped:
            continue
        header_candidate = stripped.split(":", 1)[0] if ":" in stripped else stripped
        kind, default_assertions, expected_types = _classify_header(header_candidate)
        if kind == "general":
            continue
        start = line_start
        end = line_start + len(line)
        explicit_sections.append(
            SectionContext(
                start=start,
                end=end,
                header=header_candidate.strip(),
                kind=kind,
                default_assertions=default_assertions,
                expected_types=expected_types,
            )
        )
        raw_header_start = line.lower().find(header_candidate.strip().lower())
        if raw_header_start != -1:
            anchor_start = line_start + raw_header_start
            anchors.append(
                AnchorNode(
                    start=anchor_start,
                    end=anchor_start + len(header_candidate.strip()),
                    text=text[anchor_start : anchor_start + len(header_candidate.strip())],
                    kind=kind,
                    expected_types=expected_types,
                )
            )

    sections: list[SectionContext] = []
    if not explicit_sections:
        sections.append(SectionContext(start=0, end=len(text), header="", kind="general"))
    else:
        explicit_sections.sort(key=lambda item: (item.start, item.end))
        for index, section in enumerate(explicit_sections):
            next_start = explicit_sections[index + 1].start if index + 1 < len(explicit_sections) else len(text)
            sections.append(
                SectionContext(
                    start=section.start,
                    end=next_start,
                    header=section.header,
                    kind=section.kind,
                    default_assertions=list(section.default_assertions),
                    expected_types=list(section.expected_types),
                )
            )
        if sections[0].start > 0:
            sections.insert(0, SectionContext(start=0, end=sections[0].start, header="", kind="general"))

    clauses: list[ClauseContext] = []
    for section in sections:
        current_start = section.start
        for index in range(section.start, section.end):
            if text[index] not in CLAUSE_SEPARATORS:
                continue
            _append_clause(text, current_start, index, section, clauses, anchors)
            current_start = index + 1
        _append_clause(text, current_start, section.end, section, clauses, anchors)

    return sections, clauses, anchors


def _append_clause(
    text: str,
    start: int,
    end: int,
    section: SectionContext,
    clauses: list[ClauseContext],
    anchors: list[AnchorNode],
) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return

    clause_text = text[start:end]
    anchor_text = section.header
    anchor_kind = section.kind
    expected_types = list(section.expected_types)
    default_assertions = list(section.default_assertions)

    inline_anchor_match = re.match(r"^\s*([^\n:]{1,%d}):" % INLINE_ANCHOR_LIMIT, clause_text)
    if inline_anchor_match:
        candidate = inline_anchor_match.group(1).strip(" -*")
        if candidate and len(candidate.split()) <= INLINE_ANCHOR_TOKENS:
            inferred_kind, inferred_assertions, inferred_expected_types = _classify_header(candidate)
            if inferred_kind != "general":
                anchor_text = candidate
                anchor_kind = inferred_kind
                expected_types = inferred_expected_types or expected_types
                default_assertions = sorted(set(default_assertions + inferred_assertions))
                anchor_start = start + inline_anchor_match.start(1) + (len(inline_anchor_match.group(1)) - len(candidate))
                anchors.append(
                    AnchorNode(
                        start=anchor_start,
                        end=anchor_start + len(candidate),
                        text=text[anchor_start : anchor_start + len(candidate)],
                        kind=anchor_kind,
                        expected_types=list(expected_types),
                    )
                )

    clauses.append(
        ClauseContext(
            start=start,
            end=end,
            text=text[start:end],
            section_kind=section.kind,
            default_assertions=default_assertions,
            expected_types=expected_types,
            anchor_text=anchor_text,
            anchor_kind=anchor_kind,
        )
    )


def find_enclosing_clause(clauses: list[ClauseContext], start: int, end: int) -> ClauseContext | None:
    for clause in clauses:
        if clause.start <= start and end <= clause.end:
            return clause
    for clause in clauses:
        if not (end <= clause.start or start >= clause.end):
            return clause
    return None


def find_enclosing_section(sections: list[SectionContext], start: int, end: int) -> SectionContext | None:
    for section in sections:
        if section.start <= start and end <= section.end:
            return section
    return None
