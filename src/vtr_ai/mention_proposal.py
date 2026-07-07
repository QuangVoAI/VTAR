from __future__ import annotations

import re

from .config import RulesConfig
from .schemas import MentionProposal


DIAGNOSIS_PREFIXES = (
    "được chẩn đoán",
    "chẩn đoán sơ bộ",
    "mắc bệnh",
    "bệnh lý mãn tính:",
)

SYMPTOM_TERMS = (
    "ho",
    "sốt",
    "đau bụng",
    "đau thượng vị",
    "ợ hơi",
    "tức ngực",
    "khó thở",
    "mệt mỏi",
    "buồn nôn",
    "nôn",
    "đờm xanh",
    "đờm",
    "đánh trống ngực",
    "tiêu chảy",
    "lo âu",
    "mất ngủ",
    "đau nhức",
    "táo bón",
    "ngất xỉu",
    "căng thẳng",
)

LAB_NAME_PATTERN = re.compile(
    r"(?P<name>\b(?:WBC|RBC|HGB|HCT|PLT|NEUT%|LYPH%|AST|ALT|CRP|GLUCOSE|HbA1c)\b(?:\s*\([^)]*\))?)",
    re.IGNORECASE,
)
LAB_VALUE_PATTERN = re.compile(
    r"[:\-]?\s*(?P<value>\d+(?:[.,]\d+)?(?:\s?(?:%|mg/ml|mmol/l|g/dl|10\^9/l))?)",
    re.IGNORECASE,
)
DRUG_PATTERN = re.compile(
    r"\b(?P<drug>[A-Za-zÀ-ỹđĐ][A-Za-zÀ-ỹđĐ0-9/\-]*(?:\s+[A-Za-zÀ-ỹđĐ0-9/\-]+){0,5}\s+\d+(?:[.,-]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)(?:\s+[a-z0-9:]+){0,4})\b",
    re.IGNORECASE,
)
DRUG_LINE_PATTERN = re.compile(
    r"(?P<drug>\b[a-z][a-z0-9/\-]*(?:\s+[a-z0-9/\-]+){0,4}(?:\s+\d+(?:[.,-]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)(?:\s+[a-z0-9:]+){0,4})?)",
    re.IGNORECASE,
)
DIAGNOSIS_TERMS = (
    "xơ gan",
    "hội chứng não gan",
    "tăng huyết áp",
    "phình động mạch chủ",
    "bệnh tim mạch do xơ vữa động mạch",
    "rung nhĩ",
    "nhồi máu cơ tim",
    "khối u trực tràng",
    "u ác trực tràng",
    "u tuyến",
    "bệnh động mạch vành",
    "tăng calci máu",
    "viêm mô tế bào",
    "ngoại tâm thu nhĩ",
    "ngoại tâm thu thất",
    "tim to",
)
DRUG_SECTION_HEADERS = (
    "thuốc trước khi nhập viện",
    "thuốc trước khi nhập viện lần này",
    "danh sách thuốc trước nhập viện",
)
NON_DRUG_LEADING_TOKENS = {"theo", "mất", "một", "và", "các", "bệnh", "ngày"}
DIAGNOSIS_SECTION_HEADERS = (
    "các bệnh lý mãn tính",
    "bệnh lý mãn tính",
    "các phát hiện chẩn đoán khác",
    "kết quả chẩn đoán hình ảnh",
    "kết quả hình ảnh",
)


def _iter_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    offset = 0
    for chunk in text.splitlines(keepends=True):
        line = chunk.rstrip("\n")
        lines.append((offset, line))
        offset += len(chunk)
    if not text.endswith("\n") and (not lines or lines[-1][1] != text.splitlines()[-1]):
        last = text.splitlines()[-1]
        lines.append((len(text) - len(last), last))
    return lines


def propose_mentions(text: str, config: RulesConfig) -> list[MentionProposal]:
    proposals: list[MentionProposal] = []
    lowered = text.lower()

    for prefix in DIAGNOSIS_PREFIXES:
        for match in re.finditer(re.escape(prefix), lowered):
            span_start = match.end()
            span_end = min(len(text), span_start + config.symptom_window)
            snippet = text[span_start:span_end]
            stop_match = re.search(r"[\n.;]", snippet)
            end = span_start + (stop_match.start() if stop_match else len(snippet))
            proposals.append(
                MentionProposal(
                    text=text[span_start:end],
                    start=span_start,
                    end=end,
                    proposed_type="CHẨN_ĐOÁN",
                    source="pattern:diagnosis-context",
                )
            )

    for term in SYMPTOM_TERMS:
        pattern = rf"\b{re.escape(term)}\b" if " " not in term else re.escape(term)
        for match in re.finditer(pattern, lowered):
            start = match.start()
            end = match.end()
            if term == "ho":
                tail = text[end : min(len(text), end + 20)]
                extra = re.match(r"(?:\s+(?:đờm|khan|ra máu))(?:\s+[A-Za-zÀ-ỹđĐ]+)?", tail, re.IGNORECASE)
                if extra:
                    end += len(extra.group(0))
            proposals.append(
                MentionProposal(
                    text=text[start:end],
                    start=start,
                    end=end,
                    proposed_type="TRIỆU_CHỨNG",
                    source="gazetteer:symptom",
                )
            )

    for match in LAB_NAME_PATTERN.finditer(text):
        proposals.append(
            MentionProposal(
                text=match.group("name"),
                start=match.start("name"),
                end=match.end("name"),
                proposed_type="TÊN_XÉT_NGHIỆM",
                source="regex:lab-name",
            )
        )
        trailing = text[match.end() : min(len(text), match.end() + 20)]
        value_match = LAB_VALUE_PATTERN.search(trailing)
        if value_match:
            start = match.end() + value_match.start("value")
            end = match.end() + value_match.end("value")
            proposals.append(
                MentionProposal(
                    text=text[start:end],
                    start=start,
                    end=end,
                    proposed_type="KẾT_QUẢ_XÉT_NGHIỆM",
                    source="regex:lab-value",
                )
            )

    for match in DRUG_PATTERN.finditer(text):
        drug_text = match.group("drug")
        proposals.append(
            MentionProposal(
                text=drug_text,
                start=match.start("drug"),
                end=match.end("drug"),
                proposed_type="THUỐC",
                source="regex:drug-regimen",
            )
        )

    active_drug_section = False
    active_diag_section = False
    for line_start, line in _iter_lines_with_offsets(text):
        stripped = line.strip()
        lowered_line = stripped.lower()
        if not stripped:
            active_drug_section = False
            active_diag_section = False
            continue
        if re.match(r"^\d+\.", lowered_line):
            active_drug_section = False
            active_diag_section = False
        if any(header in lowered_line for header in DRUG_SECTION_HEADERS):
            active_drug_section = True
            active_diag_section = False
            continue
        if any(header in lowered_line for header in DIAGNOSIS_SECTION_HEADERS):
            active_diag_section = True
            active_drug_section = False
            continue
        if active_drug_section and lowered_line.startswith("-"):
            content = stripped.lstrip("-").strip()
            match = DRUG_LINE_PATTERN.search(content)
            if match:
                drug_text = match.group("drug").strip()
                first_token = drug_text.split()[0].lower() if drug_text.split() else ""
                if first_token in NON_DRUG_LEADING_TOKENS or not re.match(r"^[a-z][a-z0-9/\-]*$", first_token):
                    continue
                stop_tokens = (" cho ", " điều trị ", " (", ",")
                cut = len(drug_text)
                lowered_drug = drug_text.lower()
                for token in stop_tokens:
                    pos = lowered_drug.find(token)
                    if pos != -1:
                        cut = min(cut, pos)
                drug_text = drug_text[:cut].strip()
                if len(drug_text) < 4:
                    continue
                if not re.search(r"\d", drug_text) and len(drug_text.split()[0]) < 5:
                    continue
                start = line.find(drug_text)
                if drug_text and start != -1:
                    proposals.append(
                        MentionProposal(
                            text=drug_text,
                            start=line_start + start,
                            end=line_start + start + len(drug_text),
                            proposed_type="THUỐC",
                            source="section:drug-list",
                        )
                    )
        if active_diag_section or lowered_line.startswith("-"):
            for term in DIAGNOSIS_TERMS:
                pattern = rf"\b{re.escape(term)}\b" if " " not in term else re.escape(term)
                for match in re.finditer(pattern, lowered_line):
                    raw_start = line.lower().find(match.group(0), 0)
                    if raw_start == -1:
                        continue
                    proposals.append(
                        MentionProposal(
                            text=line[raw_start : raw_start + len(match.group(0))],
                            start=line_start + raw_start,
                            end=line_start + raw_start + len(match.group(0)),
                            proposed_type="CHẨN_ĐOÁN",
                            source="section:diagnosis-term",
                        )
                    )

    return proposals
