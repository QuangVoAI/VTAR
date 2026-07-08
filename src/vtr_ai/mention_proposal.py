from __future__ import annotations

import re
import functools

from .config import RulesConfig
from .knowledge_base import KnowledgeBase
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
    r"(?P<name>\b(?:WBC|RBC|HGB|HCT|PLT|NEUT%|LYPH%|AST|ALT|CRP|GLUCOSE|HbA1c|CBC|CEA|CR|creatinine|canxi(?:\s+ion\s+hóa)?|công thức máu)\b(?:\s*\([^)]*\))?)",
    re.IGNORECASE,
)
LAB_VALUE_PATTERN = re.compile(
    r"[:\-]?\s*(?P<value>\d+(?:[.,]\d+)?(?:\s?(?:%|mg/ml|mmol/l|g/dl|10\^9/l))?)",
    re.IGNORECASE,
)
DRUG_PATTERN = re.compile(
    r"\b(?P<drug>[A-Za-zÀ-ỹđĐ][A-Za-zÀ-ỹđĐ0-9./\-]*(?:[ \t]+[A-Za-zÀ-ỹđĐ0-9./\-]+){0,5}[ \t]+\d+(?:[.,-]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)(?:[ \t]+[a-z0-9./:\-]+){0,4})\b",
    re.IGNORECASE,
)
DRUG_LINE_PATTERN = re.compile(
    r"(?P<drug>\b[a-z][a-z0-9./\-]*(?:[ \t]+[a-z0-9./\-]+){0,5}(?:[ \t]+\d+(?:[.,-]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml)(?:[ \t]+[a-z0-9./:\-]+){0,4})?)",
    re.IGNORECASE,
)
BASE_DIAGNOSIS_TERMS = (
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
    "xử trí thuốc",
)
NON_DRUG_LEADING_TOKENS = {"theo", "mất", "một", "và", "các", "bệnh", "ngày"}
DIAGNOSIS_SECTION_HEADERS = (
    "các bệnh lý mãn tính",
    "bệnh lý mãn tính",
    "các bệnh đã điều trị trước đây",
    "các bệnh lý mạn tính",
    "bệnh mãn tính",
    "các phát hiện chẩn đoán khác",
    "kết quả chẩn đoán hình ảnh",
    "kết quả hình ảnh",
    "chẩn đoán sơ bộ",
)
SYMPTOM_SECTION_HEADERS = (
    "triệu chứng hiện tại",
    "triệu chứng khi nhập viện",
    "các triệu chứng hiện tại",
    "đặc điểm triệu chứng khi khám tại khoa cấp cứu",
    "thời điểm khởi phát triệu chứng",
)


def _extend_ho_span(text: str, end: int) -> int:
    tail = text[end : min(len(text), end + 20)]
    extra = re.match(r"(?:\s+đờm(?:\s+[A-Za-zÀ-ỹđĐ]+)?|\s+khan|\s+ra máu)", tail, re.IGNORECASE)
    if extra:
        return end + len(extra.group(0))
    return end


@functools.lru_cache(maxsize=16)
def _build_trie(terms: tuple[str, ...]) -> dict:
    trie: dict = {}
    for term in terms:
        node = trie
        for char in term:
            node = node.setdefault(char, {})
        node["<end>"] = term
    return trie


def _find_trie_matches(trie: dict, text: str) -> list[tuple[int, int, str]]:
    matches = []
    text_len = len(text)
    for i in range(text_len):
        node = trie
        for j in range(i, text_len):
            char = text[j]
            if char in node:
                node = node[char]
                if "<end>" in node:
                    term = node["<end>"]
                    # Unicode-aware word boundary check
                    start_ok = (i == 0 or not text[i - 1].isalnum())
                    end_ok = (j + 1 == text_len or not text[j + 1].isalnum())
                    if start_ok and end_ok:
                        matches.append((i, j + 1, term))
            else:
                break
    return matches


def _iter_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    if not text:
        return []
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


def _get_diagnosis_terms(knowledge_base: KnowledgeBase | None) -> tuple[str, ...]:
    if knowledge_base is None:
        return BASE_DIAGNOSIS_TERMS
    merged = {term.lower() for term in BASE_DIAGNOSIS_TERMS}
    merged.update(knowledge_base.diagnosis_terms)
    return tuple(sorted(merged, key=len, reverse=True))


def _source_priority(source: str) -> int:
    if source.startswith("regex:") or source.startswith("section:"):
        return 0
    if source.startswith("pattern:"):
        return 1
    if source.startswith("gazetteer:"):
        return 2
    if source.startswith("kb:"):
        return 3
    return 4


def _prune_contained_proposals(proposals: list[MentionProposal]) -> list[MentionProposal]:
    ordered = sorted(
        proposals,
        key=lambda item: (
            item.proposed_type,
            item.start,
            -(item.end - item.start),
            _source_priority(item.source),
        ),
    )
    filtered: list[MentionProposal] = []
    for proposal in ordered:
        drop = False
        for kept in filtered:
            if kept.proposed_type != proposal.proposed_type:
                continue
            if proposal.start >= kept.start and proposal.end <= kept.end:
                drop = True
                break
        if not drop:
            filtered.append(proposal)
    return sorted(filtered, key=lambda item: (item.start, item.end, item.proposed_type))


def propose_mentions(text: str, config: RulesConfig, knowledge_base: KnowledgeBase | None = None) -> list[MentionProposal]:
    proposals: list[MentionProposal] = []
    lowered = text.lower()
    diagnosis_terms = _get_diagnosis_terms(knowledge_base)

    for prefix in DIAGNOSIS_PREFIXES:
        for match in re.finditer(re.escape(prefix), lowered):
            span_start = match.end()
            span_end = min(len(text), span_start + config.symptom_window)
            snippet = text[span_start:span_end]
            stop_match = re.search(r"[\n.;]", snippet)
            end = span_start + (stop_match.start() if stop_match else len(snippet))
            candidate_text = text[span_start:end].strip().lower()
            if "không có bệnh lý khác" in candidate_text or "khỏe mạnh" in candidate_text:
                continue
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
                end = _extend_ho_span(text, end)
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
    active_symptom_section = False
    for line_start, line in _iter_lines_with_offsets(text):
        stripped = line.strip()
        lowered_line = stripped.lower()
        if not stripped:
            active_drug_section = False
            active_diag_section = False
            active_symptom_section = False
            continue
        if re.match(r"^\d+\.", lowered_line):
            active_drug_section = False
            active_diag_section = False
            active_symptom_section = False
        if any(header in lowered_line for header in DRUG_SECTION_HEADERS):
            active_drug_section = True
            active_diag_section = False
            active_symptom_section = False
            continue
        if any(header in lowered_line for header in DIAGNOSIS_SECTION_HEADERS):
            active_diag_section = True
            active_drug_section = False
            active_symptom_section = False
            continue
        if any(header in lowered_line for header in SYMPTOM_SECTION_HEADERS):
            active_symptom_section = True
            active_drug_section = False
            active_diag_section = False
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
            if "không có bệnh lý khác" in lowered_line or "khỏe mạnh" in lowered_line:
                continue
            diag_trie = _build_trie(diagnosis_terms)
            content = stripped
            content_offset = line.lower().find(content.lower())
            if content_offset == -1:
                continue
            for m_start, m_end, matched_term in _find_trie_matches(diag_trie, content.lower()):
                proposals.append(
                    MentionProposal(
                        text=line[content_offset + m_start : content_offset + m_end],
                        start=line_start + content_offset + m_start,
                        end=line_start + content_offset + m_end,
                        proposed_type="CHẨN_ĐOÁN",
                        source="section:diagnosis-term",
                    )
                )
        if active_symptom_section and lowered_line.startswith("-"):
            symptom_trie = _build_trie(SYMPTOM_TERMS)
            content = stripped.lstrip("-").strip()
            lowered_content = content.lower()
            content_offset = line.lower().find(lowered_content)
            if content_offset != -1:
                for m_start, m_end, matched_term in _find_trie_matches(symptom_trie, lowered_content):
                    start = line_start + content_offset + m_start
                    end = line_start + content_offset + m_end
                    if matched_term == "ho":
                        end = _extend_ho_span(text, end)
                    proposals.append(
                        MentionProposal(
                            text=text[start:end],
                            start=start,
                            end=end,
                            proposed_type="TRIỆU_CHỨNG",
                            source="section:symptom-term",
                        )
                    )

    if knowledge_base is not None:
        diag_filtered = tuple(t for t in knowledge_base.diagnosis_terms if len(t) >= 4)
        diag_kb_trie = _build_trie(diag_filtered)
        for m_start, m_end, matched_term in _find_trie_matches(diag_kb_trie, lowered):
            proposals.append(
                MentionProposal(
                    text=text[m_start:m_end],
                    start=m_start,
                    end=m_end,
                    proposed_type="CHẨN_ĐOÁN",
                    source="kb:diagnosis-term",
                )
            )

        drug_filtered = tuple(t for t in knowledge_base.drug_terms if len(t) >= 5 and not re.search(r"\d", t))
        drug_kb_trie = _build_trie(drug_filtered)
        for m_start, m_end, matched_term in _find_trie_matches(drug_kb_trie, lowered):
            proposals.append(
                MentionProposal(
                    text=text[m_start:m_end],
                    start=m_start,
                    end=m_end,
                    proposed_type="THUỐC",
                    source="kb:drug-term",
                )
            )

    return _prune_contained_proposals(proposals)
