from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .fixed_span_shortlist import rank_fixed_span_shortlist, rank_fixed_span_shortlist_records
from .knowledge_base import KnowledgeBase, KnowledgeRecord, load_knowledge_base
from .schemas import ASSERTION_TYPES
from .validation import ValidationError, validate_entity_dict


QWEN_TARGET_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
SYSTEM_PROMPT = (
    "Bạn là trợ lý chuẩn hóa thực thể y khoa. "
    "Span, type, text và position đã được cố định từ bước extract trước đó. "
    "Không được sửa span hay position. "
    "Nhiệm vụ của bạn chỉ là dự đoán assertions đa nhãn và candidates mã chuẩn. "
    "Với TRIỆU_CHỨNG thì candidates luôn phải là []. "
    "Với CHẨN_ĐOÁN chỉ chọn mã ICD-10 từ shortlist nếu có. "
    "Với THUỐC chỉ chọn mã RxNorm từ shortlist nếu có. "
    "Assertions và candidates là hai quyết định độc lập. "
    "Nếu mention là CHẨN_ĐOÁN hoặc THUỐC và shortlist có mã khớp rõ ràng, vẫn phải trả mã đó kể cả khi assertion là isHistorical, isNegated hoặc isFamily. "
    "Chỉ trả candidates [] khi shortlist thật sự không có mã phù hợp với mention. "
    "Luôn trả đúng JSON schema {\"assertions\":[\"...\"],\"candidates\":[\"...\"]}."
)


@dataclass(slots=True)
class FixedSpanExample:
    example_id: str
    file_name: str
    entity_type: str
    mention: str
    position: list[int]
    context: str
    raw_text: str
    gold_assertions: list[str]
    gold_candidates: list[str]
    shortlist: list[dict]

    def to_record(self) -> dict:
        return {
            "id": self.example_id,
            "file_name": self.file_name,
            "entity_type": self.entity_type,
            "mention": self.mention,
            "position": self.position,
            "context": self.context,
            "gold_assertions": self.gold_assertions,
            "gold_candidates": self.gold_candidates,
            "shortlist": self.shortlist,
        }


def _context_window(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].strip()


def _record_label_map(records: list[KnowledgeRecord]) -> dict[str, str]:
    return {record.code: record.label for record in records}


def _rank_shortlist(
    mention: str,
    entity_type: str,
    knowledge_base: KnowledgeBase,
    shortlist_size: int,
    min_confidence: float,
) -> list[str]:
    return rank_fixed_span_shortlist(
        mention=mention,
        entity_type=entity_type,
        knowledge_base=knowledge_base,
        shortlist_size=shortlist_size,
        min_confidence=min_confidence,
    )


def _build_shortlist(
    mention: str,
    entity_type: str,
    knowledge_base: KnowledgeBase,
    shortlist_size: int,
    min_confidence: float,
    label_maps: dict[str, dict[str, str]],
    gold_candidates: list[str] | None = None,
) -> list[dict]:
    gold_candidates = gold_candidates or []
    if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
        return []
    ranked_rows = rank_fixed_span_shortlist_records(
        mention=mention,
        entity_type=entity_type,
        knowledge_base=knowledge_base,
        shortlist_size=shortlist_size,
        min_confidence=min_confidence,
    )
    ranked_codes = [row.code for row in ranked_rows]
    ranked_map = {row.code: row for row in ranked_rows}
    merged: list[str] = []
    for code in ranked_codes + gold_candidates:
        if code and code not in merged:
            merged.append(code)
    merged = merged[:shortlist_size]
    label_map = label_maps["icd10"] if entity_type == "CHẨN_ĐOÁN" else label_maps["rxnorm"]
    return [
        {
            "code": code,
            "label": label_map.get(code, "") or (ranked_map.get(code).matched_alias if code in ranked_map else ""),
            "is_gold": code in gold_candidates,
            "score": round(ranked_map.get(code).score, 4) if code in ranked_map else 0.0,
            "source": ranked_map.get(code).source if code in ranked_map else "gold",
            "matched_alias": ranked_map.get(code).matched_alias if code in ranked_map else label_map.get(code, ""),
        }
        for code in merged
    ]


def _stable_entity_id(file_name: str, start: int, end: int, entity_type: str) -> str:
    return f"{Path(file_name).stem}:{start}:{end}:{entity_type}"


def _normalize_assertions(assertions: list[str]) -> list[str]:
    values = {str(item) for item in assertions if str(item) in ASSERTION_TYPES}
    return sorted(values)


def _normalize_candidates(candidates: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in candidates:
        code = str(item).strip()
        if ":" in code:
            code = code.split(":", 1)[0].strip()
        if code and code not in deduped:
            deduped.append(code)
    return deduped[:3]


def collect_fixed_span_examples(
    input_dir: str | Path,
    gold_dir: str | Path,
    config_path: str | Path,
    context_window: int = 220,
    shortlist_size: int = 10,
    shortlist_min_confidence: float = 0.1,
) -> list[FixedSpanExample]:
    config = load_config(config_path)
    knowledge_base = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
    label_maps = {
        "icd10": _record_label_map(knowledge_base.icd10),
        "rxnorm": _record_label_map(knowledge_base.rxnorm),
    }

    examples: list[FixedSpanExample] = []
    for input_path in sorted(Path(input_dir).glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        gold_path = Path(gold_dir) / f"{input_path.stem}.json"
        if not gold_path.exists():
            raise ValidationError(f"missing gold file {gold_path.name}")
        entities = json.loads(gold_path.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            raise ValidationError(f"{gold_path.name}: root JSON must be a list")
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValidationError(f"{gold_path.name}: entity must be a dict")
            validate_entity_dict(raw_text, entity, gold_path.name)
            entity_type = str(entity["type"])
            if entity_type not in QWEN_TARGET_TYPES:
                continue
            start, end = [int(value) for value in entity["position"]]
            mention = str(entity["text"])
            gold_assertions = _normalize_assertions([str(item) for item in entity.get("assertions", [])])
            gold_candidates = (
                _normalize_candidates([str(item) for item in entity.get("candidates", [])])
                if entity_type in {"CHẨN_ĐOÁN", "THUỐC"}
                else []
            )
            shortlist = _build_shortlist(
                mention=mention,
                entity_type=entity_type,
                knowledge_base=knowledge_base,
                shortlist_size=shortlist_size,
                min_confidence=shortlist_min_confidence,
                label_maps=label_maps,
                gold_candidates=gold_candidates,
            )
            examples.append(
                FixedSpanExample(
                    example_id=_stable_entity_id(input_path.name, start, end, entity_type),
                    file_name=input_path.name,
                    entity_type=entity_type,
                    mention=mention,
                    position=[start, end],
                    context=_context_window(raw_text, start, end, context_window),
                    raw_text=raw_text,
                    gold_assertions=gold_assertions,
                    gold_candidates=gold_candidates,
                    shortlist=shortlist,
                )
            )
    return examples


def split_fixed_span_examples(
    examples: list[FixedSpanExample],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[FixedSpanExample]]:
    grouped: dict[str, list[FixedSpanExample]] = {}
    for example in examples:
        grouped.setdefault(example.file_name, []).append(example)

    keys = list(grouped)
    random.Random(seed).shuffle(keys)

    total = len(keys)
    train_cut = int(total * train_ratio)
    dev_cut = int(total * (train_ratio + dev_ratio))
    if total >= 3:
        train_cut = max(1, min(train_cut, total - 2))
        dev_cut = max(train_cut + 1, min(dev_cut, total - 1))

    split_keys = {
        "train": keys[:train_cut],
        "dev": keys[train_cut:dev_cut],
        "test": keys[dev_cut:],
    }
    splits: dict[str, list[FixedSpanExample]] = {"train": [], "dev": [], "test": []}
    for split_name, subset_keys in split_keys.items():
        rows: list[FixedSpanExample] = []
        for key in subset_keys:
            rows.extend(grouped[key])
        rows.sort(key=lambda item: (item.file_name, item.position[0], item.position[1], item.entity_type))
        splits[split_name] = rows
    return splits


def _shortlist_text(shortlist: list[dict]) -> str:
    if not shortlist:
        return "- none"
    return "\n".join(
        f"- {item['code']}: {item['label']} | matched_alias={item.get('matched_alias', item['label'])} | score={item.get('score', 0.0):.4f} | source={item.get('source', 'unknown')}"
        for item in shortlist
    )


def build_fixed_span_user_prompt(example: FixedSpanExample) -> str:
    target = "ICD-10" if example.entity_type == "CHẨN_ĐOÁN" else "RxNorm" if example.entity_type == "THUỐC" else "none"
    return (
        f"Loại thực thể: {example.entity_type}\n"
        f"Mention: {example.mention}\n"
        f"Position: [{example.position[0]}, {example.position[1]}]\n"
        f"Hệ mã đích: {target}\n"
        f"Ngữ cảnh:\n{example.context}\n\n"
        f"Shortlist được phép chọn:\n{_shortlist_text(example.shortlist)}\n\n"
        "Trả về JSON đúng schema "
        "{\"assertions\":[\"...\"],\"candidates\":[\"...\"]}. "
        "Assertions chỉ gồm isNegated, isFamily, isHistorical. "
        "Nếu không có assertion thì trả assertions rỗng. "
        "Assertions không được dùng để xóa candidate đúng. "
        "Nếu shortlist có mã khớp rõ ràng với mention, hãy giữ candidate đó ngay cả khi assertion là isHistorical, isNegated hoặc isFamily. "
        "Nếu có nhiều mã cùng phù hợp trong shortlist, có thể trả nhiều mã theo thứ tự tin cậy giảm dần, tối đa 3 mã. "
        "Nếu không có mã phù hợp hoặc loại thực thể không cần mã thì trả candidates rỗng."
    )


def build_fixed_span_assistant_response(example: FixedSpanExample) -> str:
    return json.dumps(
        {
            "assertions": example.gold_assertions,
            "candidates": example.gold_candidates,
        },
        ensure_ascii=False,
    )


def _messages_from_example(example: FixedSpanExample, include_assistant: bool) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_fixed_span_user_prompt(example)},
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": build_fixed_span_assistant_response(example)})
    return messages


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if rows else ""), encoding="utf-8")
    return path


def export_qwen_fixed_span_dataset(
    input_dir: str | Path,
    gold_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    context_window: int = 220,
    shortlist_size: int = 10,
    shortlist_min_confidence: float = 0.1,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    examples = collect_fixed_span_examples(
        input_dir=input_dir,
        gold_dir=gold_dir,
        config_path=config_path,
        context_window=context_window,
        shortlist_size=shortlist_size,
        shortlist_min_confidence=shortlist_min_confidence,
    )
    splits = split_fixed_span_examples(examples, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _write_jsonl(destination / "fixed_span_examples.jsonl", [example.to_record() for example in examples])
    for split_name, rows in splits.items():
        _write_jsonl(destination / f"{split_name}.jsonl", [example.to_record() for example in rows])
        _write_jsonl(
            destination / f"sft_{split_name}.jsonl",
            [{"messages": _messages_from_example(example, include_assistant=True), "metadata": example.to_record()} for example in rows],
        )
        _write_jsonl(
            destination / f"eval_{split_name}.jsonl",
            [
                {
                    "id": example.example_id,
                    "messages": _messages_from_example(example, include_assistant=False),
                    "expected": {
                        "assertions": example.gold_assertions,
                        "candidates": example.gold_candidates,
                    },
                    "metadata": example.to_record(),
                }
                for example in rows
            ],
        )

    summary = {
        "total_examples": len(examples),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "test_examples": len(splits["test"]),
        "entity_type_counts": {
            "TRIỆU_CHỨNG": sum(example.entity_type == "TRIỆU_CHỨNG" for example in examples),
            "CHẨN_ĐOÁN": sum(example.entity_type == "CHẨN_ĐOÁN" for example in examples),
            "THUỐC": sum(example.entity_type == "THUỐC" for example in examples),
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def parse_qwen_json_response(response_text: str) -> dict[str, list[str]]:
    match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if not match:
        return {"assertions": [], "candidates": []}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"assertions": [], "candidates": []}
    assertions = payload.get("assertions", [])
    candidates = payload.get("candidates", [])
    if not isinstance(assertions, list):
        assertions = []
    if not isinstance(candidates, list):
        candidates = []
    return {
        "assertions": _normalize_assertions([str(item) for item in assertions]),
        "candidates": _normalize_candidates([str(item) for item in candidates]),
    }


def build_runtime_example(
    raw_text: str,
    file_name: str,
    entity: dict,
    config_path: str | Path,
    context_window: int = 220,
    shortlist_size: int = 10,
    shortlist_min_confidence: float = 0.1,
    gold_candidates: list[str] | None = None,
) -> FixedSpanExample:
    config = load_config(config_path)
    knowledge_base = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
    label_maps = {
        "icd10": _record_label_map(knowledge_base.icd10),
        "rxnorm": _record_label_map(knowledge_base.rxnorm),
    }
    entity_type = str(entity["type"])
    start, end = [int(value) for value in entity["position"]]
    mention = str(entity["text"])
    shortlist = _build_shortlist(
        mention=mention,
        entity_type=entity_type,
        knowledge_base=knowledge_base,
        shortlist_size=shortlist_size,
        min_confidence=shortlist_min_confidence,
        label_maps=label_maps,
        gold_candidates=gold_candidates,
    )
    return FixedSpanExample(
        example_id=_stable_entity_id(file_name, start, end, entity_type),
        file_name=file_name,
        entity_type=entity_type,
        mention=mention,
        position=[start, end],
        context=_context_window(raw_text, start, end, context_window),
        raw_text=raw_text,
        gold_assertions=[],
        gold_candidates=[],
        shortlist=shortlist,
    )


def apply_prediction_to_entity(entity: dict, prediction: dict[str, list[str]]) -> dict:
    updated = dict(entity)
    entity_type = str(updated["type"])
    if entity_type in QWEN_TARGET_TYPES:
        updated["assertions"] = _normalize_assertions(prediction.get("assertions", []))
    else:
        updated["assertions"] = [str(item) for item in updated.get("assertions", [])]
    if entity_type in {"CHẨN_ĐOÁN", "THUỐC"}:
        updated["candidates"] = _normalize_candidates(prediction.get("candidates", []))
    else:
        updated["candidates"] = []
    return updated


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
