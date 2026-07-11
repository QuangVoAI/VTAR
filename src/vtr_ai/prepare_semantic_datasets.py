from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .candidate_generation import rank_candidates_indexed
from .config import MatchingConfig, load_config
from .knowledge_base import KnowledgeBase, KnowledgeRecord, load_knowledge_base
from .validation import validate_output_directory


SYSTEM_PROMPT = (
    "Bạn là trợ lý semantic normalization cho bài Viettel. "
    "Nhiệm vụ là chọn mã ICD-10 cho CHẨN_ĐOÁN hoặc RxNorm cho THUỐC. "
    "Chỉ được chọn mã từ shortlist đã cho. "
    "Nếu không có mã nào đủ chắc, trả về candidates rỗng. "
    "Luôn trả JSON đúng schema {\"candidates\":[\"...\"]} và tối đa 3 mã theo thứ tự tin cậy giảm dần."
)


@dataclass(slots=True)
class SemanticExample:
    example_id: str
    file_name: str
    entity_type: str
    mention: str
    position: list[int]
    assertions: list[str]
    gold_candidates: list[str]
    context: str
    shortlist: list[dict]

    def to_record(self) -> dict:
        return {
            "id": self.example_id,
            "file_name": self.file_name,
            "entity_type": self.entity_type,
            "mention": self.mention,
            "position": self.position,
            "assertions": self.assertions,
            "gold_candidates": self.gold_candidates,
            "context": self.context,
            "shortlist": self.shortlist,
        }


def _normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _context_window(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].strip()
    return snippet


def _record_label_map(records: list[KnowledgeRecord]) -> dict[str, str]:
    return {record.code: record.label for record in records}


def _rank_shortlist(
    mention: str,
    entity_type: str,
    knowledge_base: KnowledgeBase,
    shortlist_size: int,
    min_confidence: float,
) -> list[str]:
    config = MatchingConfig(max_candidates=shortlist_size, min_confidence=min_confidence)
    records = knowledge_base.icd10 if entity_type == "CHẨN_ĐOÁN" else knowledge_base.rxnorm
    return [candidate.code for candidate in rank_candidates_indexed(mention, records, config, knowledge_base, entity_type)]


def _build_shortlist(
    mention: str,
    entity_type: str,
    gold_candidates: list[str],
    knowledge_base: KnowledgeBase,
    shortlist_size: int,
    min_confidence: float,
    label_maps: dict[str, dict[str, str]],
) -> list[dict]:
    ranked_codes = _rank_shortlist(mention, entity_type, knowledge_base, shortlist_size, min_confidence)
    merged: list[str] = []
    for code in ranked_codes + gold_candidates:
        if code and code not in merged:
            merged.append(code)
    merged = merged[:shortlist_size]
    label_map = label_maps["icd10"] if entity_type == "CHẨN_ĐOÁN" else label_maps["rxnorm"]
    shortlist = []
    for code in merged:
        shortlist.append(
            {
                "code": code,
                "label": label_map.get(code, ""),
                "is_gold": code in gold_candidates,
            }
        )
    return shortlist


def collect_semantic_examples(
    input_dir: str | Path,
    gold_dir: str | Path,
    config_path: str | Path,
    context_window: int = 220,
    shortlist_size: int = 10,
    shortlist_min_confidence: float = 0.1,
) -> tuple[list[SemanticExample], list[dict]]:
    validate_output_directory(input_dir, gold_dir)
    config = load_config(config_path)
    knowledge_base = load_knowledge_base(config.knowledge_base.icd10_path, config.knowledge_base.rxnorm_path)
    label_maps = {
        "icd10": _record_label_map(knowledge_base.icd10),
        "rxnorm": _record_label_map(knowledge_base.rxnorm),
    }

    examples: list[SemanticExample] = []
    unmatched: list[dict] = []
    for input_path in sorted(Path(input_dir).glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        entities = json.loads((Path(gold_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        for entity in entities:
            entity_type = str(entity["type"])
            if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
                continue
            mention = str(entity["text"])
            start, end = [int(value) for value in entity["position"]]
            gold_candidates = [str(code) for code in entity.get("candidates", []) if str(code)]
            record = {
                "file_name": input_path.name,
                "entity_type": entity_type,
                "mention": mention,
                "position": [start, end],
                "assertions": [str(value) for value in entity.get("assertions", [])],
                "context": _context_window(raw_text, start, end, context_window),
            }
            if not gold_candidates:
                unmatched.append(record)
                continue
            shortlist = _build_shortlist(
                mention=mention,
                entity_type=entity_type,
                gold_candidates=gold_candidates,
                knowledge_base=knowledge_base,
                shortlist_size=shortlist_size,
                min_confidence=shortlist_min_confidence,
                label_maps=label_maps,
            )
            example_id = f"{input_path.stem}:{start}:{end}:{entity_type}"
            examples.append(
                SemanticExample(
                    example_id=example_id,
                    file_name=input_path.name,
                    entity_type=entity_type,
                    mention=mention,
                    position=[start, end],
                    assertions=[str(value) for value in entity.get("assertions", [])],
                    gold_candidates=gold_candidates,
                    context=record["context"],
                    shortlist=shortlist,
                )
            )
    return examples, unmatched


def _group_key(example: SemanticExample) -> str:
    normalized = _normalize_text(example.mention)
    code_key = ",".join(example.gold_candidates)
    return f"{example.entity_type}|{normalized}|{code_key}"


def split_semantic_examples(
    examples: list[SemanticExample],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[SemanticExample]]:
    grouped: dict[str, list[SemanticExample]] = {}
    for example in examples:
        grouped.setdefault(_group_key(example), []).append(example)

    keys = list(grouped)
    random.Random(seed).shuffle(keys)

    total = len(keys)
    train_cut = int(total * train_ratio)
    dev_cut = int(total * (train_ratio + dev_ratio))

    split_keys = {
        "train": keys[:train_cut],
        "dev": keys[train_cut:dev_cut],
        "test": keys[dev_cut:],
    }

    splits: dict[str, list[SemanticExample]] = {"train": [], "dev": [], "test": []}
    for split_name, subset_keys in split_keys.items():
        rows: list[SemanticExample] = []
        for key in subset_keys:
            rows.extend(grouped[key])
        rows.sort(key=lambda item: (item.file_name, item.position[0], item.position[1], item.entity_type))
        splits[split_name] = rows
    return splits


def select_support_examples(
    train_examples: list[SemanticExample],
    support_size_per_type: int,
) -> list[SemanticExample]:
    selected: list[SemanticExample] = []
    seen_keys: set[str] = set()
    for entity_type in ("CHẨN_ĐOÁN", "THUỐC"):
        typed_examples = [example for example in train_examples if example.entity_type == entity_type]
        typed_examples.sort(
            key=lambda item: (
                len(item.gold_candidates),
                len(_normalize_text(item.mention).split()),
                hashlib.sha1(item.example_id.encode("utf-8")).hexdigest(),
            ),
            reverse=True,
        )
        count = 0
        for example in typed_examples:
            key = _group_key(example)
            if key in seen_keys:
                continue
            selected.append(example)
            seen_keys.add(key)
            count += 1
            if count >= support_size_per_type:
                break
    selected.sort(key=lambda item: (item.entity_type, item.file_name, item.position[0]))
    return selected


def _token_overlap_score(left: str, right: str) -> tuple[int, int]:
    left_tokens = set(_normalize_text(left).split())
    right_tokens = set(_normalize_text(right).split())
    return len(left_tokens & right_tokens), len(right_tokens)


def support_examples_for_query(
    query: SemanticExample,
    support_pool: list[SemanticExample],
    shots_per_query: int,
) -> list[SemanticExample]:
    candidates = [example for example in support_pool if example.entity_type == query.entity_type and example.example_id != query.example_id]
    candidates.sort(
        key=lambda item: (
            _token_overlap_score(query.mention, item.mention),
            item.gold_candidates == query.gold_candidates,
            -len(item.context),
            hashlib.sha1(item.example_id.encode("utf-8")).hexdigest(),
        ),
        reverse=True,
    )
    return candidates[:shots_per_query]


def _shortlist_text(shortlist: list[dict]) -> str:
    lines = []
    for row in shortlist:
        label = row["label"] or ""
        lines.append(f"- {row['code']}: {label}")
    return "\n".join(lines)


def build_user_prompt(example: SemanticExample) -> str:
    target = "ICD-10" if example.entity_type == "CHẨN_ĐOÁN" else "RxNorm"
    return (
        f"Loại thực thể: {example.entity_type}\n"
        f"Hệ mã đích: {target}\n"
        f"Mention: {example.mention}\n"
        f"Assertions: {', '.join(example.assertions) if example.assertions else 'none'}\n"
        f"Ngữ cảnh:\n{example.context}\n\n"
        f"Shortlist mã được phép chọn:\n{_shortlist_text(example.shortlist)}\n\n"
        "Trả về JSON đúng schema {\"candidates\":[\"...\"]}. "
        "Nếu không có mã nào phù hợp trong shortlist, trả {\"candidates\":[]}."
    )


def build_assistant_response(example: SemanticExample) -> str:
    return json.dumps({"candidates": example.gold_candidates[:3]}, ensure_ascii=False)


def _messages_from_examples(final_example: SemanticExample, support_examples: list[SemanticExample]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for support in support_examples:
        messages.append({"role": "user", "content": build_user_prompt(support)})
        messages.append({"role": "assistant", "content": build_assistant_response(support)})
    messages.append({"role": "user", "content": build_user_prompt(final_example)})
    return messages


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if rows else ""), encoding="utf-8")
    return path


def export_semantic_dataset_bundle(
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
    support_size_per_type: int = 6,
    shots_per_query: int = 4,
) -> Path:
    examples, unmatched = collect_semantic_examples(
        input_dir=input_dir,
        gold_dir=gold_dir,
        config_path=config_path,
        context_window=context_window,
        shortlist_size=shortlist_size,
        shortlist_min_confidence=shortlist_min_confidence,
    )
    splits = split_semantic_examples(examples, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)
    support_pool = select_support_examples(splits["train"], support_size_per_type=support_size_per_type)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _write_jsonl(destination / "semantic_examples.jsonl", [example.to_record() for example in examples])
    _write_jsonl(destination / "semantic_unmapped.jsonl", unmatched)
    _write_jsonl(destination / "support.jsonl", [example.to_record() for example in support_pool])

    for split_name, split_examples in splits.items():
        _write_jsonl(destination / f"{split_name}.jsonl", [example.to_record() for example in split_examples])

    zero_shot_eval_rows: list[dict] = []
    few_shot_eval_rows: list[dict] = []
    sft_train_rows: list[dict] = []
    sft_dev_rows: list[dict] = []

    for example in splits["test"]:
        zero_shot_eval_rows.append(
            {
                "id": example.example_id,
                "messages": _messages_from_examples(example, []),
                "expected": {"candidates": example.gold_candidates[:3]},
                "metadata": example.to_record(),
            }
        )
        support_examples = support_examples_for_query(example, support_pool, shots_per_query=shots_per_query)
        few_shot_eval_rows.append(
            {
                "id": example.example_id,
                "messages": _messages_from_examples(example, support_examples),
                "expected": {"candidates": example.gold_candidates[:3]},
                "metadata": {
                    **example.to_record(),
                    "support_ids": [row.example_id for row in support_examples],
                },
            }
        )

    for split_name, destination_rows in (("train", sft_train_rows), ("dev", sft_dev_rows)):
        for example in splits[split_name]:
            destination_rows.append(
                {
                    "messages": _messages_from_examples(example, [])
                    + [{"role": "assistant", "content": build_assistant_response(example)}],
                    "metadata": example.to_record(),
                }
            )

    _write_jsonl(destination / "zero_shot_eval.jsonl", zero_shot_eval_rows)
    _write_jsonl(destination / "few_shot_eval.jsonl", few_shot_eval_rows)
    _write_jsonl(destination / "sft_train.jsonl", sft_train_rows)
    _write_jsonl(destination / "sft_dev.jsonl", sft_dev_rows)

    summary = {
        "total_semantic_examples": len(examples),
        "unmapped_examples": len(unmatched),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "test_examples": len(splits["test"]),
        "support_examples": len(support_pool),
        "support_size_per_type": support_size_per_type,
        "shots_per_query": shots_per_query,
        "shortlist_size": shortlist_size,
        "entity_type_counts": {
            "CHẨN_ĐOÁN": sum(example.entity_type == "CHẨN_ĐOÁN" for example in examples),
            "THUỐC": sum(example.entity_type == "THUỐC" for example in examples),
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare few-shot/zero-shot and SFT datasets for Viettel-style semantic ICD/RxNorm mapping."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--gold_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_window", type=int, default=220)
    parser.add_argument("--shortlist_size", type=int, default=10)
    parser.add_argument("--shortlist_min_confidence", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--support_size_per_type", type=int, default=6)
    parser.add_argument("--shots_per_query", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = export_semantic_dataset_bundle(
        input_dir=args.input_dir,
        gold_dir=args.gold_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        context_window=args.context_window,
        shortlist_size=args.shortlist_size,
        shortlist_min_confidence=args.shortlist_min_confidence,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        support_size_per_type=args.support_size_per_type,
        shots_per_query=args.shots_per_query,
    )
    print(f"Semantic dataset bundle exported to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
