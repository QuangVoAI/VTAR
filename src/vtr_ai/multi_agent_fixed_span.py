from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .abbreviation import AbbreviationExpander
from .assertion import infer_assertions
from .config import load_config
from .knowledge_base import load_knowledge_base
from .preprocess import build_document
from .qwen_fixed_span import (
    QWEN_TARGET_TYPES,
    SYSTEM_PROMPT,
    apply_prediction_to_entity,
    build_runtime_example,
    parse_qwen_json_response,
)
from .schemas import Entity


LLMGenerator = Callable[[list[dict]], dict[str, list[str]]]


@dataclass(slots=True)
class RetrievalResult:
    entity: dict
    example: object | None
    shortlist_codes: list[str]
    shortlist: list[dict]


class FixedSpanAgent:
    """Validate fixed spans without changing their text, type, or offsets."""

    def run(self, raw_text: str, entities: list[dict]) -> list[dict]:
        validated: list[dict] = []
        for entity in entities:
            position = entity.get("position")
            if not isinstance(position, list) or len(position) != 2:
                raise ValueError(f"Invalid fixed span position: {position!r}")
            start, end = position
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"Fixed span offsets must be integers: {position!r}")
            if start < 0 or end < start or end > len(raw_text):
                raise ValueError(f"Fixed span is out of bounds: {position!r}")
            if raw_text[start:end] != str(entity.get("text", "")):
                raise ValueError(
                    f"Fixed span mismatch for {entity.get('text')!r}: {position!r}"
                )
            validated.append(dict(entity))
        return validated


@dataclass(slots=True)
class AssertionResult:
    entity: dict
    assertions: list[str]


@dataclass(slots=True)
class SemanticSelectionResult:
    entity: dict
    candidates: list[str]
    raw_prediction: dict[str, list[str]]


@dataclass(slots=True)
class CandidateJudgeResult:
    entity: dict
    candidates: list[str]


SOURCE_BONUS = {
    "exact": 0.16,
    "indexed": 0.06,
    "embedding": 0.03,
    "gold": 0.0,
}
MULTI_CODE_SIGNAL_PATTERN = re.compile(r"(?:/|\+|\ber\b|\bher2\b)", re.IGNORECASE)


class RetrievalAgent:
    def __init__(
        self,
        config_path: str | Path,
        context_window: int,
        shortlist_size: int,
        shortlist_min_confidence: float,
    ) -> None:
        self.config_path = config_path
        self.context_window = context_window
        self.shortlist_size = shortlist_size
        self.shortlist_min_confidence = shortlist_min_confidence
        config = load_config(config_path)
        self.expander = AbbreviationExpander.from_path(config.knowledge_base.abbreviations_path)

    def _expand_mention(self, mention: str) -> str:
        expanded, _ = self.expander.expand_text_with_mapping(mention)
        return expanded.strip()

    def _merge_shortlists(self, *shortlists: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for shortlist in shortlists:
            for row in shortlist:
                code = str(row["code"])
                if code not in merged or float(row.get("score", 0.0)) > float(merged[code].get("score", 0.0)):
                    merged[code] = dict(row)
        ordered = sorted(merged.values(), key=lambda item: (-float(item.get("score", 0.0)), str(item["code"])))
        return ordered[: self.shortlist_size]

    def run(self, raw_text: str, file_name: str, entity: dict) -> RetrievalResult:
        entity_type = str(entity["type"])
        if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
            return RetrievalResult(entity=entity, example=None, shortlist_codes=[], shortlist=[])
        example = build_runtime_example(
            raw_text=raw_text,
            file_name=file_name,
            entity=entity,
            config_path=self.config_path,
            context_window=self.context_window,
            shortlist_size=self.shortlist_size,
            shortlist_min_confidence=self.shortlist_min_confidence,
            gold_candidates=[str(code) for code in entity.get("candidates", [])],
        )
        merged_shortlist = list(example.shortlist)
        expanded_mention = self._expand_mention(str(entity["text"]))
        if expanded_mention and expanded_mention.lower() != str(entity["text"]).lower():
            expanded_entity = dict(entity)
            expanded_entity["text"] = expanded_mention
            expanded_example = build_runtime_example(
                raw_text=raw_text,
                file_name=file_name,
                entity=expanded_entity,
                config_path=self.config_path,
                context_window=self.context_window,
                shortlist_size=self.shortlist_size,
                shortlist_min_confidence=self.shortlist_min_confidence,
                gold_candidates=[str(code) for code in entity.get("candidates", [])],
            )
            merged_shortlist = self._merge_shortlists(merged_shortlist, list(expanded_example.shortlist))
        example.shortlist = merged_shortlist
        shortlist_codes = [str(item["code"]) for item in merged_shortlist]
        return RetrievalResult(
            entity=entity,
            example=example,
            shortlist_codes=shortlist_codes,
            shortlist=merged_shortlist,
        )


class AssertionAgent:
    def __init__(self, config_path: str | Path) -> None:
        config = load_config(config_path)
        self.window = config.rules.assertion_window
        self.expander = AbbreviationExpander.from_path(config.knowledge_base.abbreviations_path)

    def build_document_once(self, raw_text: str):
        return build_document(raw_text, self.expander)

    def run(self, document, entity: dict) -> AssertionResult:
        entity_type = str(entity["type"])
        if entity_type not in QWEN_TARGET_TYPES:
            return AssertionResult(entity=entity, assertions=[])
        start, end = [int(value) for value in entity["position"]]
        runtime_entity = Entity(
            text=str(entity["text"]),
            start=start,
            end=end,
            entity_type=entity_type,
        )
        mapped_assertions = [
            str(value)
            for value in entity.get("assertions", [])
            if str(value) in {"isNegated", "isFamily", "isHistorical"}
        ]
        assertions = mapped_assertions or infer_assertions(document, runtime_entity, self.window)
        return AssertionResult(entity=entity, assertions=sorted(set(assertions)))


class SemanticSelectionAgent:
    def __init__(self, generator: LLMGenerator) -> None:
        self.generator = generator

    def run(self, retrieval: RetrievalResult, assertion: AssertionResult) -> SemanticSelectionResult:
        entity = retrieval.entity
        entity_type = str(entity["type"])
        if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"} or retrieval.example is None:
            return SemanticSelectionResult(entity=entity, candidates=[], raw_prediction={"assertions": [], "candidates": []})

        example = retrieval.example
        shortlist_lines = "\n".join(
            f"- {item['code']}: {item['label']} | matched_alias={item.get('matched_alias', item['label'])} | score={item.get('score', 0.0):.4f} | source={item.get('source', 'unknown')}"
            for item in example.shortlist
        ) or "- none"
        target = "ICD-10" if entity_type == "CHẨN_ĐOÁN" else "RxNorm"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Loại thực thể: {entity_type}\n"
                    f"Mention: {example.mention}\n"
                    f"Position: [{example.position[0]}, {example.position[1]}]\n"
                    f"Hệ mã đích: {target}\n"
                    f"Ngữ cảnh:\n{example.context}\n\n"
                    f"Assertions do assertion agent dự đoán: {assertion.assertions}\n"
                    f"Shortlist được phép chọn:\n{shortlist_lines}\n\n"
                    "Nhiệm vụ của bạn trong bước này chỉ là chọn candidates từ shortlist. "
                    "Không được tạo code ngoài shortlist. "
                    "Giữ nguyên assertions do assertion agent dự đoán, không tự sửa. "
                    "Có thể trả tối đa 3 mã nếu các mã đều phù hợp với mention và shortlist. "
                    "Ưu tiên giữ các mã mapped có điểm cao; không tự thêm mã ngoài shortlist. "
                    "Nếu shortlist không có mã phù hợp thì trả candidates rỗng. "
                    "Trả đúng JSON schema {\"assertions\":[\"...\"],\"candidates\":[\"...\"]}."
                ),
            },
        ]
        prediction = self.generator(messages)
        allowed = set(retrieval.shortlist_codes)
        candidates = [code for code in prediction.get("candidates", []) if code in allowed]
        return SemanticSelectionResult(
            entity=entity,
            candidates=candidates[:3],
            raw_prediction=prediction,
        )


class CandidateJudgeAgent:
    def _rank_selected(self, candidates: list[str], retrieval: RetrievalResult) -> list[tuple[str, float]]:
        shortlist_map = {str(item["code"]): item for item in retrieval.shortlist}
        ranked: list[tuple[str, float]] = []
        for index, code in enumerate(candidates):
            row = shortlist_map.get(code)
            if row is None:
                continue
            source = str(row.get("source", "embedding"))
            score = float(row.get("score", 0.0)) + SOURCE_BONUS.get(source, 0.0) - index * 0.03
            ranked.append((code, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _fallback_from_shortlist(self, retrieval: RetrievalResult, entity_type: str) -> list[str]:
        if not retrieval.shortlist:
            return []
        top = retrieval.shortlist[0]
        top_score = float(top.get("score", 0.0))
        top_source = str(top.get("source", "embedding"))
        if entity_type == "THUỐC":
            if top_source == "exact" and top_score >= 0.85:
                return [str(top["code"])]
            return []
        if top_score >= 0.95 or (top_source == "exact" and top_score >= 0.8):
            return [str(top["code"])]
        return []

    def _allow_multi_code(self, mention: str, ranked: list[tuple[str, float]]) -> bool:
        if len(ranked) < 2:
            return False
        if not MULTI_CODE_SIGNAL_PATTERN.search(mention):
            return False
        return ranked[0][1] >= 0.95 and ranked[1][1] >= ranked[0][1] - 0.12

    def run(
        self,
        retrieval: RetrievalResult,
        assertion: AssertionResult,
        semantic: SemanticSelectionResult,
    ) -> CandidateJudgeResult:
        entity = retrieval.entity
        entity_type = str(entity["type"])
        if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
            return CandidateJudgeResult(entity=entity, candidates=[])

        allowed = set(retrieval.shortlist_codes)
        mapped_codes: list[str] = []
        for value in entity.get("candidates", []):
            code = str(value).split(":", 1)[0].strip()
            if code in allowed and code not in mapped_codes:
                mapped_codes.append(code)
        if mapped_codes:
            return CandidateJudgeResult(entity=entity, candidates=mapped_codes[:3])

        selected_codes = list(dict.fromkeys(str(code) for code in semantic.candidates if code in allowed))
        if not selected_codes:
            return CandidateJudgeResult(entity=entity, candidates=self._fallback_from_shortlist(retrieval, entity_type))

        ranked = self._rank_selected(selected_codes, retrieval)
        if not ranked:
            return CandidateJudgeResult(entity=entity, candidates=self._fallback_from_shortlist(retrieval, entity_type))
        if entity_type == "THUỐC":
            return CandidateJudgeResult(entity=entity, candidates=[ranked[0][0]])
        if self._allow_multi_code(str(entity["text"]), ranked):
            return CandidateJudgeResult(entity=entity, candidates=[code for code, _ in ranked[:2]])
        return CandidateJudgeResult(entity=entity, candidates=[ranked[0][0]])


class MergeAgent:
    def run(
        self,
        entity: dict,
        assertion: AssertionResult,
        judged: CandidateJudgeResult,
    ) -> dict:
        entity_type = str(entity["type"])
        prediction = {
            "assertions": assertion.assertions if entity_type in QWEN_TARGET_TYPES else [],
            "candidates": judged.candidates if entity_type in {"CHẨN_ĐOÁN", "THUỐC"} else [],
        }
        return apply_prediction_to_entity(entity, prediction)


class FixedSpanMultiAgentOrchestrator:
    def __init__(
        self,
        config_path: str | Path,
        context_window: int,
        shortlist_size: int,
        shortlist_min_confidence: float,
        generator: LLMGenerator,
    ) -> None:
        self.retrieval_agent = RetrievalAgent(
            config_path=config_path,
            context_window=context_window,
            shortlist_size=shortlist_size,
            shortlist_min_confidence=shortlist_min_confidence,
        )
        self.assertion_agent = AssertionAgent(config_path=config_path)
        self.semantic_agent = SemanticSelectionAgent(generator)
        self.candidate_judge_agent = CandidateJudgeAgent()
        self.merge_agent = MergeAgent()
        self.span_agent = FixedSpanAgent()

    def process_entities(self, raw_text: str, file_name: str, entities: list[dict]) -> list[dict]:
        document = self.assertion_agent.build_document_once(raw_text)
        merged_entities: list[dict] = []
        fixed_entities = self.span_agent.run(raw_text, entities)
        for entity in fixed_entities:
            retrieval = self.retrieval_agent.run(raw_text=raw_text, file_name=file_name, entity=entity)
            assertion = self.assertion_agent.run(document, entity)
            semantic = self.semantic_agent.run(retrieval, assertion)
            judged = self.candidate_judge_agent.run(retrieval, assertion, semantic)
            merged_entities.append(self.merge_agent.run(entity, assertion, judged))
        return merged_entities


def build_qwen_multi_agent_generator(tokenizer, model, max_length: int, max_new_tokens: int) -> LLMGenerator:
    def _generator(messages: list[dict]) -> dict[str, list[str]]:
        import torch  # type: ignore

        encoded = tokenizer(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        target_device = getattr(model, "device", None)
        if target_device is None and hasattr(model, "get_input_embeddings"):
            target_device = model.get_input_embeddings().weight.device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        response = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        return parse_qwen_json_response(response)

    return _generator
