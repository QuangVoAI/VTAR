from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .schemas import ASSERTION_TYPES
from .validation import validate_output_directory


@dataclass(slots=True)
class Score:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class EvaluationSummary:
    files: int
    entity_span: Score
    entity_span_type: Score
    entity_record: Score
    assertions: Score
    candidates: Score

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "entity_span": asdict(self.entity_span),
            "entity_span_type": asdict(self.entity_span_type),
            "entity_record": asdict(self.entity_record),
            "assertions": asdict(self.assertions),
            "candidates": asdict(self.candidates),
        }


def _score(true_positive: int, false_positive: int, false_negative: int) -> Score:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Score(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _load_json_list(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected top-level list")
    return data


def _entity_key_span(entity: dict) -> tuple[int, int]:
    start, end = entity["position"]
    return int(start), int(end)


def _entity_key_span_type(entity: dict) -> tuple[int, int, str]:
    start, end = entity["position"]
    return int(start), int(end), str(entity["type"])


def _entity_key_record(entity: dict) -> tuple[int, int, str, str, tuple[str, ...], tuple[str, ...]]:
    start, end = entity["position"]
    assertions = tuple(sorted(str(item) for item in entity["assertions"]))
    candidates = tuple(str(item) for item in entity["candidates"])
    return int(start), int(end), str(entity["type"]), str(entity["text"]), assertions, candidates


def _set_prf(predicted: set[tuple], gold: set[tuple]) -> Score:
    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    return _score(true_positive, false_positive, false_negative)


def _merge_scores(scores: list[Score]) -> Score:
    true_positive = sum(item.true_positive for item in scores)
    false_positive = sum(item.false_positive for item in scores)
    false_negative = sum(item.false_negative for item in scores)
    return _score(true_positive, false_positive, false_negative)


def _assertion_items(entities: list[dict]) -> set[tuple[int, int, str, str]]:
    items: set[tuple[int, int, str, str]] = set()
    for entity in entities:
        start, end = entity["position"]
        entity_type = str(entity["type"])
        if entity_type not in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}:
            continue
        for assertion in entity["assertions"]:
            if assertion in ASSERTION_TYPES:
                items.add((int(start), int(end), entity_type, str(assertion)))
    return items


def _candidate_items(entities: list[dict]) -> set[tuple[int, int, str, str]]:
    items: set[tuple[int, int, str, str]] = set()
    for entity in entities:
        start, end = entity["position"]
        entity_type = str(entity["type"])
        if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
            continue
        for candidate in entity["candidates"]:
            items.add((int(start), int(end), entity_type, str(candidate)))
    return items


def evaluate_entity_lists(predicted_entities: list[dict], gold_entities: list[dict]) -> dict[str, Score]:
    return {
        "entity_span": _set_prf(
            {_entity_key_span(item) for item in predicted_entities},
            {_entity_key_span(item) for item in gold_entities},
        ),
        "entity_span_type": _set_prf(
            {_entity_key_span_type(item) for item in predicted_entities},
            {_entity_key_span_type(item) for item in gold_entities},
        ),
        "entity_record": _set_prf(
            {_entity_key_record(item) for item in predicted_entities},
            {_entity_key_record(item) for item in gold_entities},
        ),
        "assertions": _set_prf(_assertion_items(predicted_entities), _assertion_items(gold_entities)),
        "candidates": _set_prf(_candidate_items(predicted_entities), _candidate_items(gold_entities)),
    }


def evaluate_output_directories(
    input_dir: str | Path,
    predicted_dir: str | Path,
    gold_dir: str | Path,
) -> EvaluationSummary:
    validate_output_directory(input_dir, predicted_dir)
    validate_output_directory(input_dir, gold_dir)
    input_files = sorted(Path(input_dir).glob("*.txt"), key=lambda path: path.name)

    entity_span_scores: list[Score] = []
    entity_span_type_scores: list[Score] = []
    entity_record_scores: list[Score] = []
    assertion_scores: list[Score] = []
    candidate_scores: list[Score] = []

    for input_path in input_files:
        predicted_path = Path(predicted_dir) / f"{input_path.stem}.json"
        gold_path = Path(gold_dir) / f"{input_path.stem}.json"
        if not gold_path.exists():
            raise FileNotFoundError(f"missing gold file {gold_path.name}")
        if not predicted_path.exists():
            raise FileNotFoundError(f"missing prediction file {predicted_path.name}")
        per_file = evaluate_entity_lists(_load_json_list(predicted_path), _load_json_list(gold_path))
        entity_span_scores.append(per_file["entity_span"])
        entity_span_type_scores.append(per_file["entity_span_type"])
        entity_record_scores.append(per_file["entity_record"])
        assertion_scores.append(per_file["assertions"])
        candidate_scores.append(per_file["candidates"])

    return EvaluationSummary(
        files=len(input_files),
        entity_span=_merge_scores(entity_span_scores),
        entity_span_type=_merge_scores(entity_span_type_scores),
        entity_record=_merge_scores(entity_record_scores),
        assertions=_merge_scores(assertion_scores),
        candidates=_merge_scores(candidate_scores),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate predicted Viettel-style outputs against gold labels.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gold_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_output_directories(args.input_dir, args.pred_dir, args.gold_dir)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
