from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .qwen_fixed_span import QWEN_TARGET_TYPES, build_runtime_example
from .validation import ValidationError, validate_output_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze risky semantic/assertion cases for a fixed-span Viettel subset such as raw 68-100."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--span_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report_path")
    parser.add_argument("--context_window", type=int, default=220)
    parser.add_argument("--shortlist_size", type=int, default=10)
    parser.add_argument("--shortlist_min_confidence", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=30)
    return parser


def _check_duplicates(values: list[str]) -> bool:
    return len(values) != len(set(values))


def analyze_subset(
    input_dir: str | Path,
    span_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    context_window: int,
    shortlist_size: int,
    shortlist_min_confidence: float,
    top_k: int,
) -> dict:
    validate_output_directory(input_dir, output_dir)

    repeated_mentions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    risky_cases: list[dict] = []
    per_type = Counter()
    empty_candidates_by_type = Counter()
    duplicate_candidates = 0
    out_of_shortlist = 0
    files_with_issues = Counter()

    for input_path in sorted(Path(input_dir).glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        gold_entities = json.loads((Path(span_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        predicted_entities = json.loads((Path(output_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        if len(gold_entities) != len(predicted_entities):
            raise ValidationError(f"{input_path.stem}.json: predicted/gold entity count mismatch")

        for gold_entity, predicted_entity in zip(gold_entities, predicted_entities):
            entity_type = str(predicted_entity["type"])
            if entity_type not in QWEN_TARGET_TYPES:
                continue

            per_type[entity_type] += 1
            example = build_runtime_example(
                raw_text=raw_text,
                file_name=input_path.name,
                entity=gold_entity,
                config_path=config_path,
                context_window=context_window,
                shortlist_size=shortlist_size,
                shortlist_min_confidence=shortlist_min_confidence,
            )

            shortlist_codes = [item["code"] for item in example.shortlist]
            predicted_candidates = [str(code) for code in predicted_entity.get("candidates", [])]
            predicted_assertions = [str(value) for value in predicted_entity.get("assertions", [])]
            repeated_mentions[(entity_type, str(predicted_entity["text"]).strip().lower())].append(
                {
                    "file_name": input_path.name,
                    "candidates": predicted_candidates,
                    "assertions": predicted_assertions,
                    "position": predicted_entity["position"],
                }
            )

            reasons: list[str] = []
            if entity_type in {"CHẨN_ĐOÁN", "THUỐC"} and not predicted_candidates:
                empty_candidates_by_type[entity_type] += 1
                if shortlist_codes:
                    reasons.append("empty_candidates_with_nonempty_shortlist")
            if _check_duplicates(predicted_candidates):
                duplicate_candidates += 1
                reasons.append("duplicate_candidates")
            if any(code not in shortlist_codes for code in predicted_candidates):
                out_of_shortlist += 1
                reasons.append("candidate_out_of_shortlist")
            if predicted_candidates and shortlist_codes and predicted_candidates[0] != shortlist_codes[0]:
                reasons.append("predicted_not_top1_shortlist")
            if len(predicted_candidates) > 1:
                reasons.append("multi_code_prediction")
            if example.shortlist and example.shortlist[0].get("score", 0.0) < 0.35:
                reasons.append("low_confidence_shortlist_top1")

            if reasons:
                files_with_issues[input_path.name] += 1
                risky_cases.append(
                    {
                        "file_name": input_path.name,
                        "text": predicted_entity["text"],
                        "type": entity_type,
                        "position": predicted_entity["position"],
                        "predicted_candidates": predicted_candidates,
                        "predicted_assertions": predicted_assertions,
                        "shortlist": example.shortlist[:5],
                        "reasons": reasons,
                        "context": example.context,
                    }
                )

    inconsistent_mentions: list[dict] = []
    for (entity_type, mention), rows in repeated_mentions.items():
        unique_candidates = {tuple(row["candidates"]) for row in rows}
        unique_assertions = {tuple(sorted(row["assertions"])) for row in rows}
        if len(unique_candidates) > 1 or len(unique_assertions) > 1:
            inconsistent_mentions.append(
                {
                    "type": entity_type,
                    "mention": mention,
                    "count": len(rows),
                    "variants": rows[:8],
                }
            )

    risky_cases.sort(
        key=lambda row: (
            -len(row["reasons"]),
            row["type"] == "CHẨN_ĐOÁN",
            row["file_name"],
            row["position"][0],
        ),
        reverse=True,
    )
    inconsistent_mentions.sort(key=lambda row: (-row["count"], row["type"], row["mention"]))

    return {
        "input_dir": str(Path(input_dir).resolve()),
        "span_dir": str(Path(span_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "semantic_entity_counts": dict(per_type),
        "empty_candidates_by_type": dict(empty_candidates_by_type),
        "duplicate_candidate_entities": duplicate_candidates,
        "out_of_shortlist_entities": out_of_shortlist,
        "files_with_issues": files_with_issues.most_common(),
        "top_risky_cases": risky_cases[:top_k],
        "top_inconsistent_mentions": inconsistent_mentions[:top_k],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = analyze_subset(
        input_dir=args.input_dir,
        span_dir=args.span_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        context_window=args.context_window,
        shortlist_size=args.shortlist_size,
        shortlist_min_confidence=args.shortlist_min_confidence,
        top_k=args.top_k,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
