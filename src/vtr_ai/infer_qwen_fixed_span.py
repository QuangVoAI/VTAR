from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qwen_fixed_span import (
    QWEN_TARGET_TYPES,
    SYSTEM_PROMPT,
    apply_prediction_to_entity,
    build_fixed_span_user_prompt,
    build_runtime_example,
    parse_qwen_json_response,
)
from .validation import validate_output_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen fixed-span inference and merge assertions/candidates back into Viettel JSON."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--span_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_path")
    parser.add_argument("--context_window", type=int, default=220)
    parser.add_argument("--shortlist_size", type=int, default=10)
    parser.add_argument("--shortlist_min_confidence", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    return parser


def _load_model(model_name: str, adapter_path: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if adapter_path:
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loading adapter_path requires peft to be installed.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model


def _generate_prediction(tokenizer, model, messages: list[dict], max_length: int, max_new_tokens: int) -> dict[str, list[str]]:
    encoded = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        tokenizer, model = _load_model(args.model_name, args.adapter_path)
    except ImportError:
        parser.error("Inference requires transformers and torch; adapter loading also needs peft.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(Path(args.input_dir).glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        entities = json.loads((Path(args.span_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        merged_entities: list[dict] = []
        for entity in entities:
            entity_type = str(entity["type"])
            if entity_type not in QWEN_TARGET_TYPES:
                merged_entities.append(apply_prediction_to_entity(entity, {"assertions": [], "candidates": []}))
                continue
            example = build_runtime_example(
                raw_text=raw_text,
                file_name=input_path.name,
                entity=entity,
                config_path=args.config,
                context_window=args.context_window,
                shortlist_size=args.shortlist_size,
                shortlist_min_confidence=args.shortlist_min_confidence,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_fixed_span_user_prompt(example)},
            ]
            prediction = _generate_prediction(
                tokenizer=tokenizer,
                model=model,
                messages=messages,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            merged_entities.append(apply_prediction_to_entity(entity, prediction))

        (output_dir / f"{input_path.stem}.json").write_text(
            json.dumps(merged_entities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    validate_output_directory(args.input_dir, output_dir)
    print(f"Qwen fixed-span outputs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
