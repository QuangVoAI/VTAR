from __future__ import annotations

import argparse
import json
from pathlib import Path

from .multi_agent_fixed_span import FixedSpanMultiAgentOrchestrator, build_qwen_multi_agent_generator
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
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict = {"torch_dtype": "auto"}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loading adapter_path requires peft to be installed.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        tokenizer, model = _load_model(args.model_name, args.adapter_path)
    except ImportError:
        parser.error("Inference requires transformers and torch; adapter loading also needs peft.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = FixedSpanMultiAgentOrchestrator(
        config_path=args.config,
        context_window=args.context_window,
        shortlist_size=args.shortlist_size,
        shortlist_min_confidence=args.shortlist_min_confidence,
        generator=build_qwen_multi_agent_generator(
            tokenizer=tokenizer,
            model=model,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    for input_path in sorted(Path(args.input_dir).glob("*.txt"), key=lambda path: path.name):
        raw_text = input_path.read_text(encoding="utf-8")
        entities = json.loads((Path(args.span_dir) / f"{input_path.stem}.json").read_text(encoding="utf-8"))
        merged_entities = orchestrator.process_entities(
            raw_text=raw_text,
            file_name=input_path.name,
            entities=entities,
        )

        (output_dir / f"{input_path.stem}.json").write_text(
            json.dumps(merged_entities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    validate_output_directory(args.input_dir, output_dir)
    print(f"Qwen fixed-span outputs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
