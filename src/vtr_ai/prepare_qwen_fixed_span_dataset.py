from __future__ import annotations

import argparse

from .qwen_fixed_span import export_qwen_fixed_span_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert mapped Viettel entities into fixed-span Qwen SFT/eval datasets."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = export_qwen_fixed_span_dataset(
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
    )
    print(f"Fixed-span Qwen dataset exported to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
