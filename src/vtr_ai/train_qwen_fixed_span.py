from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5 style chat models on fixed-span Viettel assertion/candidate data."
    )
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_init_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--eval_strategy", default="epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            rows.append(json.loads(raw_line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        parser.error("Training requires transformers, datasets, and torch.")

    if args.use_lora or args.load_in_4bit:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        except ImportError:
            parser.error("LoRA/4bit training requires peft. Install peft and bitsandbytes if needed.")

    train_rows = _load_jsonl(args.train_jsonl)
    eval_rows = _load_jsonl(args.eval_jsonl) if args.eval_jsonl else []
    if not train_rows:
        parser.error("Training set is empty.")

    tokenizer_source = args.adapter_init_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        parser.error("Tokenizer does not expose a chat_template. Use a chat model/tokenizer such as Qwen2.5.")

    model_kwargs: dict = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        if args.adapter_init_path:
            model = PeftModel.from_pretrained(model, args.adapter_init_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config)

    def _preprocess(row: dict) -> dict:
        prompt_messages = row["messages"][:-1]
        full_messages = row["messages"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_encoded = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_length,
        )
        full_encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = list(full_encoded["input_ids"])
        attention_mask = list(full_encoded["attention_mask"])
        prompt_len = min(len(prompt_encoded["input_ids"]), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = Dataset.from_list(train_rows).map(_preprocess, remove_columns=["messages", "metadata"])
    eval_dataset = (
        Dataset.from_list(eval_rows).map(_preprocess, remove_columns=["messages", "metadata"])
        if eval_rows
        else None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        report_to=[],
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        (Path(args.output_dir) / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Saved Qwen fixed-span checkpoint to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
