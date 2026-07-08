from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ner_labels import ID_TO_LABEL, LABEL_TO_ID
from .ner_training import (
    AnnotatedEntity,
    align_entities_to_offsets,
    export_label_metadata,
    load_annotated_examples,
    split_annotated_examples,
    validate_annotated_example,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a token-classification NER model for Viettel labels.")
    parser.add_argument("--train_jsonl", required=True, help="JSONL with {text, entities:[{position:[s,e], type}]} rows")
    parser.add_argument("--model_name", required=True, help="Base HF model name or local tokenizer/model path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_jsonl", help="Optional held-out JSONL for evaluation")
    parser.add_argument("--validation_ratio", type=float, default=0.0, help="If eval_jsonl is absent, split train_jsonl by this ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        from datasets import Dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        parser.error(
            "Training requires transformers, datasets, and torch. Install them before running train_ner."
        )

    examples = load_annotated_examples(args.train_jsonl)
    for example in examples:
        validate_annotated_example(example)
    if args.eval_jsonl:
        train_examples = examples
        eval_examples = load_annotated_examples(args.eval_jsonl)
        for example in eval_examples:
            validate_annotated_example(example)
    else:
        train_examples, eval_examples = split_annotated_examples(examples, args.validation_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not train_examples:
        parser.error("Training set is empty after loading/splitting annotations.")

    def _example_to_row(example: object) -> dict:
        return {
            "text": example.text,
            "entities": [{"start": entity.start, "end": entity.end, "label": entity.label} for entity in example.entities],
        }

    train_dataset = Dataset.from_list([_example_to_row(example) for example in train_examples])
    eval_dataset = (
        Dataset.from_list([_example_to_row(example) for example in eval_examples])
        if eval_examples
        else None
    )

    def _preprocess(row: dict) -> dict:
        entities = [
            AnnotatedEntity(
                start=int(entity["start"]),
                end=int(entity["end"]),
                label=str(entity["label"]),
            )
            for entity in row["entities"]
        ]
        tokenized = tokenizer(row["text"], truncation=True, return_offsets_mapping=True)
        offsets = [(int(start), int(end)) for start, end in tokenized["offset_mapping"]]
        labels = align_entities_to_offsets(row["text"], offsets, entities)
        tokenized["labels"] = [
            -100 if start == end else LABEL_TO_ID.get(label, LABEL_TO_ID["O"])
            for label, (start, end) in zip(labels, offsets)
        ]
        return tokenized

    tokenized_train_dataset = train_dataset.map(_preprocess, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = (
        eval_dataset.map(_preprocess, remove_columns=eval_dataset.column_names) if eval_dataset is not None else None
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        eval_strategy="epoch" if tokenized_eval_dataset is not None else "no",
        logging_steps=10,
        report_to=[],
        seed=args.seed,
    )

    def _compute_metrics(eval_prediction: tuple) -> dict[str, float]:
        predictions, labels = eval_prediction
        predicted_ids = predictions.argmax(axis=-1)
        true_positive = false_positive = false_negative = correct = total = 0
        for prediction_row, label_row in zip(predicted_ids, labels):
            for predicted_id, label_id in zip(prediction_row, label_row):
                if label_id == -100:
                    continue
                total += 1
                if predicted_id == label_id:
                    correct += 1
                if predicted_id != LABEL_TO_ID["O"] and label_id != LABEL_TO_ID["O"]:
                    if predicted_id == label_id:
                        true_positive += 1
                    else:
                        false_positive += 1
                        false_negative += 1
                elif predicted_id != LABEL_TO_ID["O"] and label_id == LABEL_TO_ID["O"]:
                    false_positive += 1
                elif predicted_id == LABEL_TO_ID["O"] and label_id != LABEL_TO_ID["O"]:
                    false_negative += 1
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = correct / total if total else 0.0
        return {
            "token_accuracy": accuracy,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
        }

    import inspect
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train_dataset,
        "eval_dataset": tokenized_eval_dataset,
        "data_collator": DataCollatorForTokenClassification(tokenizer=tokenizer),
        "compute_metrics": _compute_metrics if tokenized_eval_dataset is not None else None,
    }
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    export_label_metadata(args.output_dir)
    if tokenized_eval_dataset is not None:
        metrics = trainer.evaluate()
        (Path(args.output_dir) / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Saved checkpoint to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
