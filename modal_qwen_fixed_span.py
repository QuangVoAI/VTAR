from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = "vtr-qwen-fixed-span"
REMOTE_ROOT = "/root/vtr"
REMOTE_OUTPUT_ROOT = "/root/outputs"
TRAIN_VOLUME_NAME = "vtr-qwen-fixed-span-train"
INFER_VOLUME_NAME = "vtr-qwen-fixed-span-infer"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pip>=25.0",
        "setuptools>=68",
        "wheel",
        "torch>=2.3.0",
        "transformers>=4.52.0",
        "datasets>=3.0.0",
        "accelerate>=1.1.0",
        "peft>=0.17.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "safetensors",
        "pyyaml",
        "huggingface_hub",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src"})
    .add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir("artifacts", remote_path=f"{REMOTE_ROOT}/artifacts")
    .add_local_dir("review_packet_68_100", remote_path=f"{REMOTE_ROOT}/review_packet_68_100")
    .add_local_dir("tmp", remote_path=f"{REMOTE_ROOT}/tmp")
    .add_local_file("pyproject.toml", remote_path=f"{REMOTE_ROOT}/pyproject.toml")
)

app = modal.App(APP_NAME)
train_volume = modal.Volume.from_name(TRAIN_VOLUME_NAME, create_if_missing=True)
infer_volume = modal.Volume.from_name(INFER_VOLUME_NAME, create_if_missing=True)


def _login_hf_if_present() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def _write_modal_config() -> str:
    config_path = Path("/tmp/config.modal.yaml")
    config_path.write_text(
        "\n".join(
            [
                "knowledge_base:",
                f"  icd10_path: {REMOTE_ROOT}/src/vtr_ai/data/icd10_standard.json",
                f"  rxnorm_path: {REMOTE_ROOT}/src/vtr_ai/data/rxnorm_standard.json",
                f"  abbreviations_path: {REMOTE_ROOT}/src/vtr_ai/data/vi_abbreviations.json",
                "",
                "matching:",
                "  max_candidates: 3",
                "  min_confidence: 0.55",
                "",
                "rules:",
                "  symptom_window: 80",
                "  assertion_window: 60",
                "",
                "output:",
                "  pretty: true",
                "",
                "ner:",
                "  backend: hybrid",
                "  enable_model_backend: false",
                "  provider: lexical",
                f"  checkpoint_path: {REMOTE_ROOT}/src/vtr_ai/data/checkpoints/lexical_ner.json",
                f"  metadata_path: {REMOTE_ROOT}/src/vtr_ai/data/checkpoints/transformers_ner_metadata.json",
                "  fallback_to_rules: true",
                "  min_score: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(config_path)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 8,
    volumes={REMOTE_OUTPUT_ROOT: train_volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def remote_train(
    train_jsonl: str = f"{REMOTE_ROOT}/artifacts/qwen_fixed_span_68_100/sft_train.jsonl",
    eval_jsonl: str = f"{REMOTE_ROOT}/artifacts/qwen_fixed_span_68_100/sft_dev.jsonl",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_init_path: str = "",
    output_subdir: str = "qwen25-7b-fixed-span-lora",
    num_train_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    max_length: int = 1536,
    load_in_4bit: bool = True,
) -> str:
    import sys

    _login_hf_if_present()
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from vtr_ai.train_qwen_fixed_span import main as train_main

    output_dir = f"{REMOTE_OUTPUT_ROOT}/{output_subdir}"
    args = [
        "--train_jsonl",
        train_jsonl,
        "--eval_jsonl",
        eval_jsonl,
        "--model_name",
        model_name,
        "--output_dir",
        output_dir,
        "--max_length",
        str(max_length),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--num_train_epochs",
        str(num_train_epochs),
        "--learning_rate",
        str(learning_rate),
        "--use_lora",
    ]
    if adapter_init_path:
        args.extend(
            [
                "--adapter_init_path",
                adapter_init_path,
            ]
        )
    if load_in_4bit:
        args.append("--load_in_4bit")
    code = train_main(args)
    if code != 0:
        raise RuntimeError(f"Training failed with exit code {code}")
    train_volume.commit()
    return output_dir


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={REMOTE_OUTPUT_ROOT: infer_volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def remote_infer(
    input_dir: str = f"{REMOTE_ROOT}/review_packet_68_100",
    span_dir: str = f"{REMOTE_ROOT}/review_packet_68_100",
    config_path: str = "",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: str = "SpringWang08/qwen25-7b-fixed-span-viettel-lora",
    output_subdir: str = "viettel_qwen_fixed_span_output",
    context_window: int = 220,
    shortlist_size: int = 10,
    shortlist_min_confidence: float = 0.1,
    max_length: int = 1536,
    max_new_tokens: int = 96,
) -> str:
    import sys

    _login_hf_if_present()
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from vtr_ai.infer_qwen_fixed_span import main as infer_main

    if not config_path:
        config_path = _write_modal_config()
    output_dir = f"{REMOTE_OUTPUT_ROOT}/{output_subdir}"
    code = infer_main(
        [
            "--input_dir",
            input_dir,
            "--span_dir",
            span_dir,
            "--output_dir",
            output_dir,
            "--config",
            config_path,
            "--model_name",
            model_name,
            "--adapter_path",
            adapter_path,
            "--context_window",
            str(context_window),
            "--shortlist_size",
            str(shortlist_size),
            "--shortlist_min_confidence",
            str(shortlist_min_confidence),
            "--max_length",
            str(max_length),
            "--max_new_tokens",
            str(max_new_tokens),
        ]
    )
    if code != 0:
        raise RuntimeError(f"Inference failed with exit code {code}")
    infer_volume.commit()
    return output_dir


@app.local_entrypoint()
def train(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_init_path: str = "",
    output_subdir: str = "qwen25-7b-fixed-span-lora",
    num_train_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    max_length: int = 1536,
    load_in_4bit: bool = True,
) -> None:
    remote_dir = remote_train.remote(
        model_name=model_name,
        adapter_init_path=adapter_init_path,
        output_subdir=output_subdir,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
    )
    print(json.dumps({"status": "ok", "remote_output_dir": remote_dir}, ensure_ascii=False, indent=2))
    print(
        f"Download checkpoint with: modal volume get {TRAIN_VOLUME_NAME} {output_subdir} ./artifacts/{output_subdir}"
    )


@app.local_entrypoint(name="train_continue")
def train_continue(
    adapter_init_path: str = "SpringWang08/qwen25-7b-fixed-span-viettel-lora",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    train_jsonl: str = f"{REMOTE_ROOT}/artifacts/qwen_fixed_span_68_100/sft_train.jsonl",
    eval_jsonl: str = f"{REMOTE_ROOT}/artifacts/qwen_fixed_span_68_100/sft_dev.jsonl",
    output_subdir: str = "qwen25-7b-fixed-span-continued-lora",
    num_train_epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-4,
    max_length: int = 1536,
    load_in_4bit: bool = True,
) -> None:
    remote_dir = remote_train.remote(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        model_name=model_name,
        adapter_init_path=adapter_init_path,
        output_subdir=output_subdir,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
    )
    print(json.dumps({"status": "ok", "remote_output_dir": remote_dir}, ensure_ascii=False, indent=2))
    print(
        f"Download checkpoint with: modal volume get {TRAIN_VOLUME_NAME} {output_subdir} ./artifacts/{output_subdir}"
    )


@app.local_entrypoint(name="infer")
def infer(
    adapter_path: str = "SpringWang08/qwen25-7b-fixed-span-viettel-lora",
    output_subdir: str = "viettel_qwen_fixed_span_output",
    max_new_tokens: int = 96,
) -> None:
    remote_dir = remote_infer.remote(
        adapter_path=adapter_path,
        output_subdir=output_subdir,
        max_new_tokens=max_new_tokens,
    )
    print(json.dumps({"status": "ok", "remote_output_dir": remote_dir}, ensure_ascii=False, indent=2))
    print(
        f"Download outputs with: modal volume get {INFER_VOLUME_NAME} {output_subdir} ./outputs/{output_subdir}"
    )


@app.local_entrypoint(name="infer_raw_68_100")
def infer_raw_68_100(
    adapter_path: str = "SpringWang08/qwen25-7b-fixed-span-viettel-lora",
    output_subdir: str = "viettel_qwen_fixed_span_raw_68_100_output",
    max_new_tokens: int = 96,
) -> None:
    remote_dir = remote_infer.remote(
        input_dir=f"{REMOTE_ROOT}/tmp/raw_68_100_input",
        span_dir=f"{REMOTE_ROOT}/tmp/raw_68_100_spans",
        adapter_path=adapter_path,
        output_subdir=output_subdir,
        max_new_tokens=max_new_tokens,
    )
    print(json.dumps({"status": "ok", "remote_output_dir": remote_dir}, ensure_ascii=False, indent=2))
    print(
        f"Download outputs with: modal volume get {INFER_VOLUME_NAME} {output_subdir} ./outputs/{output_subdir}"
    )


@app.local_entrypoint(name="infer_raw_1_100")
def infer_raw_1_100(
    adapter_path: str = "SpringWang08/qwen25-7b-fixed-span-viettel-lora",
    output_subdir: str = "viettel_qwen_fixed_span_raw_1_100_output",
    max_new_tokens: int = 96,
) -> None:
    remote_dir = remote_infer.remote(
        input_dir=f"{REMOTE_ROOT}/tmp/raw_1_100_input",
        span_dir=f"{REMOTE_ROOT}/tmp/raw_1_100_spans",
        adapter_path=adapter_path,
        output_subdir=output_subdir,
        max_new_tokens=max_new_tokens,
    )
    print(json.dumps({"status": "ok", "remote_output_dir": remote_dir}, ensure_ascii=False, indent=2))
    print(
        f"Download outputs with: modal volume get {INFER_VOLUME_NAME} {output_subdir} ./outputs/{output_subdir}"
    )
