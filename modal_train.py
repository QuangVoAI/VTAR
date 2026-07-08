import modal
import os
import subprocess
from pathlib import Path

# Define the container image with required libraries and source code
image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.40.0",
        "datasets",
        "torch",
        "accelerate>=1.1.0",
        "sentencepiece"
    )
    .add_local_dir(os.path.join(os.path.dirname(__file__), "src"), remote_path="/root/src")
)

app = modal.App("vtr-ner-training")
vol = modal.Volume.from_name("vtr-model-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    volumes={"/root/artifacts": vol}
)
def train_on_modal(train_jsonl_content: str) -> bool:
    train_jsonl_path = Path("/tmp/ner_train.jsonl")
    train_jsonl_path.write_text(train_jsonl_content, encoding="utf-8")
    
    import sys
    sys.path.append("/root/src")
    from vtr_ai.train_ner import main
    
    # Run training and save directly to the mounted Volume
    output_dir = "/root/artifacts/ner-xlmr"
    main([
        "--train_jsonl", str(train_jsonl_path),
        "--model_name", "xlm-roberta-base",
        "--output_dir", output_dir,
        "--num_train_epochs", "5",
        "--batch_size", "8"
    ])
    
    # Commit changes to the volume
    vol.commit()
    return True

if __name__ == "__main__":
    train_file = Path(".cache/ner_train.jsonl")
    if not train_file.exists():
        print(f"Error: local training data not found at {train_file.resolve()}")
        exit(1)
        
    print("Reading local training data...")
    train_jsonl_content = train_file.read_text(encoding="utf-8")
    
    print("Connecting to Modal and starting remote GPU training...")
    with app.run():
        success = train_on_modal.remote(train_jsonl_content)
        
    if success:
        print("Training finished on Modal GPU! Downloading checkpoint files locally...")
        output_dir = Path("artifacts/ner-xlmr")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download files from the volume using the modal volume get CLI command (highly optimized)
        subprocess.run([
            "python", "-m", "modal", "volume", "get", "vtr-model-vol", "ner-xlmr", "artifacts"
        ], check=True)
        
        print(f"Model saved to {output_dir.resolve()} successfully!")
    else:
        print("Training failed.")
