from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def export_review_packet(
    input_dir: str | Path,
    output_dir: str | Path,
    review_jsonl: str | Path,
    packet_dir: str | Path,
) -> Path:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    destination = Path(packet_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for raw_line in Path(review_jsonl).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        file_name = str(record["file_name"])
        stem = Path(file_name).stem
        shutil.copy2(input_root / file_name, destination / file_name)
        shutil.copy2(output_root / f"{stem}.json", destination / f"{stem}.json")
        records.append(record)

    (destination / "review_subset.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    manifest = {
        "files": [record["file_name"] for record in records],
        "count": len(records),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a compact review packet with txt/json/jsonl for selected files.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--review_jsonl", required=True)
    parser.add_argument("--packet_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = export_review_packet(args.input_dir, args.output_dir, args.review_jsonl, args.packet_dir)
    print(f"Review packet exported to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
