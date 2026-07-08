from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_selected_review_subset(
    bootstrap_jsonl: str | Path,
    priority_json: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    priority_payload = json.loads(Path(priority_json).read_text(encoding="utf-8"))
    top_files = priority_payload.get("top_files", [])
    file_names = [str(item["file_name"]) for item in top_files]
    if limit is not None:
        file_names = file_names[:limit]
    selected = set(file_names)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for raw_line in Path(bootstrap_jsonl).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if str(record.get("file_name")) in selected:
            lines.append(json.dumps(record, ensure_ascii=False))

    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a smaller review JSONL subset from ranked review priority files.")
    parser.add_argument("--bootstrap_jsonl", required=True)
    parser.add_argument("--priority_json", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = export_selected_review_subset(
        args.bootstrap_jsonl,
        args.priority_json,
        args.output_path,
        limit=args.limit,
    )
    print(f"Selected review subset exported to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
