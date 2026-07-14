from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zip a validated Viettel subset output directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--zip_path", required=True)
    return parser


def package_output(output_dir: str | Path, zip_path: str | Path) -> Path:
    output_root = Path(output_dir)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for json_path in sorted(output_root.glob("*.json"), key=lambda path: path.name):
            archive.write(json_path, arcname=json_path.name)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    written = package_output(args.output_dir, args.zip_path)
    print(written.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
