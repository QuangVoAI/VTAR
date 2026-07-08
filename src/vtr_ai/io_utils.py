from __future__ import annotations

from pathlib import Path
import json
import shutil
import zipfile

from .schemas import Entity


def read_txt_files(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).glob("*.txt"), key=lambda path: path.name)


def write_entities(output_path: str | Path, entities: list[Entity], pretty: bool) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [entity.to_dict() for entity in entities]
    if pretty:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def reset_directory(path: str | Path) -> Path:
    directory = Path(path)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def zip_directory(source_dir: str | Path, zip_path: str | Path) -> Path:
    source = Path(source_dir)
    archive = Path(zip_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for file_path in sorted(source.rglob("*")):
            if file_path.is_file():
                handle.write(file_path, file_path.relative_to(source.parent))
    return archive
