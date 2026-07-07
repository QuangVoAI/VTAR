from __future__ import annotations

from pathlib import Path
import json

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

