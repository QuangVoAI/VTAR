from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re


@dataclass(slots=True)
class AbbreviationExpander:
    mapping: dict[str, str]

    @classmethod
    def from_path(cls, path: str | Path) -> "AbbreviationExpander":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({key.lower(): value for key, value in data.items()})

    def expand_text_with_mapping(self, text: str) -> tuple[str, list[int]]:
        expanded_chars: list[str] = []
        expanded_to_raw: list[int] = []
        index = 0
        pattern = re.compile(r"\b[\w/]+\b", re.UNICODE)
        for match in pattern.finditer(text):
            if match.start() > index:
                prefix = text[index:match.start()]
                expanded_chars.extend(prefix)
                expanded_to_raw.extend(range(index, match.start()))
            token = match.group(0)
            replacement = self.mapping.get(token.lower(), token)
            expanded_chars.extend(replacement)
            if replacement == token:
                expanded_to_raw.extend(range(match.start(), match.end()))
            else:
                token_length = max(match.end() - match.start(), 1)
                replacement_length = max(len(replacement), 1)
                for repl_index in range(replacement_length):
                    projected = match.start() + min(
                        token_length - 1,
                        int(repl_index * token_length / replacement_length),
                    )
                    expanded_to_raw.append(projected)
            index = match.end()
        if index < len(text):
            suffix = text[index:]
            expanded_chars.extend(suffix)
            expanded_to_raw.extend(range(index, len(text)))
        return "".join(expanded_chars), expanded_to_raw
