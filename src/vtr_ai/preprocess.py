from __future__ import annotations

from .schemas import Document


PUNCT_NORMALIZATION = {
    "\u00a0": " ",
    "\t": " ",
    "\r": "\n",
    "–": "-",
    "—": "-",
    "−": "-",
}


def normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    norm_to_raw: list[int] = []
    previous_space = False
    for raw_index, char in enumerate(text):
        replacement = PUNCT_NORMALIZATION.get(char, char)
        for repl_char in replacement:
            if repl_char.isspace():
                if previous_space:
                    continue
                repl_char = " "
                previous_space = True
            else:
                previous_space = False
            normalized_chars.append(repl_char)
            norm_to_raw.append(raw_index)
    normalized = "".join(normalized_chars)
    if not normalized.strip():
        return "", []
    start_shift = 0
    while start_shift < len(normalized_chars) and normalized_chars[start_shift] == " ":
        start_shift += 1
    end_shift = len(normalized_chars)
    while end_shift > start_shift and normalized_chars[end_shift - 1] == " ":
        end_shift -= 1
    normalized = normalized[start_shift:end_shift]
    norm_to_raw = norm_to_raw[start_shift:end_shift]
    return normalized, norm_to_raw


def build_document(text: str) -> Document:
    normalized_text, norm_to_raw = normalize_with_mapping(text)
    return Document(text=text, normalized_text=normalized_text, norm_to_raw=norm_to_raw)
