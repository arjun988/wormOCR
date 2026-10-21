"""Character vocabulary for CTC recognition.

Index 0 is the CTC blank. Every other index is one character from the charset.
The charset is stored in checkpoints so a later code change cannot silently
remap an older model.
"""

from __future__ import annotations


def default_charset() -> str:
    """Printable ASCII, including space. This is the English v1 vocabulary."""
    return " " + "".join(chr(code) for code in range(33, 127))


class Vocabulary:
    def __init__(self, charset: str | None = None) -> None:
        text = default_charset() if charset is None else charset
        if not text:
            raise ValueError("charset is empty")
        if len(set(text)) != len(text):
            raise ValueError("charset contains duplicate characters")
        if any(ord(char) < 32 for char in text):
            raise ValueError("charset cannot contain control characters")
        self.charset = text
        self.blank = 0
        self.itos: list[str] = ["", *text]
        self.stoi = {char: index + 1 for index, char in enumerate(text)}

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            character = exc.args[0]
            raise ValueError(
                f"character {character!r} is outside the wormOCR charset"
            ) from exc

    def decode(self, indices: list[int]) -> str:
        parts: list[str] = []
        for index in indices:
            if index == self.blank:
                continue
            if index < 0 or index >= len(self.itos):
                raise ValueError(f"class index {index} is outside the charset")
            parts.append(self.itos[index])
        return "".join(parts)

    def to_dict(self) -> dict[str, str]:
        return {"charset": self.charset}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        charset = payload.get("charset")
        if not isinstance(charset, str) or not charset:
            raise ValueError("checkpoint vocabulary is missing a charset")
        return cls(charset)
