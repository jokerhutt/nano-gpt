from pathlib import Path

from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]):
        return "".join(self.itos[i] for i in tokens)

    def save(self, directory: Path) -> None:
        (Path(directory) / "chars.txt").write_text("".join(self.chars), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "CharacterTokenizer":
        chars = list((Path(directory) / "chars.txt").read_text(encoding="utf-8"))
        return cls(chars)
