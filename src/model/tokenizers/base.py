from pathlib import Path

import torch


class BaseTokenizer:

    @property
    def vocab_size(self):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        """Persist whatever this tokenizer needs to be reconstructed later."""
        raise NotImplementedError

    def tokenize_data(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)
