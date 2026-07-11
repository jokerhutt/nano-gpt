from pathlib import Path

import torch


class BaseTokenizer:

    @property
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token id, if this tokenizer has one."""
        return None

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token id, if this tokenizer has one."""
        return None

    @property
    def unk_token_id(self) -> int | None:
        """Return the unknown-token id, if this tokenizer has one."""
        return None

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
