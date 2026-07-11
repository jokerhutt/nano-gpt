import shutil
from pathlib import Path
from typing import Any

import sentencepiece as spm

from src.model.tokenizers.base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):

    MODEL_FILENAME = "tokenizer.model"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp: Any = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def bos_token_id(self) -> int | None:
        token_id = self.sp.bos_id()
        return token_id if token_id >= 0 else None

    @property
    def eos_token_id(self) -> int | None:
        token_id = self.sp.eos_id()
        return token_id if token_id >= 0 else None

    @property
    def unk_token_id(self) -> int | None:
        token_id = self.sp.unk_id()
        return token_id if token_id >= 0 else None

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    def save(self, directory: Path) -> None:
        dest = Path(directory) / self.MODEL_FILENAME
        if Path(self.model_path).resolve() != dest.resolve():
            shutil.copyfile(self.model_path, dest)

    @classmethod
    def load(cls, directory: Path) -> "SentencePieceTokenizer":
        return cls(str(Path(directory) / cls.MODEL_FILENAME))
