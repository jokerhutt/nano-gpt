# ∴ Jokerhut / src/model/tokenizers/__init__.py

from pathlib import Path

from src.model import config
from .base import BaseTokenizer
from .character_tokenizer import CharacterTokenizer
from .sentence_piece_tokenizer import SentencePieceTokenizer


def build_tokenizer(text: str) -> BaseTokenizer:
    """Build the tokenizer selected by config.tokenizer_type."""

    if config.tokenizer_type == "character":
        return CharacterTokenizer(sorted(set(text)))

    if config.tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(str(_ensure_sentencepiece_model(text)))

    raise ValueError(f"Unknown tokenizer_type: {config.tokenizer_type!r}")


def load_tokenizer(directory, tokenizer_type: str) -> BaseTokenizer:
    """Reconstruct a saved tokenizer of the given type from a model/checkpoint directory."""

    if tokenizer_type == "character":
        return CharacterTokenizer.load(directory)

    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer.load(directory)

    raise ValueError(f"Unknown tokenizer_type: {tokenizer_type!r}")


def _ensure_sentencepiece_model(text: str) -> Path:
    """Train the sentencepiece model from the corpus if it doesn't exist yet."""

    import sentencepiece as spm

    model_dir = Path("spm")
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = model_dir / config.model_name
    model_path = prefix.with_suffix(".model")

    if model_path.exists():
        return model_path

    print(f"Training sentencepiece model -> {model_path}")

    corpus = model_dir / f"{config.model_name}_corpus.txt"
    corpus.write_text(text, encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=config.sp_vocab_size,
        # These are SentencePiece control symbols.  Do not add <EOS> as a
        # user-defined symbol: it must remain a special token so generation
        # can stop on it.
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        character_coverage=1.0,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
    )

    corpus.unlink(missing_ok=True)
    return model_path
