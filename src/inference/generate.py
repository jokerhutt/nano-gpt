# ∴ Jokerhut / src/inference/generate.py


from pathlib import Path
import json
import sys

# allow running this file directly (e.g. `python src/inference/generate.py`)
# by putting the repo root on the path so `src` is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import questionary
import torch
from safetensors.torch import load_file

from src.model.tokenizers import load_tokenizer
from src.model.tokenizers.base import BaseTokenizer
from src.model.transformer import JGPT
from src.model import config


# anchor to the repo root so this works regardless of the current directory
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
MAX_NEW_TOKENS = 1000
DEFAULT_PROMPT = "The "


def select_model() -> Path:

    if not CHECKPOINTS_DIR.is_dir():
        sys.exit(f"No checkpoints directory found at {CHECKPOINTS_DIR}. Train a model first.")

    models = sorted(d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir())
    if not models:
        sys.exit(f"No model checkpoints found in {CHECKPOINTS_DIR}. Train a model first.")

    model_name = questionary.select(
        "Select model:",
        choices=models,
    ).ask()

    return CHECKPOINTS_DIR / model_name


def load_checkpoint(checkpoint_dir: Path) -> tuple[JGPT, BaseTokenizer]:

    with open(checkpoint_dir / "config.json") as f:
        cfg = json.load(f)

    # default to character for older checkpoints saved before tokenizer_type existed
    tokenizer = load_tokenizer(checkpoint_dir, cfg.get("tokenizer_type", "character"))

    model = JGPT(vocab_size=cfg["vocab_size"])
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(state_dict)

    model = model.to(config.device)
    model.eval()

    return model, tokenizer


def main():

    checkpoint_dir = select_model()
    model, tokenizer = load_checkpoint(checkpoint_dir)

    # Use real text as a seed.  Token 0 is SentencePiece's <unk> control token
    # and is not a useful prompt for generation.
    prompt_tokens = tokenizer.encode(DEFAULT_PROMPT)
    if not prompt_tokens:
        raise RuntimeError(f"The generation prompt could not be tokenized: {DEFAULT_PROMPT!r}")
    if tokenizer.unk_token_id in prompt_tokens:
        raise RuntimeError(
            f"The generation prompt contains an unknown token: {DEFAULT_PROMPT!r}"
        )
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=config.device)

    tokens = model.generate_inference(
        context,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
    )[0].tolist()

    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
