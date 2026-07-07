
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

from src.model.tokenizer import Tokenizer
from src.model.transformer import JGPT
from src.model import config


# anchor to the repo root so this works regardless of the current directory
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
MAX_NEW_TOKENS = 1000


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


def load_checkpoint(checkpoint_dir: Path) -> tuple[JGPT, Tokenizer]:

    with open(checkpoint_dir / "config.json") as f:
        cfg = json.load(f)

    with open(checkpoint_dir / "chars.txt", "r", encoding="utf-8") as f:
        chars = list(f.read())

    model = JGPT(vocab_size=cfg["vocab_size"])
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(state_dict)

    model = model.to(config.device)
    model.eval()

    return model, Tokenizer(chars)


def main():

    checkpoint_dir = select_model()
    model, tokenizer = load_checkpoint(checkpoint_dir)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    with torch.no_grad():
        tokens = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()

    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
