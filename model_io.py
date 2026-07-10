# ∴ Jokerhut / model_io.py


from pathlib import Path
import json

from safetensors.torch import save_file, load_file
import torch

from src.model.tokenizers import load_tokenizer


def save_model(model, tokenizer, config, model_name) :

    model_dir = Path("models") / model_name
    model_dir.mkdir(parents = True, exist_ok = True)

    # save dem weights
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(cpu_state, model_dir / "model.safetensors")

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent = 4)

    tokenizer.save(model_dir)


def save_checkpoint(model, optimizer, tokenizer, config, model_name):

    checkpoint_dir = Path("checkpoints") / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(cpu_state, checkpoint_dir / "model.safetensors")

    torch.save(
        optimizer.state_dict(),
        checkpoint_dir / "optimizer.pt"
    )

    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    tokenizer.save(checkpoint_dir)

def load_model(model_class, model_name) :

    model_dir = Path("models") / model_name

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    tokenizer = load_tokenizer(model_dir, config.get("tokenizer_type", "character"))

    model = model_class(config["vocab_size"])

    state_dict = load_file(model_dir / "model.safetensors")
    model.load_state_dict(state_dict)

    return model, tokenizer, config
