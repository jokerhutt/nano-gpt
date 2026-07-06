
from pathlib import Path
import json

from safetensors.torch import save_file, load_file
import torch


def save_model(model, chars, config, model_name) :

    model_dir = Path("models") / model_name
    model_dir.mkdir(parents = True, exist_ok = True)

    # save dem weights
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(cpu_state, model_dir / "model.safetensors")

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, index = 4)

    with open(model_dir / "chars.txt", "w", encoding ="utf-8") as f:
        f.write("".join(chars))


def save_checkpoint(model, optimizer, chars, config, model_name):

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

    with open(checkpoint_dir / "chars.txt", "w", encoding="utf-8") as f:
        f.write("".join(chars))

def load_model(model_class, model_name) :

    model_dir = Path("models") / model_name

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    with open(model_dir / "chars.txt", "r", encoding = "utf-8") as f :
        chars = list(f.read())

    model = model_class(config["vocab_size"])

    state_dict = load_file(model_dir / "model.safetensors")
    model.load_state_dict(state_dict)

    

