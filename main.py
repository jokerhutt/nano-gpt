# ∴ Jokerhut / main.py

from pathlib import Path
import loader
from src.model.tokenizers import build_tokenizer
import pandas as pd
import torch

from src.training.trainer import Trainer



def main():
    print("Hello from jgpt!")

    torch.manual_seed(1337)

    text = loader.load_data()

    print("Length of dataset in chars: ", len(text))
    print(text[:1000])

    # tokenize and encode using the tokenizer selected in config
    tokenizer = build_tokenizer(text)
    vocab_size = tokenizer.vocab_size

    print("vocab size: ", vocab_size)

    data = tokenizer.tokenize_data(text)

    # training
    training = Trainer(data, vocab_size, tokenizer)
    training.run_training()

if __name__ == "__main__":
    main()
