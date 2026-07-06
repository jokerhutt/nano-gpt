from pathlib import Path
import loader
from model.tokenizer import Tokenizer
import pandas as pd
import torch

from training.trainer import Trainer



def main():
    print("Hello from jgpt!")

    torch.manual_seed(1337)

    text = loader.load_data()

    print("Length of dataset in chars: ", len(text))
    print(text[:1000])

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print(''.join(chars))
    print(vocab_size)

    # tokenize and encode
    tokenizer = Tokenizer(chars = chars)
    data = tokenizer.tokenize_data(text)

    # training
    training = Trainer(data, vocab_size, tokenizer, chars)
    training.run_training()

if __name__ == "__main__":
    main()
