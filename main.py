from pathlib import Path
from tokenizer import Tokenizer
import pandas as pd
import torch

from train import Train


def main():
    print("Hello from jgpt!")

    torch.manual_seed(1337)

    texts = []

    for file in Path("data").iterdir():

        if file.suffix == ".txt":

            texts.append(file.read_text(encoding="utf-8"))

        elif file.suffix == ".parquet":

            df = pd.read_parquet(file)

            texts.extend(df["text"].dropna().tolist())

    text = "\n\n".join(texts)

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
    training = Train(data, vocab_size, tokenizer)
    training.run_training()












if __name__ == "__main__":
    main()
