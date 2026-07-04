from tokenizer import Tokenizer
import torch

from train import Train


def main():
    print("Hello from jgpt!")

    torch.manual_seed(1337)

    with open('input.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()

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
