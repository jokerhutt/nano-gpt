
import torch


class Tokenizer :

    def __init__(self, chars: list[str]):
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join(self.itos[i] for i in tokens)

    def tokenize_data(self, text: str) -> torch.Tensor :
        data = torch.tensor(self.encode(text), dtype = torch.long)
        print(data[:1000])
        return data




