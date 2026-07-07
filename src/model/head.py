# ∴ Jokerhut / src/model/head.py

import torch
from src.model import config

class Head (torch.nn.Module):

    def __init__(self, head_size) :
        super().__init__()

        self.n_embed = config.n_embed
        self.block_size = config.block_size

        self.key = torch.nn.Linear(self.n_embed, head_size, bias = False)
        self.query = torch.nn.Linear(self.n_embed, head_size, bias = False)
        self.value = torch.nn.Linear(self.n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x) :

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out
