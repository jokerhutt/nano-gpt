
import torch

from model.head import Head
from src.model import config


class MultiHeadAttention(torch.nn.Module) :

    def __init__(self, num_heads, head_size) :
        super().__init__()

        self.n_embed = config.n_embed
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, self.n_embed)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x) :
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
