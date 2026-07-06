
import torch

from model.block import Block
from src.model import config


class JGPT(torch.nn.Module) :

    def __init__(self, vocab_size) :
        super().__init__()


        self.vocab_size = vocab_size

        self.block_size = config.block_size
        self.n_embed = config.n_embed
        self.device = config.device


        self.token_embedding_table = torch.nn.Embedding(vocab_size, self.n_embed)
        self.position_embedding_table = torch.nn.Embedding(self.block_size, self.n_embed)
        
        self.blocks = torch.nn.Sequential(*[Block(self.n_embed, self.n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(self.n_embed)
        self.lm_head = torch.nn.Linear(self.n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding) :
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets=None) :

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = self.device)) # (T, C)

        x = tok_emb + pos_emb
        x = self.blocks(x) # apply head of self attention
        x = self.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else :
            # reshape logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # calculate loss
            loss = torch.nn.functional.cross_entropy(logits, targets) # Loss Fn

        return logits, loss

    def generate(self, idx, max_new_tokens) :

        for _ in range(max_new_tokens) :

            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = torch.nn.functional.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
