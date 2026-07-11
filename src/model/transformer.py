# ∴ Jokerhut / src/model/transformer.py


import torch

from src.model.block import Block
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
        
        self.blocks = torch.nn.Sequential(*[Block(self.n_embed, config.n_head) for _ in range(config.n_layer)])
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

    def generate_inference(
            self, 
            idx: torch.Tensor, 
            eos_token_id: int | None = None, 
            temperature: float = 1.0, 
            top_k: int | None = None, 
            top_p: int | None = None, 
            max_new_tokens: int = 512
    ) :
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # Keep only the context window, then sample from the
                    # distribution for its final position.
                    idx_cond = idx[:, -self.block_size:]
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :] / temperature

                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        if k <= 0:
                            raise ValueError("top_k must be greater than zero")
                        threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))

                    if top_p is not None:
                        if not 0 < top_p <= 1:
                            raise ValueError("top_p must be in the range (0, 1]")
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        remove = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        ) > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)

                    if eos_token_id is not None and torch.all(idx_next == eos_token_id):
                        break
        finally:
            self.train(was_training)

        return idx

    def generate(self, idx, max_new_tokens) :
        return self.generate_inference(idx, max_new_tokens=max_new_tokens)
