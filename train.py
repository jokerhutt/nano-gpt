

import torch
import math

import model_io
from tokenizer import Tokenizer

checkpoint_interval = 10000
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.3

model_name = "jgpt"

class Head (torch.nn.Module):

    def __init__(self, head_size) :
        super().__init__()

        self.key = torch.nn.Linear(n_embed, head_size, bias = False)
        self.query = torch.nn.Linear(n_embed, head_size, bias = False)
        self.value = torch.nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = torch.nn.Dropout(dropout)

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

class MultiHeadAttention(torch.nn.Module) :

    def __init__(self, num_heads, head_size) :
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x) :
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(torch.nn.Module) :

    def __init__(self, n_embd) :

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module) :

    def __init__(self, n_embd, n_head) :

        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(torch.nn.Module) :

    def __init__(self, vocab_size) :
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
        
        self.blocks = torch.nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embed)
        self.lm_head = torch.nn.Linear(n_embed, vocab_size)

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
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)

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

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = torch.nn.functional.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)

        return idx


class Train :

    def __init__(self, data: torch.Tensor, vocab_size: int, tokenizer: Tokenizer, chars: list[str]) :

        n = int(0.9*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.chars = chars

    @torch.no_grad
    def estimate_loss(self, model: BigramLanguageModel) :
        out = {}
        model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters) :
                X, Y = self._get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def run_training(self) :

        model = BigramLanguageModel(vocab_size = self.vocab_size) 
        print("Model initialised")
        m = model.to(device)
        print("Model device set")

        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        print("Optimiser set")

        for iter in range(max_iters):

            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = self.estimate_loss(m)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self._get_batch('train')

            # forward pass
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            if iter % checkpoint_interval == 0 and iter > 0 :
                config = {"vocab_size": self.vocab_size, "step": iter}
                    model_io.save_checkpoint(
                        model=m,
                        optimizer=optimizer,
                        chars=self.chars,
                        config=config,
                        model_name=model_name,
                    )

        context = torch.zeros((1, 1), dtype = torch.long, device = device)

        print("-----------")
        print("MODEL OUTPUT:")
        print("-----------")
        print(self.tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

        config = {"vocab_size": self.vocab_size}
        model_io.save_model(model = m, chars = self.chars, config = config, model_name = model_name) 


    # get batch_size amount of random contiguous items of length block_size each
    def _get_batch(self, split: str) :
        data = self.training_data if split == "train" else self.validation_data

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i: i + block_size] for i in ix])
        y = torch.stack([data[i+1: i + block_size + 1] for i in ix])

        x, y = x.to(device), y.to(device)
        return x,y







