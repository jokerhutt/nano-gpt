

import torch
import math

from tokenizer import Tokenizer



class Head (torch.nn.Module):

    def __init__(self, head_size, n_embed, block_size) :
        super().__init__()

        self.key = torch.nn.Linear(n_embed, head_size, bias = False)
        self.query = torch.nn.Linear(n_embed, head_size, bias = False)
        self.value = torch.nn.Linear(n_embed, head_size, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x) :

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1)

        # weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module) :

    def __init__(self, num_heads, head_size, n_embed, block_size) :
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size, n_embed, block_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(num_heads, n_embed)

    def forward(self, x) :
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(torch.nn.Module) :

    def __init__(self, n_embed) :

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module) :

    def __init__(self, n_embed, n_head, block_size) :

        super().__init__()

        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)

        return x


class BigramLanguageModel(torch.nn.Module) :

    def __init__(self, vocab_size, n_embed, block_size, device) :
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.device = device
        self.block_size = block_size

        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
        
        self.blocks = torch.nn.Sequential(
            Block(n_embed = n_embed, n_head = 4, block_size = block_size),
            Block(n_embed = n_embed, n_head = 4, block_size = block_size),
            Block(n_embed = n_embed, n_head = 4, block_size = block_size),
        )

        self.lm_head = torch.nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None) :

        B, T = idx.shape

        # BTC = (rows, cols, possible_next_tokens)
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = self.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.sa_heads(x) # apply head of self attention
        x = self.ffwd(x)
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


class Train :

    def __init__(self, data: torch.Tensor, vocab_size: int, tokenizer: Tokenizer) :

        n = int(0.9*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        self.batch_size = 32
        self.block_size = 8
        self.max_iters = 5000
        self.eval_interval = 300
        self.learning_rate = 1e-3
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.eval_iters = 200
        self.n_embed = 32

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    @torch.no_grad
    def estimate_loss(self, model: BigramLanguageModel) :
        out = {}
        model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters) :
                X, Y = self._get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def run_training(self) :

        model = BigramLanguageModel(vocab_size = self.vocab_size, n_embed= self.n_embed, block_size = self.block_size, device = self.device)
        print("Model initialised")
        m = model.to(self.device)
        print("Model device set")

        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)
        print("Optimiser set")

        for iter in range(self.max_iters):

            if iter % self.eval_interval == 0:
                losses = self.estimate_loss(model)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self._get_batch('train')

            # forward pass
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

        context = torch.zeros((1, 1), dtype = torch.long, device = self.device)

        print("-----------")
        print("MODEL OUTPUT:")
        print("-----------")
        print(self.tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    # get batch_size amount of random contiguous items of length block_size each
    def _get_batch(self, split: str) :
        data = self.training_data if split == "train" else self.validation_data

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i+1: i + self.block_size + 1] for i in ix])

        x, y = x.to(self.device), y.to(self.device)
        return x,y







