

import torch
import math

from tokenizer import Tokenizer




class BigramLanguageModel(torch.nn.Module) :

    def __init__(self, vocab_size) :
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None) :

        # BTC = (rows, cols, possible_next_tokens)
        logits = self.token_embedding_table(idx) # (B, T, C)

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
            logits, loss = self(idx)

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
        self.max_iters = 3000
        self.eval_interval = 300
        self.learning_rate = 1e-2
        self.device = "cpu"
        self.eval_iters = 200

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

        model = BigramLanguageModel(vocab_size = self.vocab_size)
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







