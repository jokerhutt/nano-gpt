# ∴ Jokerhut / src/training/trainer.py


import torch

from src.model.tokenizer import Tokenizer
from src.model.transformer import JGPT

import model_io
from src.model import config


class Trainer :

    def __init__(self, data: torch.Tensor, vocab_size: int, tokenizer: Tokenizer, chars: list[str]) :

        n = int(0.9*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        self.eval_iters = config.eval_iters
        self.eval_interval = config.eval_interval
        self.max_iters = config.max_iters
        self.sample_interval = config.sample_interval
        self.checkpoint_interval = config.checkpoint_interval
        self.device = config.device
        self.block_size = config.block_size
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.model_name = config.model_name

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.chars = chars

    @torch.no_grad
    def estimate_loss(self, model: JGPT) :
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

    def run_training(self, max_new_tokens = 500, max_new_sampling_tokens = 200) :

        model = JGPT(vocab_size = self.vocab_size) 
        print("Model initialised")
        m = model.to(self.device)
        print("Model device set")

        optimizer = torch.optim.AdamW(model.parameters(), lr = self.learning_rate)
        print("Optimiser set")

        for iter in range(self.max_iters):

            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss(m)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self._get_batch('train')

            # forward pass
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            # sample model output at sample_interval
            if iter % self.sample_interval == 0 and iter > 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=self.device)

                sample = self.tokenizer.decode(
                    m.generate(context, max_new_tokens=max_new_sampling_tokens)[0].tolist()
                )

                print("-----------")
                print(f"SAMPLE @ STEP {iter}")
                print("-----------")
                print(sample)

            # save checkpoint at checkpoint_interval
            if iter % self.checkpoint_interval == 0 and iter > 0 :
                config = {"vocab_size": self.vocab_size, "step": iter}
                model_io.save_checkpoint(
                    model=m,
                    optimizer=optimizer,
                    chars=self.chars,
                    config=config,
                    model_name=self.model_name,
                )

        context = torch.zeros((1, 1), dtype = torch.long, device = self.device)

        print("-----------")
        print("MODEL OUTPUT:")
        print("-----------")
        print(self.tokenizer.decode(m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))

        config = {"vocab_size": self.vocab_size}
        model_io.save_model(model = m, chars = self.chars, config = config, model_name = self.model_name) 


    # get batch_size amount of random contiguous items of length block_size each
    def _get_batch(self, split: str) :
        data = self.training_data if split == "train" else self.validation_data

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i+1: i + self.block_size + 1] for i in ix])

        x, y = x.to(self. device), y.to(self.device)
        return x,y







