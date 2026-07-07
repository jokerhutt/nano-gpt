# ∴ Jokerhut / src/model/config.py


import torch

checkpoint_interval = 10000
sample_interval = 5000

batch_size = 128
block_size = 256
max_iters = 50000
eval_interval = 500
eval_iters = 200

learning_rate = 3e-4

n_embed = 512
n_head = 8
n_layer = 8
dropout = 0.2

model_name = "jgpt"

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
