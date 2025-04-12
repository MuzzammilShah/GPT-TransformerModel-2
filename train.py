from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

#=========================================================

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 85
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb)
        ))
        self.lm_head = nn.Linear(config.vocab_size, config.n_embd, bias=False)