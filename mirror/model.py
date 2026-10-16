"""Small pre-LN causal transformer that exposes its residual stream.

Residual state r_0 is the embedding output; r_k is the state after block k.
`forward` can add noise to any r_j and returns every r_k at the answer position,
so reports and interventions refer to the same indices.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    seq_len: int
    n_out: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 256


@dataclass(frozen=True)
class Noise:
    """Gaussian noise added to r_layer, scaled by that state's per-position RMS."""

    layer: int
    scale: float


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_out)
        mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        answer_pos: torch.Tensor,
        noise: Noise | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return answer logits (B, n_out) and residual states r_0..r_n at answer_pos."""
        t = tokens.shape[1]
        x = self.tok(tokens) + self.pos(torch.arange(t, device=tokens.device))
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        states = []
        for k in range(self.cfg.n_layers + 1):
            if k > 0:
                x = self.blocks[k - 1](x, self.mask[:t, :t])
            if noise is not None and noise.layer == k:
                x = _add_noise(x, noise.scale, generator)
            states.append(x[rows, answer_pos])
        logits = self.head(self.ln_f(x[rows, answer_pos]))
        return logits, states


def _add_noise(x: torch.Tensor, scale: float, generator: torch.Generator | None) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
    eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return x + scale * rms * eps


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)
