"""Digit-sum task: predict (sum of L digits) mod 10.

Token layout: [BOS, d_1, ..., d_L, EQ, PAD...]. The answer is read at the EQ position.
Length varies, so difficulty is partly visible from the input alone.
"""

from dataclasses import dataclass

import torch

DIGITS = 10
BOS = 10
EQ = 11
PAD = 12
VOCAB = 13


@dataclass(frozen=True)
class TaskConfig:
    min_len: int = 2
    max_len: int = 12

    @property
    def seq_len(self) -> int:
        return self.max_len + 2


@dataclass
class Batch:
    tokens: torch.Tensor  # (B, T) long
    answer_pos: torch.Tensor  # (B,) long, index of EQ
    target: torch.Tensor  # (B,) long in [0, 10)
    length: torch.Tensor  # (B,) long, number of digits


def sample(cfg: TaskConfig, n: int, generator: torch.Generator) -> Batch:
    lengths = torch.randint(cfg.min_len, cfg.max_len + 1, (n,), generator=generator)
    digits = torch.randint(0, DIGITS, (n, cfg.max_len), generator=generator)
    pos = torch.arange(cfg.max_len).unsqueeze(0)
    digits = torch.where(pos < lengths.unsqueeze(1), digits, torch.zeros_like(digits))
    target = digits.sum(dim=1) % DIGITS

    tokens = torch.full((n, cfg.seq_len), PAD, dtype=torch.long)
    tokens[:, 0] = BOS
    body = torch.where(pos < lengths.unsqueeze(1), digits, torch.full_like(digits, PAD))
    tokens[:, 1 : cfg.max_len + 1] = body
    answer_pos = lengths + 1
    tokens[torch.arange(n), answer_pos] = EQ
    return Batch(tokens=tokens, answer_pos=answer_pos, target=target, length=lengths)
