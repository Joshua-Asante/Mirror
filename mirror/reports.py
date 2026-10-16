"""Reports that predict whether the task model's answer is correct.

All reports are fit on clean data only. Each maps a forward pass (tokens plus the
model's residual states and logits) to a probability of being correct.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mirror import task
from mirror.model import ModelConfig, Transformer


@dataclass
class LinearProbe:
    """Logistic regression on one residual state, standardised with clean statistics."""

    layer: int
    mean: torch.Tensor
    std: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor

    def __call__(self, states: list[torch.Tensor]) -> torch.Tensor:
        x = (states[self.layer] - self.mean) / self.std
        return torch.sigmoid(x @ self.w + self.b)


def fit_probe(states: list[torch.Tensor], correct: torch.Tensor, layer: int, l2: float = 1e-3) -> LinearProbe:
    x = states[layer].detach()
    mean, std = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / std
    y = correct.float()
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return LinearProbe(layer, mean, std, w.detach(), b.detach())


def softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(-1).max(-1).values


def fit_observer(
    batch: task.Batch, correct: torch.Tensor, seq_len: int, seed: int, steps: int = 3_000
) -> Transformer:
    """A separate transformer that sees only the input tokens and predicts correctness."""
    torch.manual_seed(seed)
    obs = Transformer(ModelConfig(vocab=task.VOCAB, seq_len=seq_len, n_out=1))
    opt = torch.optim.AdamW(obs.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    y = correct.float()
    for _ in range(steps):
        idx = torch.randint(0, len(y), (256,), generator=gen)
        logit, _ = obs(batch.tokens[idx], batch.answer_pos[idx])
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    obs.eval()
    return obs


@torch.no_grad()
def observer_prob(obs: Transformer, batch: task.Batch) -> torch.Tensor:
    logit, _ = obs(batch.tokens, batch.answer_pos)
    return torch.sigmoid(logit.squeeze(-1))
