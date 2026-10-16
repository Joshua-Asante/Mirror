"""Train the task model until clean accuracy reaches a target band."""

from dataclasses import dataclass

import torch

from mirror import task
from mirror.model import ModelConfig, Transformer, loss_fn


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    max_steps: int = 20_000
    eval_every: int = 250
    eval_n: int = 4_096
    target_acc: float = 0.75  # stop once clean accuracy reaches this


@torch.no_grad()
def accuracy(model: Transformer, batch: task.Batch) -> float:
    model.eval()
    logits, _ = model(batch.tokens, batch.answer_pos)
    return (logits.argmax(-1) == batch.target).float().mean().item()


def train(
    task_cfg: task.TaskConfig, model_cfg: ModelConfig, cfg: TrainConfig, seed: int
) -> tuple[Transformer, list[tuple[int, float]]]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_batch = task.sample(task_cfg, cfg.eval_n, torch.Generator().manual_seed(seed + 1))
    model = Transformer(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    history = []
    for step in range(1, cfg.max_steps + 1):
        model.train()
        batch = task.sample(task_cfg, cfg.batch_size, gen)
        logits, _ = model(batch.tokens, batch.answer_pos)
        loss = loss_fn(logits, batch.target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0:
            acc = accuracy(model, eval_batch)
            history.append((step, acc))
            if acc >= cfg.target_acc:
                break
    model.eval()
    return model, history
