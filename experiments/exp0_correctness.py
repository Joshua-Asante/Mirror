"""Exp0: is a clean-calibrated correctness report blind to the model's actual state?

See PREREG_exp0.md. Usage:
    python -m experiments.exp0_correctness --seeds 0 1 2 --out results/exp0
"""

import argparse
import json
from pathlib import Path

import torch

from mirror import metrics, reports, task
from mirror.model import ModelConfig, Noise, Transformer
from mirror.train import TrainConfig, train

SCALES = (0.25, 0.5, 1.0, 2.0)


@torch.no_grad()
def run_model(model: Transformer, batch: task.Batch, noise: Noise | None = None, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    logits, states = model(batch.tokens, batch.answer_pos, noise=noise, generator=gen)
    correct = logits.argmax(-1) == batch.target
    return logits, states, correct


def run_seed(seed: int, n_fit: int, n_eval: int, train_cfg: TrainConfig) -> dict:
    task_cfg = task.TaskConfig()
    model_cfg = ModelConfig(vocab=task.VOCAB, seq_len=task_cfg.seq_len, n_out=10)
    model, history = train(task_cfg, model_cfg, train_cfg, seed)

    fit = task.sample(task_cfg, n_fit, torch.Generator().manual_seed(10_000 + seed))
    ev = task.sample(task_cfg, n_eval, torch.Generator().manual_seed(20_000 + seed))
    fit_logits, fit_states, fit_correct = run_model(model, fit)
    ev_logits, ev_states, ev_correct = run_model(model, ev)

    probes = {f"probe@{k}": reports.fit_probe(fit_states, fit_correct, k) for k in range(len(fit_states))}
    observer = reports.fit_observer(fit, fit_correct, task_cfg.seq_len, seed=30_000 + seed)
    obs_ev = reports.observer_prob(observer, ev)  # input-only: identical under any intervention

    def all_reports(logits, states):
        out = {name: p(states) for name, p in probes.items()}
        out["softmax"] = reports.softmax_confidence(logits)
        out["observer"] = obs_ev
        return out

    clean = all_reports(ev_logits, ev_states)
    result = {
        "seed": seed,
        "train_history": history,
        "clean_acc": ev_correct.float().mean().item(),
        "clean_acc_by_length": {
            int(L): ev_correct[ev.length == L].float().mean().item() for L in ev.length.unique()
        },
        "clean": {
            name: {"auroc": metrics.auroc(s, ev_correct), "ece": metrics.ece(s, ev_correct)}
            for name, s in clean.items()
        },
        "interventions": [],
    }

    for layer in range(model_cfg.n_layers + 1):
        for scale in SCALES:
            noise = Noise(layer=layer, scale=scale)
            logits, states, correct = run_model(model, ev, noise, seed=40_000 + seed)
            row = {"layer": layer, "scale": scale, "acc": correct.float().mean().item(), "reports": {}}
            for name, s in all_reports(logits, states).items():
                sens, n_flip = metrics.within_input_sensitivity(clean[name], ev_correct, s, correct)
                row["reports"][name] = {
                    "auroc": metrics.auroc(s, correct),
                    "ece": metrics.ece(s, correct),
                    "sensitivity": sens,
                    "n_flip": n_flip,
                }
            result["interventions"].append(row)
    return result


def run_job(params: dict, seed: int) -> dict:
    """Sweep adapter; keep all experiment logic in run_seed."""
    unknown = params.keys() - {"n_fit", "n_eval", "target_acc"}
    if unknown:
        raise ValueError(f"unknown exp0 parameters: {sorted(unknown)}")
    return run_seed(seed, params.get("n_fit", 20_000), params.get("n_eval", 10_000),
                    TrainConfig(target_acc=params.get("target_acc", 0.75)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-fit", type=int, default=20_000)
    ap.add_argument("--n-eval", type=int, default=10_000)
    ap.add_argument("--target-acc", type=float, default=0.75)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    ap.add_argument("--out", type=Path, default=Path("results/exp0"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        res = run_seed(seed, args.n_fit, args.n_eval, TrainConfig(target_acc=args.target_acc))
        path = args.out / f"seed{seed}.json"
        path.write_text(json.dumps(res, indent=1))
        print(f"seed {seed}: clean_acc={res['clean_acc']:.3f} -> {path}")


if __name__ == "__main__":
    main()
