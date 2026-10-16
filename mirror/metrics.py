"""Metrics comparing a report (a score per example) with whether the model was correct."""

import torch


def auroc(score: torch.Tensor, label: torch.Tensor) -> float:
    """Area under the ROC curve via average ranks (ties count half). NaN if one class."""
    label = label.bool()
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(score.double())
    return ((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty_like(x)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def ece(prob: torch.Tensor, label: torch.Tensor, n_bins: int = 10) -> float:
    """Expected calibration error of a probability that the model is correct."""
    label = label.float()
    bins = torch.clamp((prob * n_bins).long(), max=n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.float().mean().item() * abs(prob[sel].mean().item() - label[sel].mean().item())
    return total


def within_input_sensitivity(
    clean_score: torch.Tensor,
    clean_correct: torch.Tensor,
    int_score: torch.Tensor,
    int_correct: torch.Tensor,
) -> tuple[float, int]:
    """Among inputs whose correctness flips under intervention, the fraction where the
    report moves strictly in the same direction. Returns (fraction, number of flips)."""
    flip = clean_correct != int_correct
    n = int(flip.sum())
    if n == 0:
        return float("nan"), 0
    want = int_correct[flip].float() - clean_correct[flip].float()
    got = torch.sign(int_score[flip] - clean_score[flip])
    return (got == want).float().mean().item(), n


def directional_sensitivity(
    clean_score: torch.Tensor,
    clean_correct: torch.Tensor,
    int_score: torch.Tensor,
    int_correct: torch.Tensor,
) -> dict:
    """Sensitivity split by flip direction (post-hoc, added after review).

    `down`: fraction of correct->wrong flips where the report strictly falls.
    `up`: fraction of wrong->correct flips where the report strictly rises.
    `balanced`: their mean. A report that always falls under intervention scores 0.5,
    as does one perturbed symmetrically at random; an unchanged report scores 0.
    """
    delta = int_score - clean_score
    to_wrong = clean_correct & ~int_correct
    to_right = ~clean_correct & int_correct
    down = (delta[to_wrong] < 0).float().mean().item() if to_wrong.any() else float("nan")
    up = (delta[to_right] > 0).float().mean().item() if to_right.any() else float("nan")
    return {
        "down": down,
        "up": up,
        "balanced": (down + up) / 2,
        "n_down": int(to_wrong.sum()),
        "n_up": int(to_right.sum()),
    }


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rank correlation; 0.0 when either input is constant (e.g. an input-only report)."""
    rx, ry = _average_ranks(x.double()), _average_ranks(y.double())
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = (rx.pow(2).sum() * ry.pow(2).sum()).sqrt()
    return 0.0 if denom == 0 else (rx * ry).sum().div(denom).item()
