import torch

from mirror import metrics, task
from mirror.model import ModelConfig, Noise, Transformer


def test_task_targets_and_layout():
    cfg = task.TaskConfig()
    b = task.sample(cfg, 500, torch.Generator().manual_seed(0))
    for i in range(500):
        L = int(b.length[i])
        digits = b.tokens[i, 1 : L + 1]
        assert b.tokens[i, 0] == task.BOS
        assert (digits < task.DIGITS).all()
        assert b.tokens[i, b.answer_pos[i]] == task.EQ
        assert (b.tokens[i, L + 2 :] == task.PAD).all()
        assert int(digits.sum()) % 10 == int(b.target[i])


def test_noise_leaves_earlier_states_identical():
    cfg = task.TaskConfig()
    model = Transformer(ModelConfig(vocab=task.VOCAB, seq_len=cfg.seq_len, n_out=10)).eval()
    b = task.sample(cfg, 64, torch.Generator().manual_seed(1))
    with torch.no_grad():
        _, clean = model(b.tokens, b.answer_pos)
        _, noisy = model(b.tokens, b.answer_pos, noise=Noise(layer=1, scale=1.0),
                         generator=torch.Generator().manual_seed(2))
    assert torch.equal(clean[0], noisy[0])
    assert not torch.allclose(clean[1], noisy[1])
    assert not torch.allclose(clean[2], noisy[2])


def test_auroc():
    y = torch.tensor([0, 0, 1, 1])
    assert metrics.auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert metrics.auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert metrics.auroc(torch.ones(4), y) == 0.5


def test_sensitivity_zero_for_unchanged_report():
    s = torch.rand(100)
    clean = torch.rand(100) > 0.5
    flipped = ~clean
    frac, n = metrics.within_input_sensitivity(s, clean, s, flipped)
    assert n == 100 and frac == 0.0


def test_sensitivity_one_for_tracking_report():
    clean = torch.tensor([True, False])
    after = torch.tensor([False, True])
    frac, _ = metrics.within_input_sensitivity(
        torch.tensor([0.9, 0.1]), clean, torch.tensor([0.2, 0.7]), after
    )
    assert frac == 1.0
