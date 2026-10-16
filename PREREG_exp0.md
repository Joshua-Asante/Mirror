# Exp0 pre-registration: calibrated is not the same as introspective

Written 2026-10-16, before any exp0 code was run.

## Question

A report about a model's own correctness can be well calibrated on ordinary
inputs without reading the model's internal state: it can learn which *inputs*
are hard. Does a correctness report trained only on clean data track the
model's actual state when that state is changed without changing the input?

## Setup

- **Task:** sum of L digits mod 10, L uniform in [2, 12]. Longer sequences are
  harder, so difficulty is partly visible from the input alone.
- **Model:** small causal transformer, trained until clean test accuracy is
  roughly 60-85% (partial competence, so correctness varies).
- **Reports**, all fit on clean held-out data to predict "model's answer is correct":
  - `observer`: a separate model that sees only the input.
  - `probe@0`: a linear probe on the embedding-layer residual (input information only).
  - `probe@k`: a linear probe on the residual after block k, for each k.
  - `softmax`: max answer probability (free, read from the output).
- **Interventions:** Gaussian noise of scale s (relative to the residual's RMS)
  added to residual state r_j, for each (j, s). The input is unchanged.

## Metrics

- Clean AUROC of each report against correctness.
- Intervened AUROC on the pooled intervened forward passes.
- **Within-input sensitivity:** across inputs whose correctness flips under an
  intervention, the fraction where the report moves in the right direction
  (strictly). An input-only report scores 0 by construction.

## Predictions

- **P1.** `observer` and `probe@0` reach clean AUROC well above 0.5, because
  length predicts difficulty.
- **P2.** On clean data, the best input-only report is within 0.10 AUROC of the
  final-layer probe. That is, calibration alone barely separates them.
- **P3.** Under intervention, `probe@final` and `softmax` keep a within-input
  sensitivity above 0.6; input-only reports stay at 0.
- **P4 (blind-spot sanity check).** Index residual states r_0 (embeddings) to
  r_n (after the last block); noise is added to r_j and `probe@k` reads r_k.
  For j > k the probe cannot see the intervention, so its within-input
  sensitivity is exactly 0.
  If it is not, the measurement is broken.
- **P5 (open; no confident prediction).** Does the final-layer probe stay
  calibrated (ECE) under intervention, or does it become overconfident? I
  lean towards overconfident, at maybe 60%.

## What would change my mind

If P2 fails, with clean calibration already separating internal from input-only
reports by a wide margin, the "calibrated-but-blind" story is weak for this
task, and exp1 needs a task where input-only difficulty is stronger.
If P4 fails, nothing else in the run is trusted until the bug is found.
