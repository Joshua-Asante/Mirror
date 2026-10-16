# Exp0 results (revised after independent review)

Pre-registration: [PREREG_exp0.md](../../PREREG_exp0.md).
Independent review: [PR #1 comment](https://github.com/Joshua-Asante/Mirror/pull/1#issuecomment-6224001382).
Seeds 0–2, `torch 2.14.1+cpu`, 2 threads per seed, deterministic algorithms on:
`python -m experiments.exp0_correctness --seeds <s> --threads 2`.
Full tables: [tables.md](tables.md) (mean [min, max] over seeds); raw data: `seed*.json`.
The first version of this write-up and its numbers are in commit `aaa875e`.

## Headline

**Clean calibration does not tell a report that tracks the model's state apart from one that
doesn't. Probes fit on clean data, linear on any layer or on the logits, do not track the
model's state when its internals are perturbed.**

- On clean data, the final-layer probe, the softmax and the input-only observer are
  indistinguishable: AUROC 0.92, 0.93 and 0.93.
- Under noise, the change in each probe has essentially no rank correlation with the change in
  the model's log-probability of the right answer: |ρ| ≤ 0.10 in every cell.
- The softmax does correlate (ρ 0.20–0.47 under mild or late noise). That is weak evidence,
  because it is partly true by construction: whenever the model is right, its softmax confidence
  *is* the probability of the right answer.

So exp0 shows the negative half cleanly: calibrated is not the same as tracking. It does **not**
show any report introspecting in a non-trivial sense.

## Pre-registered predictions, scored on the registered metric

| | Prediction | Outcome |
|---|---|---|
| P1 | Input-only reports beat chance on clean data | **Held.** Observer AUROC 0.93; probe@0, which sees only the length, 0.87 |
| P2 | Best input-only report within 0.10 AUROC of the final-layer probe on clean data | **Held.** 0.93 vs 0.92 |
| P3 | Final probe and softmax keep within-input sensitivity > 0.6; input-only stay at 0 | **Failed for the probe:** 0.48–0.61, above 0.6 in 1 of 12 cells. Softmax 0.52–0.94, above 0.6 in 10 of 12 cells (not at r_2 with scale 0.25 or 0.5). Input-only exactly 0. **But the registered metric is confounded; see below** |
| P4 | probe@k is unchanged by noise added after layer k | **Held, by construction.** States are recorded before any later noise is added. This checks the indexing, not the metric |
| P5 | (open) Final probe becomes overconfident under intervention | **Partly supported.** probe@2's ECE rises from 0.03 to 0.04–0.62, against the softmax's 0.04–0.46. ECE alone doesn't show the direction; the reviewer's separate re-run found the probe's mean prediction above actual accuracy under r_0/r_1 noise |

### Why the registered sensitivity metric misleads

It counts how often a report moves the right way when an intervention flips correctness. Under
noise at r_0 or r_1, 79–97% of flips are correct→wrong (flip composition table in
[tables.md](tables.md)). So a report that simply drops whenever the model is perturbed scores
highly: a report that always decreases would score 0.79–0.97. There is no single "chance"
level. The first version of this file called ≈0.5 "a coin flip", and that was wrong.

## Post-hoc analyses (added after review, not pre-registered)

1. **Direction-balanced sensitivity** is the mean of P(report falls | correct→wrong) and
   P(report rises | wrong→correct). A report that always falls, or one perturbed symmetrically
   at random, scores 0.5.
   - Probes: 0.48–0.62.
   - Softmax: 0.52–0.73.
   - On wrong→correct flips under r_0 or r_1 noise, the softmax rises only 0.45–0.55 of the time
     (per-seed range 0.27–0.74). There, even the softmax mostly registers "something was
     perturbed".
2. **Rank correlation between Δreport and Δ log p(target)** over all inputs. This is the measure
   that separates the reports:
   - softmax: ρ 0.20–0.47 under mild noise (r_0 at 0.25; r_1 at 0.25 and 0.5; r_2 at every scale);
   - every probe: |ρ| ≤ 0.10 in seed means (≤ 0.15 in any single seed).
   - Under heavy early noise (accuracy ≤ 0.2), everything falls to ρ ≈ 0–0.12.
3. **Linear probe on the logits.** It has strictly more decision information than the softmax,
   yet its clean AUROC is only 0.72, and under noise its ρ is ≤ 0.09. A linear readout cannot
   express "how peaked is the distribution", so this failure says more about the probe's form
   than about access to internals.

## What I take from it

- **Holds:** a report can be calibrated on clean data while blind to the model's actual state.
  Every clean-fit probe here is such a report, including probes with full access to the
  internals.
- **Withdrawn:** "the probe learned which inputs are hard" was never tested. Within a single
  length, probe@2's clean AUROC is 0.76–0.94 (reviewer's analysis), so it reads more than length.
- **Withdrawn:** "the softmax tracks the state 54–94% while the probe is a coin flip". That
  contrast was mostly produced by the metric.
- **For exp1:** pre-register a direction-balanced metric and ρ against the model's own internal
  quantity. Include a report that is *not* a function of the output distribution and could
  still track the state, such as a nonlinear probe trained with interventions in the fit set. And
  use on-distribution interventions such as activation patching, so distribution shift isn't a
  confound.

## Deviations and choices made after registration

- The noise scales (0.25–2.0), 3 seeds, `n_fit` = 20 000, `n_eval` = 10 000 and the 0.75
  stopping target were chosen by me and are not fixed in the pre-registration.
- In the first run, seed 0 reached 85.8% clean accuracy, outside the 60–85% band. The revised
  run, with 2 threads and deterministic algorithms, gives 78.2%, 78.4% and 77.6%, all in band.
  Results depend on the thread count, which is now recorded in each `seed*.json`.
- Each intervention cell now draws its own noise. The first run reused one noise tensor
  rescaled across all cells.
- Everything in "Post-hoc analyses" was added after seeing results and the review.
