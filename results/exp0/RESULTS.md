# Exp0 results

Pre-registration: [PREREG_exp0.md](../../PREREG_exp0.md). Three seeds (0, 1, 2), 4 CPU threads,
`python -m experiments.exp0_correctness --seeds <s>`. Full tables: [tables.md](tables.md); raw: `seed*.json`.

## Headline

**A linear probe on the model's final residual stream is as well calibrated on clean data as the
model's own softmax (AUROC 0.930 vs 0.932), yet under intervention it does not track the model's
actual state.** When noise flips whether the model is right, the probe moves in the correct
direction about 50-60% of the time, roughly a coin flip. The softmax moves correctly 54-94% of the
time, rising with intervention strength. Having access to internals did not make the probe
introspective. It learned something closer to "which inputs are hard", which it reads from the
residual instead of from the tokens.

## Predictions

| | Prediction | Outcome |
|---|---|---|
| P1 | Input-only reports beat chance on clean data | **Held.** observer 0.928, probe@0 (length only) 0.849 |
| P2 | Best input-only report within 0.10 AUROC of final-layer probe on clean data | **Held.** 0.928 vs 0.930 |
| P3 | Final probe and softmax keep within-input sensitivity > 0.6 | **Failed for the probe** (0.50-0.61, mostly ≈0.5). Softmax held in 10 of 12 cells; it fell to 0.54 and 0.57 for the two smallest noise levels at r_2 |
| P4 | Probe@k has sensitivity exactly 0 for noise at r_j, j > k | **Held** in every cell (sanity check passes) |
| P5 | (open) Final probe becomes overconfident under intervention | **Held:** ECE rises to 0.15-0.75 as accuracy falls. Softmax ECE stays lower (0.03-0.41) |

## What I take from it

- Clean calibration did not distinguish the three kinds of report: the observer, which has no
  internal access, the probe, which has full access, and the softmax, which reads the decision
  itself. Only the intervention did. This is the core claim of the lab, shown on a toy.
- The surprise is P3. I expected a probe on the final residual to inherit the model's state. It
  didn't, because nothing in clean training rewards reading the *decision* over reading
  *difficulty features*. On clean data they are interchangeable; off-distribution they are not.

## Caveats

- One task, one architecture, three seeds, and only Gaussian noise as the intervention. Noise also
  pushes the residual off-distribution, so part of the probe's failure may be distribution shift
  rather than "reads the wrong feature". Exp1 should use on-distribution interventions such as
  activation patching between inputs.
- **Deviation:** seed 0 reached 85.8% clean accuracy, above the pre-registered 60-85% band; the
  stopping check runs every 250 steps, and accuracy jumped from 63% to 85% in one interval.
  Seeds 1 and 2 (78.4%, 77.6%) are inside the band, and the conclusions are the same with seed 0
  excluded.
- "Sensitivity" counts strict moves in the right direction. A report that is perturbed at random
  scores about 0.5, which is the probe's level.
