# mirror

An introspection lab for small transformers.

**Question:** when a model reports on itself ("I'll get this right", "I used
this cue"), is the report connected to the model's internal state, or is it a
plausible story learned from the inputs?

Large models can't answer this about themselves from the inside. Here we train
small models from scratch, where the internals are fully known. We give them a
self-report channel, then **intervene on their internals without changing the
input**. A report that reads internal state should move with the intervention;
one that only learned which inputs are hard cannot.

## Method

1. Tasks where internal ground truth is controllable (what the model computes,
   which cue it uses, whether it will be right).
2. Report mechanisms of increasing "selfness": an external observer that sees
   only the input, linear probes at each layer, the model's own output
   confidence, and later a self-report emitted by the model as tokens.
3. Faithfulness tests by intervention: noise, ablation and activation patching.
4. Every experiment is pre-registered in a `PREREG_*.md` committed before it runs.
   Results, including failed predictions, land in `results/`.

## Experiments

| ID | Question | Status |
|---|---|---|
| exp0 | Can a report be calibrated on clean data yet blind to the model's actual state? | [done](results/exp0/RESULTS.md): yes. Even a probe on the final residual is |

## Running

```
pip install -e '.[dev]'
pytest
python -m experiments.exp0_correctness --seeds 0 1 2 --out results/exp0
python -m experiments.summarize_exp0 results/exp0
```
