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

## Sweeps

Run one seed/config per spawned process, with atomic JSON results and automatic resume:

```sh
python -m mirror.sweep sweeps/exp0.toml --dry-run
python -m mirror.sweep sweeps/exp0.toml --workers 16 --threads-per-job 2
```

The sample expands to 20 jobs (seeds 0-19, including the original seeds 0-2),
using the existing exp0 defaults. It writes to `sweep-output/exp0`, leaving the
committed `results/` untouched. Each experiment module must provide
`run_job(params: dict, seed: int) -> dict`; `meta` is reserved for the runner.

A sweep TOML contains `experiment` (module path), `out` (directory), `seeds`
(integer array), optional `[params]` for fixed parameters, and optional `[grid]`
for arrays whose Cartesian product is run for every seed. Fixed and grid keys
must be disjoint. Paths are relative to the working directory of the command.
`--threads-per-job` defaults to 1. `--workers` defaults to CPU count divided by
threads per job, with a minimum of 1. Keep workers times threads within your
machine's CPU/memory budget.

Each filename contains readable parameters, the seed and a short identity hash.
The full `meta.params_hash` includes the experiment module, parameters, seed and
thread count. Matching complete results are skipped; malformed or conflicting
results abort the whole sweep before starting jobs. Different parameters get
different filenames. A thread-count change conflicts at the same filename: use
a separate output directory to intentionally run with another thread count.
`--dry-run` lists done/pending without importing the experiment or writing files.

Failed jobs write `.error.json` with traceback and parameters; other jobs continue,
and the command exits nonzero if any job failed. Re-running retries failures.
A killed process can leave hidden `.tmp` files; these are never treated as results.
Use one sweep invocation per output directory at a time. The runner does not
coordinate multiple simultaneous invocations targeting the same directory.

Metadata records parameters, seed, hash, Git SHA/dirty status, torch/Python
versions, thread count, hostname and UTC timing. Git fields are null if Git
provenance is unavailable (for example, an installed copy without a checkout).
Resume intentionally does not invalidate results for Git or library-version
changes: use a new output directory when intentionally comparing code versions.
For fixed code/environment and threads, each worker seeds Python and torch for
each job; equality across different thread counts is not promised.
