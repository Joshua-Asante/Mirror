# Handoff: sweep runner

**For:** Codex · **From:** Claude (coordinator) · **Date:** 2026-10-16
**Base branch:** `claude/exp0` (PR #1) until it merges, then `main`. **Your branch:** `codex/sweep-runner`.

## Why

Mirror trains small transformers and tests whether their self-reports track their internal state
(see `README.md`, `PREREG_exp0.md`, `results/exp0/RESULTS.md`). Each experiment job is one seed
of one config: a CPU-bound PyTorch run of about 2.5 minutes. The next experiments need hundreds of
jobs on a single 32-core Azure VM. Today, `experiments/exp0_correctness.py` runs seeds one after
another in a single process. Build the runner that fans jobs out across cores, and resumes safely.

## What to build

1. **`mirror/sweep.py`**, a CLI: `python -m mirror.sweep <sweep.toml> [--workers N] [--threads-per-job K] [--dry-run]`.
   - Expands the sweep file into jobs: the cartesian product of the grid, times the seeds.
   - Runs jobs in separate processes (`concurrent.futures.ProcessPoolExecutor`, `spawn` context).
     The default is `--workers = os.cpu_count() // K`. Each worker calls
     `torch.set_num_threads(K)` before importing or running the experiment.
   - Each job writes `<out>/<job_id>.json`, where `job_id` is a stable, readable name built from
     the parameters and seed (with a short hash suffix if needed). Write to a temp file in the
     same directory and `os.replace` it into place, so a crash never leaves a partial result.
   - **Resume is the default:** skip a job whose result file exists and has the same
     `params_hash`. If the hash differs, stop with an error rather than overwrite.
   - **Failure isolation:** a job that raises writes `<job_id>.error.json` (traceback and
     parameters); the sweep continues. The exit code is non-zero if any job failed.
   - Every result gets a `meta` block: params, seed, `params_hash`, git SHA plus a dirty flag,
     torch and Python versions, threads per job, hostname, start and end times, and wall seconds.
   - `--dry-run` prints the job list, with done/pending for each, and runs nothing.
   - Prints one progress line per finished job: done/total, job_id, wall seconds, failed count.
2. **Experiment entry point:** add `run_job(params: dict, seed: int) -> dict` to
   `experiments/exp0_correctness.py`. It maps `params` to the existing `run_seed` arguments
   (`n_fit`, `n_eval`, `target_acc`). The sweep file names its experiment by module path.
3. **`sweeps/exp0.toml`:** reproduces exp0 (seeds 0–2, current defaults) and adds seeds 3–19.
   Parse it with stdlib `tomllib`.
4. **Tests in `tests/test_sweep.py`:** use a fake, fast experiment module under `tests/`, never
   real training. Cover grid expansion; resume skips a finished job; a changed params hash
   errors; a failing job doesn't stop the others; a killed write leaves no result file; meta
   fields are present.

## Constraints

- **No new runtime dependencies.** Use the stdlib and torch only; no Ray, Dask, Hydra or
  Slurm. `pytest` stays a dev dependency.
- **Don't change experiment science:** leave the logic in `mirror/model.py`, `mirror/task.py`,
  `mirror/reports.py`, `mirror/metrics.py`, `mirror/train.py` and `run_seed` alone. Never edit
  `PREREG_*.md` or anything under `results/`. If you think science code needs changing, say so
  in your report instead.
- **Single machine only.** No SSH, Azure SDK, cloud credentials or remote execution code. I'll
  run the runner on the VM myself.
- **Determinism:** the same params, seed and threads per job must give an identical result,
  apart from `meta` timing fields. Don't promise equality across different thread counts;
  record the thread count instead.

## Done means

- `pytest` passes.
- `python -m mirror.sweep sweeps/exp0.toml --dry-run` lists 20 jobs.
- A real run of seeds 0–1 with `--threads-per-job 2` on your machine finishes, and both results
  match `run_seed` called directly with the same seed and 2 threads. Compare everything except
  `meta`, and show the comparison.
- Re-running the same command skips both jobs.
- A PR from `codex/sweep-runner`, with the commands you ran and their output in the description.

## Report back

Return the PR link, what you verified and how (commands and outputs), anything in this handoff
that was wrong or ambiguous, and anything you chose not to do. Don't widen the scope. If a
constraint blocks the task, stop and report it.
