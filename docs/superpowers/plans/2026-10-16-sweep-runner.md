# Sweep Runner Implementation Plan

> **For agentic workers:** Execute with superpowers:executing-plans. Preserve the integration owner and return boundary.

**Goal:** Deliver the single-machine sweep runner handoff as one verified PR.
**Architecture:** Parent expands TOML and checks resume files before spawn workers. Workers set torch threads before experiment import and publish JSON by same-directory temp file plus os.replace.
**Tech Stack:** Python >=3.11 stdlib, existing torch, pytest for tests.
**Spec:** User's 2026-10-16 sweep runner handoff in this chat.

## Global constraints

No new runtime dependencies, science edits, run_seed changes, PREREG edits or results/ edits. No cloud/remote execution. Base aaa875e7278f7b7cb119f37c9cd7dd6db2ee6cc3 (claude/exp0); branch codex/sweep-runner. Combined acceptance stays with Claude coordinator.

## Roadmap

One bounded outcome: runner, adapter, example sweep, tests and docs. Acceptance requires pytest, 20-job dry-run, real seeds 0 and 1 at two threads matching direct run_seed excluding meta, resume skips, and PR evidence. Future experiments, VM runs, scientific interpretation and merging are excluded.

## Current execution handoff

**Selected outcome:** A reviewable PR implementing all runner behavior and the exp0 integration.
**Prerequisites:** Production run_seed inspected; defaults 20000 fit, 10000 eval, target 0.75. Python 3.13 has torch 2.7.1+cpu and pytest. PR #1 remains open, so base claude/exp0.
**Ownership:** Codex executes and integrates this PR. Claude coordinator owns combined acceptance.
**Verification:** Fake module tests for Cartesian jobs, resume/conflicts, isolated exceptions, killed publication, metadata and thread-before-import. Adapter mapping test. Full pytest, 20-job dry-run, two real seeds/direct comparisons, resume, protected-path diff, independent review.
**Checkpoint:** Progress and findings in chat; actual command outputs and final revision in PR.
**Return boundary:** Return PR link and evidence after checks; stop for necessary science changes or constraint conflicts. Do not merge, run VM work or start another experiment.

## Behavioral contract

TOML keys: experiment (module), out, seeds, [params] fixed values and [grid] Cartesian values; overlapping keys error. Paths relative to invocation directory. CLI K defaults to 1, workers defaults max(1, cpu_count // K). Job identity includes module/params/seed; readable ID plus short hash. Resume hash additionally includes K so a thread change conflicts at the same path. Parent checks all existing results before executing. Invalid/corrupt/mismatched result errors without overwrite. Worker initializer calls torch.set_num_threads before importlib experiment import. Each job reseeds Python/torch, calls run_job, writes result with meta; exceptions write error JSON and return failure. Atomic temp files do not count as results. Dry-run creates nothing and imports no experiment. Same-output simultaneous sweep invocations are outside the single-runner contract.

## Steps

- [x] Write tests in tests/test_sweep.py and fast tests/sweep_fixture.py. Literal Cartesian expectations, subprocess CLI, mtime resume, tampered hash rejection, isolated raise and killed atomic writer establish behavior.
- [x] Run `py -3.13 -m pytest tests/test_sweep.py -q` and observe missing-runner failures.
- [x] Implement mirror/sweep.py: expand_jobs(config, threads_per_job), atomic_write(path, data), run_sweep(config, workers, threads_per_job, dry_run), main(). Use spawn ProcessPoolExecutor, metadata provenance, resume preflight and progress.
- [x] Run tests; add adapter mapping test that fails for missing run_job before implementation.
- [x] Add only run_job to experiments/exp0_correctness.py, mapping defaults to TrainConfig and rejecting unknown keys. Add sweeps/exp0.toml seeds 0 through 19, default grid and output outside results/. Document CLI and schema in README.
- [x] Full pytest, dry-run, real two-thread seeds 0 and 1, direct normalized JSON comparisons, resume. Record actual outputs and inspect protected paths unchanged.
- [x] Review full task diff, resolve defects, rerun affected checks. Commit, push and create PR against claude/exp0 if #1 is open, otherwise main; attach PR and return evidence to coordinator.

## Review record

Import/cache ordering family: confirmed independent-review finding that seeding before experiment import shifts the first job when an import consumes RNG. The fast fixture now consumes both Python and torch RNG at import. Fresh versus reused workers with reversed scheduling reproduced unequal samples before the fix. The common job boundary now imports after thread initialization and reseeds immediately before run_job. Resume skips do not execute the worker; retry/fresh-worker jobs pass through the same reseeding boundary. Exp0 train already seeds before model construction and sampling, so this generic runner fix does not change the scientific algorithm.

## Returned evidence

Implementation revision: 08b0f8d2e71498db8bd2acdd38744464630fd3cd. Independent review accepted the import-order repair. Normal `py -3.13 -m pytest -q`: 18 passed in 47.43 s (also 18 passed with unrelated global plugins disabled). Sample dry-run: 20 jobs; 0 done; 20 pending.

Full default exp0 jobs at workers=2, threads-per-job=2 finished: seed 1 wall 917.088 s, seed 0 wall 1120.157 s, failed=0. Independent direct run_seed calls with the same thread count finished: seed 0 wall 1040.172 s, seed 1 wall 914.715 s. After JSON normalization, both sweep payloads equal their direct payloads in all fields excluding meta. Repeating the exact two-job command reports skipped 2; pending 0; both file digests and modification times remain unchanged.

The handoff's approximate 2.5-minute runtime did not hold on this local Windows/PyTorch installation. No experiment defaults were reduced. K=1, invocation-relative paths and thread-aware hashes resolve unspecified configuration details, as documented in README. Scientific files, run_seed, pre-registration, results and dependency manifest are unchanged. No VM execution, new experiment, science adjustment or merge was performed.

PR: https://github.com/Joshua-Asante/Mirror/pull/3 (base claude/exp0 while PR #1 is open). Return this bounded implementation/evidence to Claude coordinator for combined acceptance. Full commands and actual outputs are in the PR description. Local acceptance artifacts remain ignored under .validation/.
