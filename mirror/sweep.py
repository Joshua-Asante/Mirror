"""Single-machine, resumable process sweeps: python -m mirror.sweep sweep.toml."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import tempfile
import time
import tomllib
import traceback


@dataclass(frozen=True)
class Job:
    experiment: str
    params: dict
    seed: int
    job_id: str
    params_hash: str


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def expand_jobs(config: dict, threads_per_job: int = 1) -> list[Job]:
    """Expand sorted grid axes, preserving each axis's and the seeds' order."""
    _positive(threads_per_job, "threads_per_job")
    unknown = config.keys() - {"experiment", "out", "seeds", "params", "grid"}
    if unknown:
        raise ValueError(f"unknown sweep keys: {sorted(unknown)}")
    experiment = config.get("experiment")
    if not isinstance(experiment, str) or not experiment or any(
        not part.isidentifier() for part in experiment.split(".")
    ):
        raise ValueError("experiment must be an importable module path")
    if not isinstance(config.get("out"), str) or not config["out"]:
        raise ValueError("out must be a nonempty path string")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds):
        raise ValueError("seeds must be a nonempty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    fixed, grid = config.get("params", {}), config.get("grid", {})
    if not isinstance(fixed, dict) or not isinstance(grid, dict):
        raise ValueError("params and grid must be tables")
    if fixed.keys() & grid.keys():
        raise ValueError("params and grid keys overlap")
    if any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("each grid axis must be a nonempty list")
    jobs = []
    identities = set()
    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = deepcopy({**fixed, **dict(zip(keys, values))})
        for seed in seeds:
            identity = {"experiment": experiment, "params": params, "seed": seed}
            try:
                identity_hash = _digest(identity)
                params_hash = _digest({**identity, "threads_per_job": threads_per_job})
            except (TypeError, ValueError) as exc:
                raise ValueError("parameters must be finite JSON values") from exc
            readable = "_".join(f"{k}-{_canonical(v)}" for k, v in sorted(params.items()))
            readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", readable).strip("-.")[:100]
            job_id = f"seed{seed}_{readable + '_' if readable else ''}{identity_hash[:12]}"
            if job_id in identities:
                raise ValueError(f"duplicate job identity: {job_id}")
            identities.add(job_id)
            jobs.append(Job(experiment, deepcopy(params), seed, job_id, params_hash))
    return jobs


def atomic_write(path: Path, data: dict) -> None:
    """Publish only a complete JSON file; interrupted temp files are not results."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finished(path: Path, job: Job) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        existing_hash = data["meta"]["params_hash"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid result {path}; refusing to overwrite") from exc
    if existing_hash != job.params_hash:
        raise ValueError(f"params hash mismatch for {path}; refusing to overwrite")
    return True


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {"git_sha": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        # Installed distributions may not contain a Git checkout.
        return {"git_sha": None, "git_dirty": None}


def _initialize_worker(threads_per_job: int) -> None:
    import torch
    torch.set_num_threads(threads_per_job)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_job(job: Job, out: str, threads_per_job: int, provenance: dict) -> tuple[bool, float]:
    import torch
    started = time.perf_counter()
    meta = {**provenance, "experiment": job.experiment, "params": deepcopy(job.params),
            "seed": job.seed, "params_hash": job.params_hash,
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "threads_per_job": threads_per_job, "hostname": socket.gethostname(),
            "start_time": _timestamp()}
    error = None
    try:
        experiment = importlib.import_module(job.experiment)
        # Imports may consume RNG state only on the first job in a worker.
        # Reset after import so cache state and scheduling cannot shift the stream.
        random.seed(job.seed)
        torch.manual_seed(job.seed)
        result = experiment.run_job(deepcopy(job.params), job.seed)
        if not isinstance(result, dict):
            raise TypeError("run_job must return a dict")
        if "meta" in result:
            raise ValueError("run_job result must not contain reserved key 'meta'")
        meta.update(end_time=_timestamp(), wall_seconds=time.perf_counter() - started)
        atomic_write(Path(out) / f"{job.job_id}.json", {**result, "meta": meta})
    except Exception:
        error = traceback.format_exc()
        meta.update(end_time=_timestamp(), wall_seconds=time.perf_counter() - started)
        atomic_write(Path(out) / f"{job.job_id}.error.json",
                     {"traceback": error, "params": job.params, "seed": job.seed, "meta": meta})
    return error is None, meta["wall_seconds"]


def run_sweep(config: dict, workers: int | None = None, threads_per_job: int = 1,
              dry_run: bool = False) -> int:
    jobs = expand_jobs(config, threads_per_job)
    workers = _positive(workers if workers is not None else
                        max(1, (os.cpu_count() or 1) // threads_per_job), "workers")
    out = Path(config["out"])
    # Preflight the whole sweep before any experiment or directory creation.
    pending = []
    states = []
    for job in jobs:
        done = _finished(out / f"{job.job_id}.json", job)
        states.append((job, done))
        if not done:
            pending.append(job)
    if dry_run:
        for job, done in states:
            print(f"{'done' if done else 'pending'} {job.job_id} "
                  f"params={_canonical(job.params)} seed={job.seed}")
        print(f"{len(jobs)} jobs; {len(jobs) - len(pending)} done; {len(pending)} pending")
        return 0
    skipped = len(jobs) - len(pending)
    print(f"{len(jobs)} jobs; skipped {skipped}; pending {len(pending)}", flush=True)
    if not pending:
        return 0
    provenance = _git_provenance()
    out.mkdir(parents=True, exist_ok=True)
    failed, finished = 0, skipped
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_worker, initargs=(threads_per_job,)) as pool:
        futures = {pool.submit(_run_job, j, str(out), threads_per_job, provenance): j for j in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                success, wall = future.result()
            except Exception:
                # A hard worker death breaks the pool; record unfinished futures too.
                success, wall = False, 0.0
                atomic_write(out / f"{job.job_id}.error.json",
                             {"traceback": traceback.format_exc(), "params": job.params,
                              "seed": job.seed, "params_hash": job.params_hash})
            finished += 1
            failed += not success
            print(f"{finished}/{len(jobs)} {job.job_id} wall={wall:.3f}s failed={failed}", flush=True)
    return int(failed > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        with args.sweep.open("rb") as stream:
            config = tomllib.load(stream)
        return run_sweep(config, args.workers, args.threads_per_job, args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
