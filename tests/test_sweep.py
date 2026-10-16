"""Runner contract tests; real spawn processes, no real training."""
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.sweep_fixture import interrupted_writer


def runner():
    return importlib.import_module("mirror.sweep")


def config(out, **updates):
    value = {"experiment": "tests.sweep_fixture", "out": str(out),
             "seeds": [0, 1], "params": {}, "grid": {"x": [10]}}
    value.update(updates)
    return value


def test_cartesian_expansion_stable_and_readable(tmp_path):
    # Catches omitted axes/seeds, unstable IDs, or missing fixed parameters.
    spec = config(tmp_path, seeds=[2, 3], params={"fixed": True},
                  grid={"x": [1, 2], "y": ["a", "b"]})
    jobs = runner().expand_jobs(spec, 2)
    assert [(j.params, j.seed) for j in jobs] == [
        ({"fixed": True, "x": x, "y": y}, seed)
        for x in [1, 2] for y in ["a", "b"] for seed in [2, 3]]
    again = runner().expand_jobs(dict(reversed(list(spec.items()))), 2)
    assert [j.job_id for j in jobs] == [j.job_id for j in again]
    assert len({j.job_id for j in jobs}) == 8
    assert all("seed" in j.job_id and "x-" in j.job_id for j in jobs)


def test_resume_metadata_threads_and_conflict(tmp_path, capsys):
    # Catches overwriting completed jobs, wrong K, late thread initialization,
    # absent provenance, and conflicts that start work before being rejected.
    spec = config(tmp_path)
    sweep = runner()
    assert sweep.run_sweep(spec, workers=2, threads_per_job=2) == 0
    paths = sorted(tmp_path.glob("*.json"))
    assert len(paths) == 2
    for path in paths:
        data = json.loads(path.read_text())
        assert data["threads"] == data["import_threads"] == 2
        assert data["pid"] != os.getpid()
        meta = data["meta"]
        assert set(meta) >= {"params", "seed", "params_hash", "git_sha", "git_dirty",
                             "torch_version", "python_version", "threads_per_job",
                             "hostname", "start_time", "end_time", "wall_seconds"}
        assert meta["params"] == {"x": 10}
        assert meta["threads_per_job"] == 2
        assert len(meta["params_hash"]) == 64
        assert meta["git_sha"] and isinstance(meta["git_dirty"], bool)
        assert meta["hostname"] and meta["torch_version"] and meta["python_version"]
        assert meta["start_time"] <= meta["end_time"]
        assert meta["wall_seconds"] >= 0
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    assert sweep.run_sweep(spec, workers=2, threads_per_job=2) == 0
    assert "skipped 2" in capsys.readouterr().out
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    with pytest.raises(ValueError, match="hash"):
        sweep.run_sweep(spec, workers=1, threads_per_job=1)
    data = json.loads(paths[0].read_text())
    data["meta"]["params_hash"] = "wrong"
    paths[0].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        sweep.run_sweep(spec, workers=1, threads_per_job=2)
    assert json.loads(paths[0].read_text())["meta"]["params_hash"] == "wrong"


def test_failing_job_does_not_stop_other_jobs(tmp_path, capsys):
    # Catches exceptions aborting the executor or success-shaped error outputs.
    sweep = runner()
    spec = config(tmp_path, seeds=[0, 1, 2], params={"fail_seed": 1})
    assert sweep.run_sweep(spec, workers=2, threads_per_job=1) == 1
    errors = list(tmp_path.glob("*.error.json"))
    results = [p for p in tmp_path.glob("*.json") if not p.name.endswith(".error.json")]
    assert sorted(json.loads(p.read_text())["value"] for p in results) == [10, 12]
    assert len(errors) == 1
    error = json.loads(errors[0].read_text())
    assert "RuntimeError: intentional failure for seed 1" in error["traceback"]
    assert error["meta"]["params"] == {"fail_seed": 1, "x": 10}
    assert error["meta"]["seed"] == 1
    assert "3/3" in capsys.readouterr().out


def test_killed_write_leaves_no_result(tmp_path):
    # Catches direct-to-final publication, using a real forcibly killed process.
    runner()  # Missing implementation must fail in the parent, not hang a child.
    result = tmp_path / "result.json"
    ready = tmp_path / "ready"
    process = multiprocessing.get_context("spawn").Process(
        target=interrupted_writer, args=(str(result), str(ready)))
    process.start()
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and process.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "writer did not reach a partial temporary write"
        assert not result.exists()
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()
        assert not result.exists()
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def test_dry_run_cli_creates_nothing_or_imports_experiment(tmp_path):
    # Nonexistent experiment proves dry-run doesn't import/run it.
    spec = tmp_path / "sweep.toml"
    out = tmp_path / "absent"
    spec.write_text(f'experiment = "does.not.exist"\nout = {json.dumps(str(out))}\n'
                    'seeds = [0, 1]\n[grid]\nx = [1, 2]\n')
    result = subprocess.run([sys.executable, "-m", "mirror.sweep", str(spec), "--dry-run"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert sum(line.startswith("pending ") for line in result.stdout.splitlines()) == 4
    assert not out.exists()


@pytest.mark.parametrize("updates", [
    {"seeds": []}, {"seeds": [True]}, {"grid": {"x": []}},
    {"params": {"x": 1}}, {"seeds": [0, 0]},
])
def test_reject_invalid_sweeps(tmp_path, updates):
    with pytest.raises(ValueError):
        runner().expand_jobs(config(tmp_path, **updates), 1)


def test_exp0_adapter_maps_defaults_and_overrides(monkeypatch):
    # Expensive training replaced only at its entry boundary; inspect actual mapping.
    from experiments import exp0_correctness as exp
    seen = []

    def capture(seed, n_fit, n_eval, train_cfg):
        seen.append((seed, n_fit, n_eval, train_cfg.target_acc))
        return {"seed": seed}

    monkeypatch.setattr(exp, "run_seed", capture)
    assert exp.run_job({}, 7) == {"seed": 7}
    assert exp.run_job({"n_fit": 30, "n_eval": 20, "target_acc": 0.6}, 8) == {"seed": 8}
    assert seen == [(7, 20000, 10000, 0.75), (8, 30, 20, 0.6)]
    with pytest.raises(ValueError, match="unknown"):
        exp.run_job({"typo": 3}, 0)


def test_resume_preflight_rejects_corrupt_file_before_any_job(tmp_path):
    # A conflicting later result must not allow an earlier pending job to run.
    sweep = runner()
    spec = config(tmp_path)
    jobs = sweep.expand_jobs(spec, 1)
    bad = tmp_path / f"{jobs[1].job_id}.json"
    bad.write_text('{"meta":')
    with pytest.raises(ValueError, match="invalid result"):
        sweep.run_sweep(spec, workers=1)
    assert not (tmp_path / f"{jobs[0].job_id}.json").exists()
    assert bad.read_text() == '{"meta":'


def test_reused_worker_reseeds_each_job(tmp_path):
    # Catches RNG state leaking from earlier jobs in a reused worker.
    sweep = runner()
    one, two = tmp_path / "one", tmp_path / "two"
    assert sweep.run_sweep(config(one), workers=1) == 0
    assert sweep.run_sweep(config(two, seeds=[1, 0]), workers=2) == 0
    first = {json.loads(p.read_text())["meta"]["seed"]: json.loads(p.read_text())["random"]
             for p in one.glob("*.json")}
    second = {json.loads(p.read_text())["meta"]["seed"]: json.loads(p.read_text())["random"]
              for p in two.glob("*.json")}
    assert first == second
