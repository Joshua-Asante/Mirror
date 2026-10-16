"""Fast spawn-safe experiment and interrupted-write helpers for sweep tests."""
import os
import random
import time
from pathlib import Path

import torch

# A fixture that fails to import if worker initialization was too late.
IMPORT_THREADS = torch.get_num_threads()
# Exercise import side effects before the job-specific seed is applied.
_IMPORT_DRAW = (random.random(), torch.rand(1))


def run_job(params, seed):
    if params.get("fail_seed") == seed:
        raise RuntimeError(f"intentional failure for seed {seed}")
    return {"value": params.get("x", 1) + seed,
            "threads": torch.get_num_threads(), "import_threads": IMPORT_THREADS,
            "pid": os.getpid(), "random": [random.random(), *torch.rand(3).tolist()]}


def interrupted_writer(destination, ready):
    from mirror import sweep
    original_dump = sweep.json.dump

    def pause_mid_write(data, stream, **kwargs):
        stream.write('{"partial":')
        stream.flush()
        Path(ready).touch()
        time.sleep(120)
        original_dump(data, stream, **kwargs)

    sweep.json.dump = pause_mid_write
    sweep.atomic_write(Path(destination), {"complete": True})
