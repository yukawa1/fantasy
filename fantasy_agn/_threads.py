"""Thread counts for the parallel parts of a single fit."""
import contextlib
import os

# Worker processes of fit_many each fit one spectrum; inside them everything
# runs on one thread, otherwise processes x threads oversubscribe the cores.
SINGLE_THREADED = {"FANTASY_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def n_threads():
    """FANTASY_THREADS if set, else the number of cores (at most 8)."""
    env = os.environ.get("FANTASY_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(8, os.cpu_count() or 1))


@contextlib.contextmanager
def environment(values):
    """Temporarily set environment variables (inherited by new processes)."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
