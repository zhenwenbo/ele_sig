"""One complete shuffled training pass; keep every query row exactly once."""
import numpy as np

TRAIN_EPOCHS = 1
TRAIN_CONTEXT_ROWS = 1_024


def query_batches(n_rows, batch_size=128, seed=42):
    if n_rows < 1 or batch_size < 1:
        raise ValueError('Require positive row and batch sizes')
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    for step, start in enumerate(range(0, n_rows, batch_size), start=1):
        query = permutation[start:start + batch_size]
        yield step, 1, query, start + len(query) == n_rows


def should_validate(step, total_steps):
    return step > 0 and step == total_steps
