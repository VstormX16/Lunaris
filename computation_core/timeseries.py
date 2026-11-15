"""Run-length helpers for boolean time series."""
import numpy as np


def runs(mask):
    """Return (starts, lengths) of consecutive True runs."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return edges[0::2], edges[1::2] - edges[0::2]


def longest_run_hours(mask, step_hours):
    _, lengths = runs(mask)
    return float(lengths.max() * step_hours) if lengths.size else 0.0


def longest_run_rows(mask):
    """Longest run of True per row of a (N, T) array, fully vectorized."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.arange(mask.shape[1])
    last_false = np.maximum.accumulate(np.where(mask, -1, idx), axis=1)
    return np.where(mask, idx - last_false, 0).max(axis=1)


def count_runs_rows(mask):
    """Number of True runs per row of a (N, T) array."""
    mask = np.asarray(mask, dtype=bool)
    starts = mask.copy()
    starts[:, 1:] &= ~mask[:, :-1]
    return starts.sum(axis=1)
