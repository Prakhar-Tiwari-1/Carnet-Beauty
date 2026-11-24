"""Paired bootstrap over users: the same resampled users score every model."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: dict[str, np.ndarray],
    baseline: str,
    n_samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """values: model -> per-user metric, all aligned on the same users.

    Returns mean, CI, and the mean difference vs baseline with its CI.
    """
    n = len(next(iter(values.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_samples, n))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    base_boot = values[baseline][idx].mean(axis=1)
    out = {}
    for name, v in values.items():
        boot = v[idx].mean(axis=1)
        diff = boot - base_boot
        out[name] = {
            "mean": float(v.mean()),
            "ci": [float(np.percentile(boot, lo)), float(np.percentile(boot, hi))],
            "diff_vs_baseline": float(v.mean() - values[baseline].mean()),
            "diff_ci": [float(np.percentile(diff, lo)), float(np.percentile(diff, hi))],
        }
    return out
