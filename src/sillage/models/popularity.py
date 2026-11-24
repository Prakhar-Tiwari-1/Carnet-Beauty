"""Baseline: what has been liked most in the past year (window in config.yaml).

The 365-day window was chosen by best worst case over four backtests inside
period A; no window won consistently.

Counted within each routine step. Ranking a merged list across steps by the
same count is identical to merging the per-step popularity lists.
Ties (e.g. every launch has zero history) are broken at random with a fixed
seed, so popularity honestly scores "no information" where it has none.
"""
from __future__ import annotations

import numpy as np


def popularity_vector(product_feats) -> np.ndarray:
    return product_feats["likes_recent"].to_numpy(float)


def tie_break(shape, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(shape) * 1e-6


def score(ctx, seed: int = 0) -> np.ndarray:
    shape = (len(ctx.users), len(ctx.products))
    return popularity_vector(ctx.product_feats)[None, :] + tie_break(shape, seed)


def random_scores(ctx, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((len(ctx.users), len(ctx.products)))
