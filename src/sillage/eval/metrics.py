"""Ranking metrics per user, computed from a score matrix.

recall@k = hits in top k / number of positives
ndcg@k   = DCG of top k / ideal DCG (binary relevance)
"""
from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best allowed items per row, best first. allowed: bool U x P."""
    s = np.where(allowed, scores, -np.inf)
    part = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def per_user_metrics(topk: np.ndarray, target: np.ndarray, k: int) -> dict[str, np.ndarray]:
    hits = np.take_along_axis(target, topk, axis=1)            # U x k bool
    n_pos = target.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.array([discounts[: min(n, k)].sum() for n in n_pos])
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_pos > 0, hits.sum(axis=1) / n_pos, np.nan)
        ndcg = np.where(n_pos > 0, dcg / ideal, np.nan)
    return {"recall": recall, "ndcg": ndcg}


def evaluate_scores(ctx, scores: np.ndarray, slice_name: str, k: int) -> dict:
    """Per-user metrics for users with at least one positive in the slice."""
    target = ctx.slice_target(slice_name)
    rows = target.any(axis=1)
    allowed = ~ctx.seen & ctx.candidates(slice_name)[None, :]
    tk = top_k(scores[rows], allowed[rows], k)
    m = per_user_metrics(tk, target[rows], k)
    return {"rows": np.flatnonzero(rows), "topk": tk, **m}


def coverage(topk: np.ndarray, n_items: int) -> float:
    """Share of the candidate catalogue recommended to at least one user."""
    return len(np.unique(topk)) / n_items


def bestseller_share(topk: np.ndarray, popularity: np.ndarray, candidates: np.ndarray) -> float:
    """Share of top-k slots filled by the top-decile most popular candidate products."""
    pop = np.where(candidates, popularity, -np.inf)
    n_top = max(1, int(round(0.1 * candidates.sum())))
    top_items = np.argsort(-pop, kind="stable")[:n_top]
    return float(np.isin(topk, top_items).mean())
