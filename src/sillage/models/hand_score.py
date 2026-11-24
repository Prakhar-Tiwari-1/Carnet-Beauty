"""Hand-built scores for new clients (no purchase history).

v1 (brief, kept for the record; performs like random on validation):
    like_rate_s(p) = (likes_s(p) + m * step_rate_s) / (reviews_s(p) + m)
    score          = like_rate_s(p) + lam * concern_overlap

v2 "consultation score" (shipped):
    established products: log(1 + likes in the popularity window, 365 days)
    launches (no history): price fit = -|log(price / client's reference price)|
    guard     : exclude p if >= min_n reviewers with the client's skin type gave it a
                shrunk like rate more than `gap` below the step median for that skin type
    concerns  : stated by the client in the app -> eligibility filter (not a weight)

Why v2: almost every product has an 80-90% like rate, so the like rate measures
quality, not what a client chooses next; volume does. Skin type works as a
guard (it removes 6.6% of pairs at no accuracy cost on validation), not as a booster.
Launches and established products are never pooled: their scores are on different
scales and the app shows launches in a separate slot.

Why price fit for launches: over four backtests no review-history signal reliably
beat random on launches (brand prior, concern match and "Best for" skin tags each
lost to random in some period). Price fit was the only one never clearly worse
(mean +0.024 recall@10, worst -0.003). It is presented as a match, not a prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKIN_KEYS = ["dry", "normal", "combination", "oily", "all"]
EXCLUDED = -1e9


# ---- shared building blocks -------------------------------------------------

def like_rate_matrix(skin_table: pd.DataFrame, catalogue: pd.DataFrame, m: float) -> pd.DataFrame:
    """Shrunk like rate, index = product_id (catalogue order), columns = skin types."""
    products = catalogue["product_id"].to_numpy()
    step_of = catalogue.set_index("product_id")["step"].loc[products].to_numpy()
    out = {}
    for s in SKIN_KEYS:
        t = skin_table[skin_table["skin_type"] == s].set_index("product_id")
        n = t["n"].reindex(products).fillna(0).to_numpy(float)
        likes = t["likes"].reindex(products).fillna(0).to_numpy(float)
        step_rate = pd.DataFrame({"step": step_of, "n": n, "likes": likes}).groupby("step")[["likes", "n"]].sum()
        prior = (step_rate["likes"] / step_rate["n"]).reindex(step_of).to_numpy()
        out[s] = (likes + m * prior) / (n + m)
    return pd.DataFrame(out, index=products)


def count_matrix(skin_table: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    products = catalogue["product_id"].to_numpy()
    n = skin_table.pivot_table(index="product_id", columns="skin_type", values="n", fill_value=0)
    return n.reindex(index=products, columns=SKIN_KEYS, fill_value=0)


def skin_guard(rates: pd.DataFrame, counts: pd.DataFrame, steps: np.ndarray, gap: float, min_n: int) -> pd.DataFrame:
    """True where a product is clearly disliked by that skin type (P x skin types)."""
    median = rates.groupby(steps).transform("median")
    return (counts >= min_n) & (rates < median - gap)


def volume(product_feats: pd.DataFrame) -> np.ndarray:
    return np.log1p(product_feats["likes_recent"].to_numpy(float))


def price_fit(prices: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """U x P closeness of each price to each client's reference price (0 = exact)."""
    return -np.abs(np.log(prices[None, :] / reference[:, None]))


def concern_matrix(concern_lists, vocab: list[str]) -> np.ndarray:
    col = {c: i for i, c in enumerate(vocab)}
    m = np.zeros((len(concern_lists), len(vocab)), dtype=np.float32)
    for r, cs in enumerate(concern_lists):
        for c in cs:
            if c in col:
                m[r, col[c]] = 1.0
    return m


def concern_overlap_matrix(client_concerns, product_concerns, vocab) -> np.ndarray:
    """U x P share of each client's concerns that each product is tagged for."""
    cu = concern_matrix(client_concerns, vocab)
    cp = concern_matrix(product_concerns, vocab)
    n = cu.sum(axis=1, keepdims=True)
    return np.divide(cu @ cp.T, n, out=np.zeros((len(cu), len(cp)), np.float32), where=n > 0)


def _client_skin(ctx) -> np.ndarray:
    s = ctx.client_feats["skin_type"]
    return s.where(s.isin(SKIN_KEYS), "all").to_numpy()


# ---- scorers -----------------------------------------------------------------

def score_v1(ctx, m: float, lam: float) -> np.ndarray:
    rates = like_rate_matrix(ctx.skin_table, ctx.catalogue, m)
    base = rates[_client_skin(ctx)].to_numpy().T
    vocab = sorted({c for cs in ctx.catalogue["concerns"] for c in cs})
    overlap = concern_overlap_matrix(ctx.client_feats["concerns"], ctx.catalogue["concerns"], vocab)
    return base + lam * overlap


def score_v2(ctx, m: float, gap: float, min_n: int, seed: int = 0) -> np.ndarray:
    rates = like_rate_matrix(ctx.skin_table, ctx.catalogue, m)
    guard = skin_guard(rates, count_matrix(ctx.skin_table, ctx.catalogue), ctx.catalogue["step"].to_numpy(), gap, min_n)
    shape = (len(ctx.users), len(ctx.products))
    prices = ctx.catalogue["price_usd"].to_numpy(float)
    # Reference price = client's median past spend; catalogue median if unknown.
    ref = ctx.client_feats["median_price"].fillna(np.median(prices)).to_numpy(float)
    launch = ctx.product_feats["is_launch"].to_numpy()
    s = np.where(launch[None, :], price_fit(prices, ref), volume(ctx.product_feats)[None, :])
    s = s + np.random.default_rng(seed).random(shape) * 1e-6
    return np.where(guard[_client_skin(ctx)].to_numpy().T, EXCLUDED, s)
