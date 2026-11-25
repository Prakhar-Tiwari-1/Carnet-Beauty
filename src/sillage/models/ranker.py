"""LightGBM LambdaRank over candidates from simpler models.

Candidates per client: top-k by popularity, top-k by item-kNN, plus every launch
(so the launch slice is rankable). Trained on validation-period labels with
features computed before the validation cutoff; applied to test with features
recomputed before the test cutoff.

Feature groups (the ablation units):
  product : popularity, shrunk rating, review count, 90-day reviews, product age,
            sampled-review share, launch flag, brand prior, step, number of concern tags
  price   : log price, price tier within step, log(price / client's median past price)
  client  : history length, days since last review, mean rating given, skin type
  match   : like rate for her skin type, concern overlap, brand affinity, skin-guard flag
  collab  : item-kNN similarity sum and max
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from sillage.models import hand_score
from sillage.models.hand_score import SKIN_KEYS

GROUPS = {
    "product": ["pop_log", "rating_shrunk", "log_n_reviews", "reviews_90d_log", "days_since_first",
                "incentivised_share", "is_launch", "brand_prior", "step_code", "n_concern_tags"],
    "price": ["log_price", "price_tier", "log_price_ratio"],
    "client": ["log_hist_len", "days_since_last", "mean_rating_given", "skin_code"],
    "match": ["like_rate_skin", "concern_overlap", "brand_affinity", "skin_guard"],
    "collab": ["knn_sum", "knn_max"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
STEP_CODES = {"cleanser": 0, "treatment": 1, "moisturiser": 2}

FEATURE_LABELS = {
    "pop_log": "how often it was liked in the past year",
    "rating_shrunk": "its average rating",
    "log_n_reviews": "its number of reviews",
    "reviews_90d_log": "recent review activity",
    "days_since_first": "how long it has been on sale",
    "incentivised_share": "share of free-sample reviews",
    "is_launch": "being new",
    "brand_prior": "how well the brand's similar products did",
    "step_code": "routine step",
    "n_concern_tags": "number of concern tags",
    "log_price": "price",
    "price_tier": "price tier",
    "log_price_ratio": "price versus her usual spend",
    "log_hist_len": "length of her history",
    "days_since_last": "time since her last purchase",
    "mean_rating_given": "how generously she rates",
    "skin_code": "her skin type",
    "like_rate_skin": "like rate among her skin type",
    "concern_overlap": "match with her concerns",
    "brand_affinity": "she liked this brand before",
    "skin_guard": "skin-type caution flag",
    "knn_sum": "similarity to products she liked",
    "knn_max": "closeness to one product she liked",
}


def candidates(ctx, pop: np.ndarray, knn: np.ndarray, k: int) -> np.ndarray:
    """bool U x P: union of top-k popularity, top-k kNN and all launches, minus seen."""
    mask = np.zeros(ctx.seen.shape, bool)
    for s in (pop, knn):
        s = np.where(ctx.seen | ctx.is_launch[None, :], -np.inf, s)
        top = np.argpartition(-s, kth=k, axis=1)[:, :k]
        np.put_along_axis(mask, top, True, axis=1)
    mask |= ctx.is_launch[None, :]
    return mask & ~ctx.seen


def build_features(ctx, knn_sum: np.ndarray, knn_max: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                   m: float, gap: float, min_n: int, extra: dict | None = None) -> pd.DataFrame:
    """extra: name -> U x P score matrix from other models, added as feature 'cf_<name>'."""
    pf, cf, cat = ctx.product_feats, ctx.client_feats, ctx.catalogue
    price = cat["price_usd"].to_numpy(float)
    skin_idx = cf["skin_type"].map({s: i for i, s in enumerate(SKIN_KEYS)}).fillna(len(SKIN_KEYS) - 1).astype(int).to_numpy()

    rates = hand_score.like_rate_matrix(ctx.skin_table, cat, m)
    guard = hand_score.skin_guard(rates, hand_score.count_matrix(ctx.skin_table, cat), cat["step"].to_numpy(), gap, min_n)
    rates_np, guard_np = rates[SKIN_KEYS].to_numpy().T, guard[SKIN_KEYS].to_numpy().T   # skin x P

    vocab = sorted({c for cs in cat["concerns"] for c in cs})
    cu = hand_score.concern_matrix(cf["concerns"], vocab)
    cp = hand_score.concern_matrix(cat["concerns"], vocab)
    n_cu = cu.sum(1)
    overlap = np.divide((cu[rows] * cp[cols]).sum(1), n_cu[rows], out=np.zeros(len(rows), np.float32), where=n_cu[rows] > 0)

    brands = pd.Categorical(cat["brand_name"])
    bcode = brands.codes
    nb = len(brands.categories)
    liked_pairs = {
        u * nb + brands.categories.get_loc(b)
        for u, bl in enumerate(cf["liked_brands"]) for b in bl if b in brands.categories
    }
    brand_aff = np.isin(rows * nb + bcode[cols], np.fromiter(liked_pairs, int, len(liked_pairs)))

    median = cf["median_price"].fillna(np.median(price)).to_numpy(float)
    f = {
        "pop_log": np.log1p(pf["likes_recent"].to_numpy(float))[cols],
        "rating_shrunk": pf["rating_shrunk"].to_numpy(float)[cols],
        "log_n_reviews": pf["log_n_reviews"].to_numpy(float)[cols],
        "reviews_90d_log": np.log1p(pf["reviews_90d"].to_numpy(float))[cols],
        "days_since_first": pf["days_since_first"].to_numpy(float)[cols],
        "incentivised_share": pf["incentivised_share"].fillna(0).to_numpy(float)[cols],
        "is_launch": pf["is_launch"].to_numpy(float)[cols],
        "brand_prior": pf["brand_prior"].to_numpy(float)[cols],
        "step_code": cat["step"].map(STEP_CODES).to_numpy(float)[cols],
        "n_concern_tags": cat["concerns"].str.len().to_numpy(float)[cols],
        "log_price": np.log(price)[cols],
        "price_tier": pf["price_tier"].to_numpy(float)[cols],
        "log_price_ratio": np.log(price[cols] / median[rows]),
        "log_hist_len": np.log1p(cf["hist_len"].to_numpy(float))[rows],
        "days_since_last": cf["days_since_last"].to_numpy(float)[rows],
        "mean_rating_given": cf["mean_rating_given"].to_numpy(float)[rows],
        "skin_code": skin_idx.astype(float)[rows],
        "like_rate_skin": rates_np[skin_idx[rows], cols],
        "concern_overlap": overlap,
        "brand_affinity": brand_aff.astype(float),
        "skin_guard": guard_np[skin_idx[rows], cols].astype(float),
        "knn_sum": knn_sum[rows, cols],
        "knn_max": knn_max[rows, cols],
    }
    for name, mat in (extra or {}).items():
        f[f"cf_{name}"] = mat[rows, cols]
    order = ALL_FEATURES + [f"cf_{n}" for n in (extra or {})]
    return pd.DataFrame(f)[order].astype(np.float32)


def pairs(mask: np.ndarray, users: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Candidate (row, col) pairs sorted by user, optionally restricted to some user rows."""
    if users is not None:
        sub = np.zeros(mask.shape[0], bool)
        sub[users] = True
        mask = mask & sub[:, None]
    r, c = np.nonzero(mask)
    return r, c


def params(num_leaves: int, min_child_samples: int, seed: int, objective: str = "lambdarank",
           device: str = "cpu") -> dict:
    p = dict(
        objective=objective, learning_rate=0.05,
        num_leaves=num_leaves, min_child_samples=min_child_samples, feature_fraction=0.9,
        bagging_fraction=0.8, bagging_freq=1, seed=seed, deterministic=True, num_threads=4,
        verbose=-1, device_type=device,
    )
    if objective == "lambdarank":
        p.update(metric="ndcg", eval_at=[10], lambdarank_truncation_level=20)
    else:
        p.update(metric="binary_logloss")
    if device == "cpu":
        p["force_row_wise"] = True
    return p


def _dataset(X, y, rows, features):
    _, counts = np.unique(rows, return_counts=True)
    return lgb.Dataset(X[features], label=y, group=counts, free_raw_data=False)


def train(data_train: tuple, p: dict, features: list[str], data_stop: tuple | None = None,
          fixed_rounds: int | None = None, num_boost_round: int = 2000):
    """data_* = (X, y, rows). Groups with no positive are dropped (no ranking signal).

    Early stopping uses a later period (data_stop), never users from the training period.
    """
    def prep(X, y, rows):
        has_pos = pd.Series(y).groupby(rows).transform("max").to_numpy() > 0
        return _dataset(X[has_pos], y[has_pos], rows[has_pos], features)

    dtrain = prep(*data_train)
    if fixed_rounds:
        return lgb.train(p, dtrain, num_boost_round=fixed_rounds)
    return lgb.train(p, dtrain, num_boost_round=num_boost_round, valid_sets=[prep(*data_stop)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def to_matrix(model, X: pd.DataFrame, rows: np.ndarray, cols: np.ndarray, shape, features: list[str]) -> np.ndarray:
    s = np.full(shape, -np.inf, np.float32)
    s[rows, cols] = model.predict(X[features], num_iteration=model.best_iteration or None)
    return s
