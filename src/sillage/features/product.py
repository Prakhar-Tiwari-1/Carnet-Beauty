"""Product features computed only from reviews before the cutoff.

Snapshot fields from product_info (rating, reviews, loves_count) are never used:
they are March-2023 values and contain the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sillage.config import load_config
from sillage.data.splits import history

PRIOR_STRENGTH = 20  # pseudo-reviews pulling a product's mean rating toward its step mean


def product_features(reviews: pd.DataFrame, catalogue: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cfg = load_config()
    h = history(reviews, cutoff)
    h = h[h["product_id"].isin(catalogue["product_id"])]
    clean = h[~h["incentivised"]]
    days = (cutoff - h["submission_time"]).dt.days

    g = h.groupby("product_id")
    agg = pd.DataFrame({
        "n_reviews": g.size(),
        "rating_sum": g["rating"].sum(),
        "incentivised_share": g["incentivised"].mean(),
        "first_review": g["submission_time"].min(),
        "reviews_90d": h[days <= 90].groupby("product_id").size(),
    })
    is_like = clean["rating"] >= cfg["positive_min_rating"]
    recent = (cutoff - clean["submission_time"]).dt.days <= cfg["popularity"]["window_days"]
    agg["likes_recent"] = clean[is_like & recent].groupby("product_id").size()
    agg["likes"] = clean[is_like].groupby("product_id").size()

    f = catalogue[["product_id", "step", "price_usd"]].merge(agg, on="product_id", how="left")
    for c in ["n_reviews", "rating_sum", "reviews_90d", "likes_recent", "likes"]:
        f[c] = f[c].fillna(0)

    step_mean = f.groupby("step")["rating_sum"].transform("sum") / f.groupby("step")["n_reviews"].transform("sum")
    f["rating_shrunk"] = (f["rating_sum"] + PRIOR_STRENGTH * step_mean) / (f["n_reviews"] + PRIOR_STRENGTH)
    f["log_n_reviews"] = np.log1p(f["n_reviews"])
    f["is_launch"] = f["n_reviews"] == 0
    f["days_since_first"] = (cutoff - f["first_review"]).dt.days.fillna(-1)
    # Quartile of price within the step (0 = cheapest quarter, 3 = most expensive).
    pct = f.groupby("step")["price_usd"].rank(pct=True, method="average")
    f["price_tier"] = np.minimum((pct * 4).astype(int), 3)
    return f.drop(columns=["rating_sum", "first_review"])
