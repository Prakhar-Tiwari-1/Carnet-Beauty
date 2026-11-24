"""Client features and profiles computed only from reviews before the cutoff.

A client's concern profile is the set of Sephora "Good for" tags on routine
products they liked. Nothing is inferred from review text.
"""
from __future__ import annotations

import pandas as pd

from sillage.config import load_config
from sillage.data.splits import history


def client_features(
    reviews: pd.DataFrame,
    catalogue: pd.DataFrame,
    price_map: pd.Series,
    cutoff: pd.Timestamp,
    authors=None,
) -> pd.DataFrame:
    """One row per author with history. price_map: product_id -> price_usd (all products)."""
    min_rating = load_config()["positive_min_rating"]
    h = history(reviews, cutoff)
    if authors is not None:
        h = h[h["author_id"].isin(authors)]
    h = h.assign(price=h["product_id"].map(price_map))
    g = h.groupby("author_id")

    f = pd.DataFrame({
        "hist_len": g.size(),
        "days_since_last": (cutoff - g["submission_time"].max()).dt.days,
        "mean_rating_given": g["rating"].mean(),
        "median_price": g["price"].median(),
        "skin_type": h.dropna(subset=["skin_type"]).groupby("author_id")["skin_type"]
        .agg(lambda s: s.value_counts().index[0]),
    })
    f["skin_type"] = f["skin_type"].fillna("unknown")

    liked = h[h["rating"] >= min_rating].merge(
        catalogue[["product_id", "brand_name", "concerns"]], on="product_id", how="inner"
    )
    f["liked_brands"] = liked.groupby("author_id")["brand_name"].agg(lambda s: sorted(set(s)))
    f["concerns"] = liked.groupby("author_id")["concerns"].agg(
        lambda s: sorted({c for cs in s for c in cs})
    )
    for c in ["liked_brands", "concerns"]:
        f[c] = f[c].apply(lambda v: v if isinstance(v, list) else [])
    return f.reset_index()
