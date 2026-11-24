"""Skin-type x product like counts, the basis of the hand-built score.

Counts use only reviews before the cutoff and ignore sampled reviews, which
are rated higher (4.53 vs 4.26 stars) and are not purchases.
"""
from __future__ import annotations

import pandas as pd

from sillage.config import load_config
from sillage.data.splits import history


def skin_type_table(reviews: pd.DataFrame, catalogue: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows: product_id, skin_type, step, n (reviews), likes. Includes an 'all' skin type."""
    min_rating = load_config()["positive_min_rating"]
    h = history(reviews, cutoff)
    h = h[~h["incentivised"] & h["product_id"].isin(catalogue["product_id"])]
    h = h.assign(like=h["rating"] >= min_rating)

    by_type = (
        h.dropna(subset=["skin_type"])
        .groupby(["product_id", "skin_type"])["like"].agg(n="size", likes="sum")
        .reset_index()
    )
    overall = h.groupby("product_id")["like"].agg(n="size", likes="sum").reset_index()
    overall["skin_type"] = "all"
    t = pd.concat([by_type, overall], ignore_index=True)
    t = t.merge(catalogue[["product_id", "step"]], on="product_id")
    t["likes"] = t["likes"].astype(int)
    return t


def concern_overlap(client_concerns: list[str], product_concerns: list[str]) -> float:
    """Share of the client's concerns that the product is tagged for (0 if none stated)."""
    if not client_concerns:
        return 0.0
    return len(set(client_concerns) & set(product_concerns)) / len(set(client_concerns))
