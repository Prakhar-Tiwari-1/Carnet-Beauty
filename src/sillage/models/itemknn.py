"""Item-based collaborative filtering ("clients who liked X also liked Y").

similarity(i, j) = cosine(i, j) * n_co / (n_co + shrink)
    n_co   = clients who liked both i and j before the cutoff
    cosine = n_co / sqrt(n_i * n_j)
The shrink term down-weights pairs seen together only a few times.
A client's score for product j = sum (and max) of similarities to the products she liked.
History covers all skincare she liked (masks, SPF...), not only routine products.
Sampled reviews are excluded from "liked".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from sillage.config import load_config
from sillage.data.splits import history


class ItemKNN:
    def __init__(self, shrink: float):
        self.shrink = shrink

    def fit(self, reviews: pd.DataFrame, cutoff: pd.Timestamp) -> "ItemKNN":
        min_rating = load_config()["positive_min_rating"]
        h = history(reviews, cutoff)
        liked = h[(h["rating"] >= min_rating) & ~h["incentivised"]][["author_id", "product_id"]]
        self.items = np.sort(liked["product_id"].unique())
        item_idx = pd.Series(np.arange(len(self.items)), index=self.items)
        users = pd.Categorical(liked["author_id"])
        x = sparse.csr_matrix(
            (np.ones(len(liked), np.float32), (users.codes, item_idx.loc[liked["product_id"]].to_numpy())),
            shape=(len(users.categories), len(self.items)),
        )
        x.data[:] = 1.0
        co = (x.T @ x).toarray().astype(np.float32)
        n = np.diag(co).copy()
        np.fill_diagonal(co, 0)
        denom = np.sqrt(np.outer(n, n))
        cos = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
        self.co = co
        self.sim = cos * co / (co + self.shrink)
        self.item_idx = item_idx
        self._liked = liked.groupby("author_id")["product_id"].agg(list)
        return self

    def liked_indices(self, authors) -> list[np.ndarray]:
        out = []
        for a in authors:
            items = self._liked.get(a, [])
            out.append(self.item_idx.reindex(items).dropna().to_numpy(int))
        return out

    def score(self, liked_lists: list[np.ndarray], products: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (sum, max) similarity, each U x P (products in the given order)."""
        cols = self.item_idx.reindex(products).to_numpy()
        known = ~np.isnan(cols)
        sub = np.zeros((len(self.items), len(products)), np.float32)
        sub[:, known] = self.sim[:, cols[known].astype(int)]
        s_sum = np.zeros((len(liked_lists), len(products)), np.float32)
        s_max = np.zeros_like(s_sum)
        for u, idx in enumerate(liked_lists):
            if len(idx):
                block = sub[idx]
                s_sum[u] = block.sum(axis=0)
                s_max[u] = block.max(axis=0)
        return s_sum, s_max

    def explain(self, liked: np.ndarray, product_id: str) -> tuple[str | None, int]:
        """The liked product most similar to product_id, and how many clients liked both."""
        if product_id not in self.item_idx.index or not len(liked):
            return None, 0
        j = self.item_idx[product_id]
        best = liked[np.argmax(self.sim[liked, j])]
        return self.items[best], int(self.co[best, j])


def score(ctx, model: ItemKNN, seed: int = 0) -> np.ndarray:
    s_sum, _ = model.score(model.liked_indices(ctx.users), ctx.products)
    return s_sum + np.random.default_rng(seed).random(s_sum.shape) * 1e-6
