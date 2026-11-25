"""Content-based scores: TF-IDF over product name, Sephora highlights and ingredients.

A client's profile = mean TF-IDF vector of the products she liked before the cutoff.
Score = cosine similarity between profile and product. Works for brand-new products,
because it needs no reviews of the candidate, only its published description.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from sillage.config import load_config
from sillage.data.catalogue import parse_list
from sillage.data.splits import history

_TOKEN = re.compile(r"[a-z][a-z0-9\-]{2,}")


def _tokens(row) -> list[str]:
    text = " ".join([
        str(row["product_name"]),
        " ".join(parse_list(row["highlights"])),
        " ".join(parse_list(row["ingredients"])),
    ]).lower()
    return _TOKEN.findall(text)


class ContentTFIDF:
    def __init__(self, products: pd.DataFrame, min_df: int = 3):
        """products: product_info rows for every skincare product (published text only)."""
        docs = products.apply(_tokens, axis=1)
        df = Counter(t for d in docs for t in set(d))
        vocab = {t: i for i, t in enumerate(sorted(t for t, n in df.items() if n >= min_df))}
        n = len(docs)
        idf = np.array([np.log(n / df[t]) for t in sorted(vocab, key=vocab.get)], np.float32)
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            for t, c in Counter(x for x in d if x in vocab).items():
                rows.append(r)
                cols.append(vocab[t])
                vals.append((1 + np.log(c)) * idf[vocab[t]])
        M = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        norms = np.sqrt(M.multiply(M).sum(1)).A1
        self.M = sparse.diags(1 / np.maximum(norms, 1e-9)) @ M
        self.idx = pd.Series(np.arange(n), index=products["product_id"].to_numpy())

    def score(self, reviews: pd.DataFrame, cutoff: pd.Timestamp, authors, products) -> np.ndarray:
        min_rating = load_config()["positive_min_rating"]
        h = history(reviews, cutoff)
        liked = h[(h["rating"] >= min_rating) & ~h["incentivised"] & h["author_id"].isin(authors)]
        liked = liked[liked["product_id"].isin(self.idx.index)]
        a_idx = pd.Series(np.arange(len(authors)), index=authors)
        U = sparse.csr_matrix(
            (np.ones(len(liked), np.float32), (a_idx.loc[liked["author_id"]].to_numpy(), self.idx.loc[liked["product_id"]].to_numpy())),
            shape=(len(authors), len(self.idx)),
        )
        profile = U @ self.M
        pn = np.sqrt(profile.multiply(profile).sum(1)).A1
        profile = sparse.diags(1 / np.maximum(pn, 1e-9)) @ profile
        cols = self.idx.reindex(products).to_numpy()
        return (profile @ self.M[cols].T).toarray().astype(np.float32)
