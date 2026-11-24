"""Everything a model needs to score one period, as aligned arrays.

Rows = evaluated users (returning clients with a positive in the period).
Columns = routine catalogue products, in catalogue order.
All inputs are computed from reviews strictly before period.cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sillage.data.splits import Period, seen_items, targets
from sillage.features.client import client_features
from sillage.features.match import skin_type_table
from sillage.features.product import product_features

SLICES = ("catalogue", "launch", "all")


@dataclass
class EvalContext:
    period: Period
    reviews: pd.DataFrame          # full table; models must use history(reviews, cutoff)
    catalogue: pd.DataFrame
    users: np.ndarray              # author_id per row
    products: np.ndarray           # product_id per column
    product_feats: pd.DataFrame    # aligned to products
    client_feats: pd.DataFrame     # aligned to users
    skin_table: pd.DataFrame
    seen: np.ndarray               # bool U x P, reviewed before cutoff
    target: np.ndarray             # bool U x P, positives in the period
    is_launch: np.ndarray          # bool P, no review before cutoff
    extra: dict = field(default_factory=dict)

    def candidates(self, slice_name: str) -> np.ndarray:
        """Products a model may rank in this slice."""
        if slice_name == "catalogue":
            return ~self.is_launch
        if slice_name == "launch":
            return self.is_launch
        return np.ones_like(self.is_launch)

    def slice_target(self, slice_name: str) -> np.ndarray:
        return self.target & self.candidates(slice_name)[None, :]


def _index(values, order: np.ndarray) -> np.ndarray:
    pos = pd.Series(np.arange(len(order)), index=order)
    return pos.reindex(values).to_numpy()


def build_context(
    reviews: pd.DataFrame,
    catalogue: pd.DataFrame,
    price_map: pd.Series,
    period: Period,
    include_incentivised: bool = False,
) -> EvalContext:
    tg = targets(reviews, period, catalogue["product_id"], include_incentivised)
    users = np.sort(tg["author_id"].unique())
    products = catalogue["product_id"].to_numpy()

    pf = product_features(reviews, catalogue, period.cutoff).set_index("product_id").loc[products]
    cf = client_features(reviews, catalogue, price_map, period.cutoff, authors=users)
    cf = cf.set_index("author_id").loc[users]

    def mask(pairs: pd.DataFrame) -> np.ndarray:
        pairs = pairs[pairs["author_id"].isin(users) & pairs["product_id"].isin(products)]
        m = np.zeros((len(users), len(products)), dtype=bool)
        m[_index(pairs["author_id"], users), _index(pairs["product_id"], products)] = True
        return m

    return EvalContext(
        period=period,
        reviews=reviews,
        catalogue=catalogue.set_index("product_id").loc[products].reset_index(),
        users=users,
        products=products,
        product_feats=pf.reset_index(),
        client_feats=cf.reset_index(),
        skin_table=skin_type_table(reviews, catalogue, period.cutoff),
        seen=mask(seen_items(reviews, period.cutoff)),
        target=mask(tg),
        is_launch=pf["is_launch"].to_numpy(),
    )
