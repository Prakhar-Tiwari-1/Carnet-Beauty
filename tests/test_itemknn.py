from __future__ import annotations

import numpy as np
import pandas as pd

from sillage.models.itemknn import ItemKNN


def _reviews():
    rows = [("a", "P1"), ("a", "P2"), ("b", "P1"), ("b", "P2"), ("c", "P3")]
    return pd.DataFrame({"author_id": [r[0] for r in rows], "product_id": [r[1] for r in rows], "rating": 5,
                         "submission_time": pd.Timestamp("2022-01-01"), "incentivised": False, "skin_type": "dry"})


def test_no_nan_with_zero_shrink():
    knn = ItemKNN(shrink=0).fit(_reviews(), pd.Timestamp("2022-06-01"))
    assert np.isfinite(knn.sim).all()
    s, mx = knn.score(knn.liked_indices(["a", "c"]), np.array(["P1", "P2", "P3"]))
    assert np.isfinite(s).all()


def test_similarity_and_explanation():
    knn = ItemKNN(shrink=0).fit(_reviews(), pd.Timestamp("2022-06-01"))
    item, co = knn.explain(np.array([knn.item_idx["P1"]]), "P2")
    assert item == "P1" and co == 2
    assert knn.sim[knn.item_idx["P1"], knn.item_idx["P3"]] == 0
