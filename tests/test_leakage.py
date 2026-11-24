"""No feature may use information from its target period."""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from sillage.data.ingest import NEVER_LOAD, REVIEW_COLUMNS, SNAPSHOT_COLUMNS, flag_incentivised
from sillage.data.splits import Period, targets
from sillage.features.client import client_features
from sillage.features.match import skin_type_table
from sillage.features.product import product_features

B = Period("B", pd.Timestamp("2022-07-01"), pd.Timestamp("2022-12-01"))


def _with_future(reviews, future_rows):
    return pd.concat([reviews, future_rows], ignore_index=True)


def test_product_features_ignore_future(reviews, future_rows, catalogue):
    a = product_features(reviews, catalogue, B.cutoff)
    b = product_features(_with_future(reviews, future_rows), catalogue, B.cutoff)
    pdt.assert_frame_equal(a, b)


def test_client_features_ignore_future(reviews, future_rows, catalogue):
    prices = catalogue.set_index("product_id")["price_usd"]
    a = client_features(reviews, catalogue, prices, B.cutoff)
    b = client_features(_with_future(reviews, future_rows), catalogue, prices, B.cutoff)
    pdt.assert_frame_equal(a, b)
    assert "u4" not in set(b["author_id"])  # only appears after the cutoff


def test_skin_type_table_ignores_future(reviews, future_rows, catalogue):
    a = skin_type_table(reviews, catalogue, B.cutoff)
    b = skin_type_table(_with_future(reviews, future_rows), catalogue, B.cutoff)
    pdt.assert_frame_equal(a, b)


def test_features_at_cutoff_exclude_same_day(reviews, catalogue):
    """A review dated exactly at the cutoff belongs to the target period."""
    same_day = reviews.iloc[[0]].assign(submission_time=B.cutoff, product_id="P4")
    f = product_features(pd.concat([reviews, same_day]), catalogue, B.cutoff)
    assert f.set_index("product_id").loc["P4", "n_reviews"] == 0


def test_targets(reviews, catalogue):
    t = targets(reviews, B, catalogue["product_id"])
    pairs = set(zip(t["author_id"], t["product_id"]))
    assert pairs == {("u1", "P3"), ("u3", "P4")}
    assert t.set_index("product_id").loc["P4", "launch"]
    assert not t.set_index("product_id").loc["P3", "launch"]


def test_targets_sensitivity_includes_sampled(reviews, catalogue):
    t = targets(reviews, B, catalogue["product_id"], include_incentivised=True)
    assert ("u2", "P1") in set(zip(t["author_id"], t["product_id"]))


def test_no_snapshot_or_appearance_columns(reviews, catalogue):
    feats = product_features(reviews, catalogue, B.cutoff)
    assert not SNAPSHOT_COLUMNS & set(feats.columns)
    assert not NEVER_LOAD & set(REVIEW_COLUMNS)


def test_incentive_flag():
    text = pd.Series([
        "I received this product complimentary from Influenster.",
        "Received it for free in exchange for my honest review",
        "Love it, bought a second bottle",
        None,
    ])
    assert flag_incentivised(text).tolist() == [True, True, False, False]
