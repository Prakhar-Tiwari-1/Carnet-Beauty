from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sillage.models.hand_score import count_matrix, like_rate_matrix, price_fit, skin_guard


@pytest.fixture
def setup():
    cat = pd.DataFrame({"product_id": ["A", "B", "C", "D"], "step": ["treatment"] * 4})
    # dry-skin reviewers: A loved, B disliked with many reviews, C disliked with few, D average
    t = pd.DataFrame({
        "product_id": ["A", "B", "C", "D"],
        "skin_type": ["dry"] * 4,
        "n": [200, 200, 5, 200],
        "likes": [190, 100, 1, 170],
    })
    return cat, t


def test_guard_excludes_clearly_disliked_with_enough_reviews(setup):
    cat, t = setup
    rates = like_rate_matrix(t, cat, m=20)
    guard = skin_guard(rates, count_matrix(t, cat), cat["step"].to_numpy(), gap=0.10, min_n=20)
    assert guard.loc["B", "dry"]           # 50% vs median ~ 80%: excluded
    assert not guard.loc["C", "dry"]       # only 5 dry reviewers: never excluded
    assert not guard.loc["A", "dry"]
    assert not guard.loc["D", "dry"]


def test_shrinkage_pulls_small_samples_to_step_rate(setup):
    cat, t = setup
    rates = like_rate_matrix(t, cat, m=20)
    step_rate = t["likes"].sum() / t["n"].sum()
    assert abs(rates.loc["C", "dry"] - step_rate) < abs(1 / 5 - step_rate)


def test_price_fit_prefers_reference_price():
    pf = price_fit(np.array([30.0, 60.0, 120.0]), np.array([60.0]))
    assert pf.argmax() == 1
    assert pf[0, 0] == pytest.approx(pf[0, 2])   # symmetric on a log scale: half = double
