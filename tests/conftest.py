"""Small synthetic fixtures so tests run without the raw Kaggle data."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def catalogue() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4"],
        "product_name": ["Gel Cleanser", "Calm Serum", "Rich Cream", "New Serum"],
        "brand_name": ["A", "B", "A", "C"],
        "step": ["cleanser", "treatment", "moisturiser", "treatment"],
        "price_usd": [20.0, 60.0, 45.0, 80.0],
        "out_of_stock": [False, False, False, False],
        "concerns": [["Pores"], ["Redness", "Dryness"], ["Dryness"], ["Anti-Aging"]],
    })


def _r(author, product, date, rating=5, skin="dry", incent=False):
    return {
        "author_id": author, "product_id": product, "rating": rating,
        "is_recommended": 1.0, "submission_time": pd.Timestamp(date),
        "skin_type": skin, "incentivised": incent,
    }


@pytest.fixture
def reviews() -> pd.DataFrame:
    rows = [
        _r("u1", "P1", "2022-01-10"),
        _r("u1", "P2", "2022-03-01", rating=2),
        _r("u2", "P2", "2022-02-01", skin="oily"),
        _r("u2", "P3", "2022-05-01", skin="oily", incent=True),
        _r("u3", "P3", "2022-06-15", rating=4),
        # validation period B
        _r("u1", "P3", "2022-08-01"),            # new + liked -> positive
        _r("u1", "P2", "2022-08-05"),            # already seen -> not a target
        _r("u2", "P1", "2022-09-01", incent=True),  # sampled -> excluded
        _r("u3", "P4", "2022-10-01"),            # launch product -> positive, launch
        _r("u9", "P1", "2022-10-02"),            # no history -> not evaluated
        _r("u3", "P1", "2022-10-03", rating=3),  # 3 stars -> not positive
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def future_rows() -> pd.DataFrame:
    """Rows dated on/after the B cutoff that must never change A-based features."""
    return pd.DataFrame([
        _r("u1", "P1", "2022-07-01", rating=1),
        _r("u2", "P2", "2023-01-01", rating=1, skin="dry"),
        _r("u4", "P3", "2022-12-24", rating=5),
        _r("u1", "P4", "2022-07-02", rating=1, skin="oily"),
    ])
