from __future__ import annotations

import pandas as pd
import pytest

from sillage.export import ExportError, check_export


def test_rejects_author_id():
    with pytest.raises(ExportError, match="author_id"):
        check_export(pd.DataFrame({"author_id": ["1"], "x": [1]}))


def test_rejects_review_text_any_case():
    with pytest.raises(ExportError):
        check_export(pd.DataFrame({"Review_Text": ["nice"]}))


def test_rejects_small_aggregates():
    df = pd.DataFrame({"product_id": ["P1", "P2"], "n": [25, 19]})
    with pytest.raises(ExportError, match="n < 20"):
        check_export(df, count_columns=("n",))


def test_accepts_clean_frame():
    check_export(pd.DataFrame({"product_id": ["P1"], "n": [20]}), count_columns=("n",))
