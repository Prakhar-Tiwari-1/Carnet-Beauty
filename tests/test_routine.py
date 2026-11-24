from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from sillage.routine import (
    Consultation, best_combo, build_routine, contains_avoided, filter_products, launch_candidates,
)


def _products():
    rows = []
    rng = np.random.default_rng(0)
    for step in ["cleanser", "treatment", "moisturiser"]:
        for i in range(25):
            rows.append({
                "product_id": f"{step[0]}{i}", "product_name": f"{step} {i}", "brand_name": "A" if i % 2 else "B",
                "step": step, "price_usd": float(10 + 5 * i), "out_of_stock": i == 0,
                "concerns": ["Dryness"] if i % 3 == 0 else ["Pores"],
                "ingredients_text": "Water, Glycerin, Parfum (Fragrance)" if i % 4 == 0 else "Water, Glycerin",
                "is_new": i == 24, "score": float(rng.random()),
            })
    return pd.DataFrame(rows)


def test_avoid_is_whole_word_and_uses_inci_synonym():
    s = pd.Series(["Water, Parfum (Fragrance)", "Water, Fragrance-free base?", "Alcohol Denat.", "Cetearyl Alcohol"])
    assert contains_avoided(s, ["fragrance"]).tolist() == [True, True, False, False]
    assert contains_avoided(s, ["alcohol denat"]).tolist() == [False, False, True, False]


def test_filters_remove_out_of_stock_avoided_and_guarded():
    p = _products()
    c = Consultation("dry", [], 999, avoid=["fragrance"])
    elig, funnel, notes = filter_products(p, c, guarded={"t5"}, min_products=20)
    assert not elig["out_of_stock"].any()
    assert not contains_avoided(elig["ingredients_text"], ["fragrance"]).any()
    assert "t5" not in set(elig["product_id"])
    assert any("not an allergy or safety guarantee" in n for n in notes)
    assert list(funnel["filter"])[:2] == ["catalogue", "in stock"]


def test_concern_filter_skipped_and_reported_when_few_products():
    p = _products()
    # 'Dryness' is on 9 products per step (< 20): filter skipped and reported, not silently relaxed.
    elig, _, notes = filter_products(p, Consultation("dry", ["Dryness"], 999), set(), min_products=20)
    assert len(elig) == len(p[~p["out_of_stock"]])
    assert sum("not filtered on Dryness" in n for n in notes) == 3
    # With a lower threshold the filter applies.
    elig, _, _ = filter_products(p, Consultation("dry", ["Dryness"], 999), set(), min_products=5)
    assert elig["concerns"].map(lambda cs: "Dryness" in cs).all()


def test_best_combo_is_exact():
    p = _products().assign(pct=lambda d: d.groupby("step")["score"].rank(pct=True))
    budget = 80.0
    combo = best_combo(p, budget)
    parts = [p[p["step"] == s] for s in ["cleanser", "treatment", "moisturiser"]]
    best = max(
        (a.pct + b.pct + c.pct for a, b, c in itertools.product(*(x.itertuples() for x in parts))
         if a.price_usd + b.price_usd + c.price_usd <= budget),
    )
    assert combo["price_usd"].sum() <= budget
    assert combo["pct"].sum() == pytest.approx(best)


def test_nothing_passes_names_budget_constraint():
    p = _products()
    scores = p.set_index("product_id")["score"]
    r = build_routine(p.drop(columns="score"), scores, Consultation("dry", [], budget=20), set(), 20)
    assert r.picks is None
    assert "above the budget of $20.00" in r.failure[0]


def test_nothing_passes_names_filter_that_emptied_a_step():
    p = _products()
    p.loc[p["step"] == "moisturiser", "ingredients_text"] = "Water, Parfum"
    scores = p.set_index("product_id")["score"]
    r = build_routine(p.drop(columns="score"), scores, Consultation("dry", [], 500, avoid=["fragrance"]), set(), 20)
    assert r.picks is None
    assert r.failure[0].startswith("No moisturiser passes the filters: the filter 'without fragrance, parfum'")


def test_launch_slot_orders_by_price_fit():
    p = _products()
    p.loc[p["step"] == "cleanser", "is_new"] = True
    out, note = launch_candidates(p, Consultation("dry", [], 999, usual_spend=50), set(), 20)
    assert out["price_usd"].iloc[0] == 50
    assert "not a prediction" not in note and "matches" in note


def test_launch_slot_uses_full_catalogue_for_concern_rules():
    p = _products()
    p["concerns"] = [["Pores"]] * len(p)
    p.loc[p["product_id"].isin(["c24", "t24"]), "concerns"] = [["Dryness"], ["Dryness"]]
    out, _ = launch_candidates(p, Consultation("dry", ["Pores"], 999), set(), 20)
    # Pores is carried by 24+ products per step in the catalogue, so it filters launches too.
    assert set(out["product_id"]) == {"m24"}
