from __future__ import annotations

import pandas as pd
import pytest

from sillage import app_core
from sillage.messaging.guardrails import check_draft
from sillage.messaging.templates import render


def _data():
    rows = []
    for s_i, step in enumerate(["cleanser", "treatment", "moisturiser"]):
        for i in range(4):
            rows.append({
                "product_id": f"{step[0]}{i}", "product_name": f"{step.title()} No {i}", "brand_name": f"B{i % 2}",
                "step": step, "price_usd": 20.0 + 10 * i, "out_of_stock": False, "concerns": ["Dryness"],
                "best_for_skin": [], "ingredients_text": "Water", "is_new": i == 3,
                "pop_score": float(4 - i), "likes_12m": pd.NA if i == 2 else 100 - i,
            })
    products = pd.DataFrame(rows)
    products["likes_12m"] = products["likes_12m"].astype("Int64")
    skin = pd.DataFrame({"product_id": ["c0", "t0", "m0", "m1"], "skin_type": ["dry"] * 4,
                         "n": [50, 40, 30, 25], "likes": [45, 30, 27, 5], "guard": [False, False, False, True]})
    returning = pd.DataFrame({
        "client_id": "r", "product_id": products["product_id"], "routine_score": products["pop_score"][::-1].to_numpy(),
        "launch_score": 1.0, "reason": "63 clients who liked X also liked this.", "launch_reason": None,
    })
    meta = {"min_group_size": 20, "returning_clients": {"r": {
        "usual_spend": 30.0, "routine_model_label": "item-kNN", "launch_model_label": "content match"}}}
    clients = [
        {"id": "n", "name": "Nina A.", "type": "new", "language": "en", "consent": True, "skin_type": "dry",
         "concerns": [], "budget_usd": 200, "avoid": [], "usual_spend_usd": 30},
        {"id": "r", "name": "Rose B.", "type": "returning", "language": "fr", "consent": True, "skin_type": "dry",
         "concerns": [], "budget_usd": 200, "avoid": [], "usual_spend_usd": None},
    ]
    return app_core.AppData(products, skin, returning, meta, {}, clients)


def test_new_client_path_and_reasons_respect_min_n():
    d = _data()
    client = d.client("n")
    view = app_core.build_view(d, client, app_core.consultation(client))
    assert view.path_label.startswith("New client")
    assert view.routine.picks is not None
    assert "m1" not in set(view.routine.picks["product_id"])          # skin-type guard
    texts = [r.text for rs in view.reasons.values() for r in rs]
    assert any("Liked by 90% of 50 dry-skin reviewers" in t for t in texts)
    assert not any("of 5 " in t for t in texts)


def test_returning_client_uses_model_scores_and_reason():
    d = _data()
    client = d.client("r")
    view = app_core.build_view(d, client, app_core.consultation(client))
    assert view.path_label == "Returning client: item-kNN"
    assert view.usual_spend == 30.0
    first = view.routine.picks.iloc[0]["product_id"]
    assert view.reasons[first][0].text.startswith("63 clients")


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_message_facts_pass_guardrails(lang):
    d = _data()
    client = d.client("n")
    view = app_core.build_view(d, client, app_core.consultation(client))
    facts = app_core.message_facts(d, client, view, lang)
    text = render(facts)
    assert check_draft(text, facts, d.products["product_name"].tolist()).ok
    # The displayed figures may be quoted; an invented one may not.
    assert check_draft(text + " Liked by 90% of 50 reviewers.", facts, []).ok
    assert not check_draft(text + " Liked by 91% of reviewers.", facts, []).ok


def test_brand_gaps_flags_missing_step():
    d = _data()
    d.products.loc[(d.products["brand_name"] == "B1") & (d.products["step"] == "cleanser"), "out_of_stock"] = True
    g = app_core.brand_gaps(d, "B1")
    assert (g[g["step"] == "cleanser"]["products"] == 0).all()
