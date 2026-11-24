"""Build a three-step routine: hard filters first, then an exact budget search.

Filters (never relaxed silently; every exclusion is counted and reported):
1. in stock
2. ingredients to avoid: case-insensitive whole-word text match on the listed
   ingredients. NOT an allergy or safety guarantee.
3. skin-type guard: product clearly disliked by reviewers with the client's skin type
4. stated concerns: product tagged "Good for" at least one stated concern, applied to
   a step only for concerns carried by >= min_group_size products in that step
5. brand (brand-owner view only)

Search: maximise the sum of within-step score percentiles subject to the total
budget. Exact: for every (cleanser, treatment) pair, the best moisturiser that
fits the remaining budget is found with a sorted prefix maximum.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STEPS = ["cleanser", "treatment", "moisturiser"]

# INCI labels list fragrance as "Parfum (Fragrance)"; the two words are synonyms.
INGREDIENT_SYNONYMS = {"fragrance": ["fragrance", "parfum"], "parfum": ["fragrance", "parfum"]}

INGREDIENT_DISCLAIMER = (
    "Ingredient filtering is a text match on the ingredient list published by the retailer. "
    "It is not an allergy or safety guarantee: lists can be incomplete, the same "
    "substance can appear under other names, and a product whose list says "
    "'fragrance-free' is also excluded when avoiding fragrance."
)


@dataclass
class Consultation:
    skin_type: str
    concerns: list[str]
    budget: float
    avoid: list[str] = field(default_factory=list)
    usual_spend: float | None = None
    brand: str | None = None


@dataclass
class RoutineResult:
    picks: pd.DataFrame | None              # one row per step, in STEPS order
    total: float | None
    notes: list[str]                        # filters applied or skipped, in plain words
    failure: list[str]                      # why nothing passed (empty when picks exist)
    funnel: pd.DataFrame                    # products left per step after each filter
    eligible: pd.DataFrame                  # all products passing every filter


def avoid_terms(avoid: list[str]) -> list[str]:
    terms: list[str] = []
    for a in avoid:
        a = a.strip().lower()
        if a:
            terms.extend(INGREDIENT_SYNONYMS.get(a, [a]))
    return sorted(set(terms))


def contains_avoided(ingredients: pd.Series, avoid: list[str]) -> pd.Series:
    terms = avoid_terms(avoid)
    if not terms:
        return pd.Series(False, index=ingredients.index)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    return ingredients.fillna("").str.contains(pattern)


def concern_rules(products: pd.DataFrame, concerns: list[str], min_products: int):
    """Per step, which stated concerns can filter (enough tagged products) and which cannot."""
    applied, skipped = {}, {}
    for step in STEPS:
        sp = products[products["step"] == step]
        counts = {c: int(sp["concerns"].map(lambda cs, c=c: c in cs).sum()) for c in concerns}
        applied[step] = [c for c in concerns if counts[c] >= min_products]
        skipped[step] = {c: counts[c] for c in concerns if counts[c] < min_products}
    return applied, skipped


def filter_products(products: pd.DataFrame, c: Consultation, guarded: set[str], min_products: int,
                    catalogue: pd.DataFrame | None = None):
    """Apply filters in order. Returns eligible products, the funnel and notes.

    catalogue: full routine catalogue used to decide which concerns can filter a step
    (defaults to `products`).
    """
    p = products.copy()
    notes: list[str] = []
    stages = [("catalogue", np.ones(len(p), bool))]

    if c.brand:
        stages.append((f"brand {c.brand}", (p["brand_name"] == c.brand).to_numpy()))
    stages.append(("in stock", (~p["out_of_stock"]).to_numpy()))
    if c.avoid:
        stages.append((f"without {', '.join(avoid_terms(c.avoid))}",
                       (~contains_avoided(p["ingredients_text"], c.avoid)).to_numpy()))
        notes.append(INGREDIENT_DISCLAIMER)
    stages.append((f"not disliked by {c.skin_type} skin", (~p["product_id"].isin(guarded)).to_numpy()))

    if c.concerns:
        applied, skipped = concern_rules(products if catalogue is None else catalogue, c.concerns, min_products)
        ok = np.ones(len(p), bool)
        for step in STEPS:
            in_step = (p["step"] == step).to_numpy()
            if applied[step]:
                match = p["concerns"].map(lambda cs, a=applied[step]: bool(set(cs) & set(a))).to_numpy()
                ok &= ~in_step | match
            for concern, n in skipped[step].items():
                notes.append(f"{step.capitalize()} not filtered on {concern}: only {n} "
                             f"{step}s carry that Sephora tag (minimum {min_products}).")
        stages.append(("matches a stated concern", ok))

    keep = np.ones(len(p), bool)
    rows = []
    for name, mask in stages:
        keep &= mask
        rows.append({"filter": name, **{s: int((keep & (p["step"] == s).to_numpy()).sum()) for s in STEPS}})
    return p[keep], pd.DataFrame(rows), notes


def best_combo(eligible: pd.DataFrame, budget: float):
    """Exact maximum of summed step-percentile scores with total price <= budget."""
    parts = {s: eligible[eligible["step"] == s].reset_index(drop=True) for s in STEPS}
    if any(len(v) == 0 for v in parts.values()):
        return None
    c, t, m = parts["cleanser"], parts["treatment"], parts["moisturiser"]
    m = m.sort_values(["price_usd", "pct"], ascending=[True, False]).reset_index(drop=True)
    m_price = m["price_usd"].to_numpy(float)
    best_idx = np.zeros(len(m), int)
    for i in range(1, len(m)):
        best_idx[i] = i if m["pct"].iat[i] > m["pct"].iat[best_idx[i - 1]] else best_idx[i - 1]

    pc, pt = c["price_usd"].to_numpy(float), t["price_usd"].to_numpy(float)
    sc, st = c["pct"].to_numpy(float), t["pct"].to_numpy(float)
    remaining = budget - (pc[:, None] + pt[None, :])
    pos = np.searchsorted(m_price, remaining + 1e-9, side="right") - 1
    valid = pos >= 0
    if not valid.any():
        return None
    mi = best_idx[np.clip(pos, 0, None)]
    total_score = np.where(valid, sc[:, None] + st[None, :] + m["pct"].to_numpy()[mi], -np.inf)
    i, j = np.unravel_index(np.argmax(total_score), total_score.shape)
    return pd.concat([c.iloc[[i]], t.iloc[[j]], m.iloc[[mi[i, j]]]], ignore_index=True)


def build_routine(products: pd.DataFrame, scores: pd.Series, c: Consultation, guarded: set[str], min_products: int) -> RoutineResult:
    """products: routine catalogue rows; scores: product_id -> model score (higher = better)."""
    p = products.assign(score=products["product_id"].map(scores))
    p = p[p["score"].notna()]
    p["pct"] = p.groupby("step")["score"].rank(pct=True)
    eligible, funnel, notes = filter_products(p, c, guarded, min_products)
    combo = best_combo(eligible, c.budget)
    if combo is not None:
        return RoutineResult(combo, float(combo["price_usd"].sum()), notes, [], funnel, eligible)
    return RoutineResult(None, None, notes, diagnose(funnel, eligible, c.budget), funnel, eligible)


def diagnose(funnel: pd.DataFrame, eligible: pd.DataFrame, budget: float) -> list[str]:
    """Name the binding constraint in plain words."""
    out = []
    for step in STEPS:
        col = funnel[step].to_numpy()
        if col[-1] == 0:
            zero_at = int(np.argmax(col == 0))
            before = int(col[zero_at - 1]) if zero_at > 0 else 0
            out.append(f"No {step} passes the filters: the filter '{funnel['filter'].iat[zero_at]}' "
                       f"removed the last {before}.")
    if out:
        return out
    cheapest = eligible.sort_values("price_usd").groupby("step").head(1).set_index("step").loc[STEPS]
    total = cheapest["price_usd"].sum()
    detail = "; ".join(f"{s} ${cheapest.loc[s, 'price_usd']:.2f} ({cheapest.loc[s, 'product_name']})" for s in STEPS)
    return [f"The cheapest routine that passes every filter costs ${total:.2f}, above the budget of "
            f"${budget:.2f}. Cheapest per step: {detail}."]


def launch_candidates(products: pd.DataFrame, c: Consultation, guarded: set[str], min_products: int) -> tuple[pd.DataFrame, str]:
    """'New this season' slot: Sephora-flagged new products passing the same filters.

    Ordered by closeness to the client's usual spend per product. Without a usual
    spend, ordered by price, and the note says so. This is a match, not a prediction.
    """
    new = products[products["is_new"]].assign(score=0.0, pct=0.0)
    eligible, _, _ = filter_products(new, c, guarded, min_products, catalogue=products)
    if c.usual_spend:
        eligible = eligible.assign(price_fit=-np.abs(np.log(eligible["price_usd"] / c.usual_spend)))
        eligible = eligible.sort_values(["price_fit", "price_usd"], ascending=[False, True])
        note = (f"New at Sephora, too recent for reliable review data. Shown because it matches "
                f"the stated concern and is close to the usual spend of ${c.usual_spend:.0f} per product.")
    else:
        eligible = eligible.sort_values("price_usd")
        note = "New at Sephora, too recent for reliable review data. No usual spend given, so ordered by price."
    return eligible, note
