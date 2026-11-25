"""Error analysis on the catalogue slice: where does each model work or fail?

Client groups (history length, skin type): metrics over those clients.
Product groups (product age, price tier): recall of each model's top 10 against
only the positives in that product group.
Groups with fewer clients than min_group_size are hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sillage.config import load_config
from sillage.eval.metrics import evaluate_scores, per_user_metrics


def error_slices(ctx, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    cfg = load_config()
    k, min_n = cfg["eval"]["k"], cfg["min_group_size"]
    target = ctx.slice_target("catalogue")
    evals = {name: evaluate_scores(ctx, s, "catalogue", k) for name, s in scores.items()}
    rows = next(iter(evals.values()))["rows"]
    cf = ctx.client_feats.iloc[rows]
    out = []

    client_groups = {
        "history length": pd.cut(cf["hist_len"], [0, 1, 4, 9, np.inf], labels=["1", "2-4", "5-9", "10+"]).astype(str),
        "skin type": cf["skin_type"],
    }
    for dim, labels in client_groups.items():
        for g in sorted(labels.unique()):
            sel = (labels == g).to_numpy()
            entry = {"dimension": dim, "group": g, "clients": int(sel.sum())}
            for name, e in evals.items():
                entry[name] = float(np.nanmean(e["recall"][sel])) if sel.sum() >= min_n else None
            out.append(entry)

    pf = ctx.product_feats
    age = pd.cut(pf["days_since_first"], [-0.5, 90, 365, np.inf], labels=["< 90 days", "90-365 days", "> 1 year"])
    product_groups = {"product age": age.astype(str).to_numpy(), "price tier": pf["price_tier"].map(
        {0: "cheapest quarter", 1: "2nd quarter", 2: "3rd quarter", 3: "most expensive quarter"}).to_numpy()}
    for dim, labels in product_groups.items():
        for g in [x for x in pd.unique(labels) if x != "nan"]:
            group_target = target[rows] & (labels == g)[None, :]
            has = group_target.any(axis=1)
            entry = {"dimension": dim, "group": g, "clients": int(has.sum())}
            for name, e in evals.items():
                if has.sum() >= min_n:
                    m = per_user_metrics(e["topk"][has], group_target[has], k)
                    entry[name] = float(np.nanmean(m["recall"]))
                else:
                    entry[name] = None
            out.append(entry)
    return pd.DataFrame(out)
