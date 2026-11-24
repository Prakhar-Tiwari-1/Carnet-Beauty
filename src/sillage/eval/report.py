"""Compare models on one context: metrics with paired bootstrap CIs, coverage, bestseller bias."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sillage.config import load_config
from sillage.eval.context import SLICES
from sillage.eval.metrics import bestseller_share, coverage, evaluate_scores
from sillage.eval.bootstrap import bootstrap_ci
from sillage.models.popularity import popularity_vector

BASELINE = "popularity"


def compare(ctx, scores: dict[str, np.ndarray], slices=SLICES) -> dict:
    cfg = load_config()
    k, n_boot, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap_samples"], cfg["seed"]
    pop = popularity_vector(ctx.product_feats)
    out = {}
    for sl in slices:
        cand = ctx.candidates(sl)
        evals = {name: evaluate_scores(ctx, s, sl, k) for name, s in scores.items()}
        any_eval = next(iter(evals.values()))
        res = {"n_users": int(len(any_eval["rows"])), "n_candidates": int(cand.sum()), "models": {}}
        for metric in ("recall", "ndcg"):
            ci = bootstrap_ci({n: e[metric] for n, e in evals.items()}, BASELINE, n_boot, seed)
            for name, v in ci.items():
                res["models"].setdefault(name, {})[metric] = v
        for name, e in evals.items():
            cand_idx = np.flatnonzero(cand)
            res["models"][name]["coverage"] = coverage(e["topk"], len(cand_idx))
            res["models"][name]["bestseller_share"] = (
                bestseller_share(e["topk"], pop, cand) if sl != "launch" else None
            )
        out[sl] = res
    return out


def to_table(results: dict) -> pd.DataFrame:
    rows = []
    for sl, res in results.items():
        for name, m in res["models"].items():
            rows.append({
                "slice": sl,
                "users": res["n_users"],
                "model": name,
                "recall@10": f"{m['recall']['mean']:.4f} [{m['recall']['ci'][0]:.4f}, {m['recall']['ci'][1]:.4f}]",
                "Δrecall vs pop": f"{m['recall']['diff_vs_baseline']:+.4f} [{m['recall']['diff_ci'][0]:+.4f}, {m['recall']['diff_ci'][1]:+.4f}]",
                "ndcg@10": f"{m['ndcg']['mean']:.4f}",
                "Δndcg vs pop": f"{m['ndcg']['diff_vs_baseline']:+.4f} [{m['ndcg']['diff_ci'][0]:+.4f}, {m['ndcg']['diff_ci'][1]:+.4f}]",
                "coverage": f"{m['coverage']:.1%}",
                "bestseller": "-" if m["bestseller_share"] is None else f"{m['bestseller_share']:.0%}",
            })
    return pd.DataFrame(rows)
