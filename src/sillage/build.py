"""One command rebuilds everything: python -m sillage.build [--stage NAME]."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from sillage.config import load_config

STAGES = ["step0", "data", "eval_val"]


def _seed() -> None:
    seed = load_config()["seed"]
    random.seed(seed)
    np.random.seed(seed)


def stage_step0() -> None:
    from sillage import step0

    print(json.dumps(step0.run(), indent=2))
    print("Wrote docs/step0.md")


def stage_data() -> None:
    """Rebuild the cached review table and print period counts (no labels are used)."""
    import pandas as pd

    from sillage.config import path
    from sillage.data.catalogue import build_catalogue
    from sillage.data.ingest import read_products, read_reviews
    from sillage.data.splits import get_periods, history, targets
    from sillage.features.client import client_features
    from sillage.features.match import skin_type_table
    from sillage.features.product import product_features

    products = read_products()
    reviews = read_reviews(rebuild=True)
    cat = build_catalogue(products)
    cat.to_parquet(path("interim") / "catalogue.parquet", index=False)
    prices = products.set_index("product_id")["price_usd"]
    t = reviews["submission_time"]
    periods = get_periods()

    rows = [{"period": "A (history)", "reviews": int((t < periods["B"].start).sum())}]
    for name, p in periods.items():
        tg = targets(reviews, p, cat["product_id"])
        rows.append({
            "period": name,
            "reviews": int(((t >= p.start) & (t < p.end)).sum()),
            "eval_users": tg["author_id"].nunique(),
            "positives": len(tg),
            "launch_share": round(tg["launch"].mean(), 3),
            "users_with_catalogue_pos": tg.loc[~tg["launch"], "author_id"].nunique(),
        })
        pf = product_features(reviews, cat, p.cutoff)
        cf = client_features(reviews, cat, prices, p.cutoff, authors=tg["author_id"].unique())
        st = skin_type_table(reviews, cat, p.cutoff)
        print(f"{name}: features from {len(history(reviews, p.cutoff)):,} reviews | "
              f"product {pf.shape} client {cf.shape} skin-type {st.shape} | "
              f"eval users with known skin type {(cf['skin_type'] != 'unknown').mean():.1%}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Concern tags kept: {sorted({c for cs in cat['concerns'] for c in cs})}")


def _load_inputs():
    import pandas as pd

    from sillage.config import path
    from sillage.data.ingest import read_products, read_reviews

    products = read_products()
    catalogue = pd.read_parquet(path("interim") / "catalogue.parquet")
    return read_reviews(), catalogue, products.set_index("product_id")["price_usd"]


def stage_eval_val() -> None:
    """Baselines on validation period B; tunes the hand score's m and lambda."""
    import pandas as pd

    from sillage.config import path
    from sillage.data.splits import get_periods
    from sillage.eval.context import build_context
    from sillage.eval.metrics import evaluate_scores
    from sillage.eval.report import compare, to_table
    from sillage.models import hand_score, popularity

    cfg = load_config()
    reviews, catalogue, prices = _load_inputs()
    ctx = build_context(reviews, catalogue, prices, get_periods()["B"])
    k = cfg["eval"]["k"]
    hs = cfg["hand_score"]

    # Tune on the catalogue slice (the headline comparison).
    grid = []
    for m in cfg["hand_score"]["m_grid"]:
        for lam in cfg["hand_score"]["lambda_grid"]:
            e = evaluate_scores(ctx, hand_score.score_v1(ctx, m, lam), "catalogue", k)
            grid.append({"m": m, "lambda": lam, "ndcg@10": e["ndcg"].mean(), "recall@10": e["recall"].mean()})
    grid = pd.DataFrame(grid).sort_values("ndcg@10", ascending=False)
    print("Hand-score v1 grid (B, catalogue slice):")
    print(grid.round(4).to_string(index=False))
    best = grid.iloc[0]

    scores = {
        "random": popularity.random_scores(ctx, cfg["seed"]),
        "popularity": popularity.score(ctx, cfg["seed"]),
        "hand_v1_like_rate": hand_score.score_v1(ctx, best["m"], best["lambda"]),
        "hand_v2_consultation": hand_score.score_v2(
            ctx, hs["m"], hs["guard_gap"], hs["guard_min_n"], cfg["seed"]),
    }
    results = compare(ctx, scores)
    print(to_table(results).to_string(index=False))

    out = {"period": "B", "hand_v1_params": {"m": float(best["m"]), "lambda": float(best["lambda"])},
           "hand_v2_params": {k2: hs[k2] for k2 in ("m", "guard_gap", "guard_min_n")},
           "grid": grid.to_dict("records"), "results": results}
    path("results").mkdir(exist_ok=True)
    (path("results") / "val_baselines.json").write_text(json.dumps(out, indent=2))
    print("Wrote results/val_baselines.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sillage.build")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    args = parser.parse_args(argv)
    _seed()
    for name in STAGES if args.stage == "all" else [args.stage]:
        print(f"== stage: {name}")
        globals()[f"stage_{name}"]()


if __name__ == "__main__":
    main()
