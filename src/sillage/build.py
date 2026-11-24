"""One command rebuilds everything: python -m sillage.build [--stage NAME]."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from sillage.config import load_config

STAGES = ["step0", "data"]


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
