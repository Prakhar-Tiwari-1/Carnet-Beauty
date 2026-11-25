"""Build app_data/: the only data the deployed app reads.

"Today" for the app is the end of the dataset (2023-03-22). Product statistics use
all real reviews before that date. Synthetic returning clients are scored by the
models selected in the study, refitted on the latest data (selection is final by
then; refitting on the newest labels is standard deployment practice).

Rules enforced by export.check_export: no author IDs or review text; every
displayed count >= min_group_size (smaller counts are dropped or set to null).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from sillage.config import ROOT, load_config, path
from sillage.data.ingest import read_products
from sillage.data.splits import Period, history
from sillage.eval.context import EvalContext
from sillage.export import write_export
from sillage.features.client import client_features
from sillage.features.match import skin_type_table
from sillage.features.product import product_features
from sillage.models import hand_score

STEP_WORD = {"cleanser": "cleanser", "treatment": "treatment", "moisturiser": "moisturiser"}


def load_clients() -> list[dict]:
    with open(ROOT / "clients" / "synthetic_clients.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["clients"]


def synthetic_reviews(clients: list[dict]) -> pd.DataFrame:
    rows = []
    for c in clients:
        for h in c.get("history", []):
            rows.append({
                "author_id": f"synthetic_{c['id']}", "product_id": h["product_id"], "rating": h["rating"],
                "is_recommended": 1.0, "submission_time": pd.Timestamp(h["date"]),
                "skin_type": c["skin_type"], "incentivised": False,
            })
    return pd.DataFrame(rows)


def product_table(reviews, catalogue, products_raw, cutoff) -> pd.DataFrame:
    min_n = load_config()["min_group_size"]
    pf = product_features(reviews, catalogue, cutoff).set_index("product_id")
    p = catalogue.copy()
    p["is_new"] = p["product_id"].map(products_raw.set_index("product_id")["new"]).astype(bool)
    p["pop_score"] = np.log1p(p["product_id"].map(pf["likes_recent"]).astype(float))
    likes = p["product_id"].map(pf["likes_recent"]).astype(float)
    p["likes_12m"] = likes.where(likes >= min_n).astype("Int64")
    keep = ["product_id", "product_name", "brand_name", "step", "price_usd", "out_of_stock", "concerns",
            "best_for_skin", "ingredients_text", "is_new", "pop_score", "likes_12m"]
    return p[keep]


def skin_stats(reviews, catalogue, cutoff) -> pd.DataFrame:
    cfg = load_config()
    hs, min_n = cfg["hand_score"], cfg["min_group_size"]
    t = skin_type_table(reviews, catalogue, cutoff)
    rates = hand_score.like_rate_matrix(t, catalogue, hs["m"])
    guard = hand_score.skin_guard(rates, hand_score.count_matrix(t, catalogue), catalogue["step"].to_numpy(),
                                  hs["guard_gap"], hs["guard_min_n"])
    g = guard.stack().rename("guard").reset_index().rename(columns={"level_0": "product_id", "level_1": "skin_type"})
    t = t.merge(g, on=["product_id", "skin_type"], how="left")
    t["guard"] = t["guard"].fillna(False).astype(bool)
    return t[t["n"] >= min_n][["product_id", "skin_type", "n", "likes", "guard"]].reset_index(drop=True)


def app_context(reviews_real, reviews_aug, catalogue, prices, clients, cutoff) -> EvalContext:
    users = np.array([f"synthetic_{c['id']}" for c in clients if c["type"] == "returning"])
    products = catalogue["product_id"].to_numpy()
    seen = np.zeros((len(users), len(products)), bool)
    syn = reviews_aug[reviews_aug["author_id"].isin(users)]
    u_idx = pd.Series(np.arange(len(users)), index=users)
    p_idx = pd.Series(np.arange(len(products)), index=products)
    syn = syn[syn["product_id"].isin(products)]
    seen[u_idx.loc[syn["author_id"]].to_numpy(), p_idx.loc[syn["product_id"]].to_numpy()] = True
    pf = product_features(reviews_real, catalogue, cutoff)
    cf = client_features(reviews_aug, catalogue, prices, cutoff, authors=users).set_index("author_id").loc[users]
    return EvalContext(
        period=Period("app", cutoff, cutoff), reviews=reviews_aug, catalogue=catalogue, users=users,
        products=products, product_feats=pf, client_feats=cf.reset_index(),
        skin_table=skin_type_table(reviews_real, catalogue, cutoff), seen=seen,
        target=np.zeros_like(seen), is_launch=pf["is_launch"].to_numpy(),
    )


def write_meta(extra: dict) -> None:
    cfg = load_config()
    meta = {
        "data_end": cfg["periods"]["c_end"],
        "popularity_window_days": cfg["popularity"]["window_days"],
        "min_group_size": cfg["min_group_size"],
        "guard": {k: cfg["hand_score"][k] for k in ("m", "guard_gap", "guard_min_n")},
        "dataset_credit": "Sephora Products and Skincare Reviews by nadyinky (Kaggle), CC BY 4.0, scraped March 2023",
        **extra,
    }
    path("app_data").mkdir(exist_ok=True)
    (path("app_data") / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def base_exports(reviews, catalogue) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(load_config()["periods"]["c_end"])
    products_raw = read_products()
    prod = product_table(reviews, catalogue, products_raw, cutoff)
    stats = skin_stats(reviews, catalogue, cutoff)
    write_export(prod, path("app_data") / "products.parquet", count_columns=("likes_12m",))
    write_export(stats, path("app_data") / "skin_stats.parquet", count_columns=("n",))
    return prod, stats


def history_rows(reviews, cutoff) -> pd.DataFrame:
    return history(reviews, cutoff)


MODEL_LABELS = {
    "popularity": "popularity (what clients liked most in the past year)",
    "itemknn": "item-kNN: clients who liked what she liked also liked this",
    "rp3beta": "RP3beta graph model on what she liked",
    "ease": "EASE linear model on what she liked",
    "ials": "iALS matrix factorisation",
    "bpr": "BPR matrix factorisation",
    "content": "content match to products she liked (name, Sephora highlights, ingredients)",
    "random": "random order (no model beat random on validation)",
}


def model_label(name: str) -> str:
    if name.startswith("ranker_"):
        return f"LightGBM ranker ({name.removeprefix('ranker_').replace('_', '-')})"
    return MODEL_LABELS.get(name, name)


def _names(products_raw) -> pd.Series:
    return products_raw.set_index("product_id").apply(lambda r: f"{r['brand_name']}, {r['product_name']}", axis=1)


def explain_matrix(name, ctx, reviews_aug, tuned, content, products_raw, ranker_bundle=None) -> np.ndarray:
    """Reason text per (client, product), faithful to the model's mechanics."""
    from sillage.models.cf import EASE, LikedMatrix, RP3beta
    from sillage.models.itemknn import ItemKNN

    min_n = load_config()["min_group_size"]
    names = _names(products_raw)
    U, P = len(ctx.users), len(ctx.products)
    out = np.full((U, P), None, dtype=object)
    cutoff = ctx.period.cutoff

    if name == "itemknn":
        knn = ItemKNN(tuned["itemknn"]["shrink"]).fit(reviews_aug, cutoff)
        liked = knn.liked_indices(ctx.users)
        for u in range(U):
            for j, pid in enumerate(ctx.products):
                item, co = knn.explain(liked[u], pid)
                if item is None or co == 0:
                    continue
                out[u, j] = (f"{co} clients who liked {names[item]} also liked this." if co >= min_n else
                             f"Most similar to {names[item]}, which she liked (shared-client count under {min_n}, hidden).")
    elif name in ("ease", "rp3beta"):
        m = LikedMatrix(reviews_aug, cutoff)
        model = EASE(tuned["ease"]["lam"]) if name == "ease" else RP3beta(tuned["rp3beta"]["alpha"], tuned["rp3beta"]["beta"])
        model.fit(m)
        W = (model.B if name == "ease" else model.W).cpu().numpy()
        cols = m.product_cols(ctx.products)
        for u, row in enumerate(m.user_rows(ctx.users)):
            if row < 0:
                continue
            liked = m.X[row].indices
            for j, cj in enumerate(cols):
                if cj < 0 or not len(liked):
                    continue
                contrib = W[liked, cj]
                if contrib.max() > 0:
                    out[u, j] = f"Score driven mostly by her liking {names[m.items[liked[contrib.argmax()]]]}."
    elif name == "content":
        hist = history(reviews_aug, cutoff)
        for u, a in enumerate(ctx.users):
            liked = hist[(hist["author_id"] == a) & (hist["rating"] >= 4)]["product_id"]
            liked = [p for p in liked if p in content.idx.index]
            if not liked:
                continue
            L = content.M[content.idx.loc[liked].to_numpy()]
            sims = (L @ content.M[content.idx.reindex(ctx.products).to_numpy()].T).toarray()
            for j in range(P):
                if sims[:, j].max() > 0:
                    out[u, j] = f"Closest in description and ingredients to {names[liked[int(sims[:, j].argmax())]]}, which she liked."
    elif name.startswith("ranker_") and ranker_bundle is not None:
        from sillage.models.ranker import FEATURE_LABELS

        model, feats, rows, cols, X, rounds = ranker_bundle
        contrib = model.predict(X[feats], num_iteration=rounds, pred_contrib=True)[:, :-1]
        for k in range(len(rows)):
            top = np.argsort(-np.abs(contrib[k]))[:3]
            parts = [f"{FEATURE_LABELS.get(feats[t], feats[t].removeprefix('cf_') + ' score')} "
                     f"({'+' if contrib[k, t] > 0 else '−'})" for t in top]
            out[rows[k], cols[k]] = "Main factors in the ranker's score: " + ", ".join(parts) + "."
    elif name in ("ials", "bpr"):
        out[:] = "Latent-factor model: its score does not decompose into a per-product reason."
    return out


def build_returning(reviews, catalogue, prices, clients, log=print) -> dict:
    from sillage.data.splits import get_periods
    from sillage.eval.context import build_context
    from sillage.models.content import ContentTFIDF
    from sillage.study import Scorer, _ranker_frame, train_rankers, VARIANTS
    from sillage.models import ranker as rk

    cfg = load_config()
    seed = cfg["seed"]
    study = json.loads((path("results") / "model_study.json").read_text())
    tuned, chosen = study["tuned"], {k: v["chosen"] for k, v in study["selection"].items()}
    for k, v in cfg.get("deployment", {}).items():  # post-test override, documented in config and results
        if k in chosen and v:
            chosen[k] = v
    cutoff = pd.Timestamp(cfg["periods"]["c_end"])
    products_raw = read_products()
    skincare = products_raw[products_raw["primary_category"] == "Skincare"]
    content = ContentTFIDF(skincare)

    aug = pd.concat([reviews, synthetic_reviews(clients)], ignore_index=True)
    ctx = app_context(reviews, aug, catalogue, prices, clients, cutoff)
    scorer = Scorer(aug, content, seed)
    scores = scorer.all_scores(ctx, tuned)

    bundle = None
    if any(v.startswith("ranker_") for v in chosen.values()):
        # Final deployment refit: ranker trained on the newest labels (test period C).
        tr = build_context(reviews, catalogue, prices, get_periods()["C"])
        models = train_rankers(_ranker_frame(tr, scorer.all_scores(tr, tuned), cfg["study"]["candidates_per_model"]), seed)
        rows, cols = np.nonzero(~ctx.seen)
        hs = cfg["hand_score"]
        extra = {n: scores[n] for n in ("rp3beta", "ease", "ials", "bpr", "content")}
        X = rk.build_features(ctx, scores["itemknn"], scores["itemknn_max"], rows, cols,
                              hs["m"], hs["guard_gap"], hs["guard_min_n"], extra=extra)
        for key, name in chosen.items():
            if name.startswith("ranker_"):
                variant, rounds = name.removeprefix("ranker_").split("@")
                feats = VARIANTS[variant][0]
                s = np.full(ctx.seen.shape, -np.inf, np.float32)
                s[rows, cols] = models[variant].predict(X[feats], num_iteration=int(rounds))
                scores[name] = s
                bundle = (models[variant], feats, rows, cols, X, int(rounds))

    def matrix(name):
        return scores[name] if name in scores else np.zeros(ctx.seen.shape, np.float32)

    routine_m, launch_m = chosen["returning_established"], chosen["launch_slot"]
    reasons_r = explain_matrix(routine_m, ctx, aug, tuned, content, products_raw, bundle)
    reasons_l = explain_matrix(launch_m, ctx, aug, tuned, content, products_raw, bundle)
    rows = []
    for u, a in enumerate(ctx.users):
        for j, pid in enumerate(ctx.products):
            rows.append({"client_id": a.removeprefix("synthetic_"), "product_id": pid,
                         "routine_score": float(matrix(routine_m)[u, j]), "launch_score": float(matrix(launch_m)[u, j]),
                         "reason": reasons_r[u, j], "launch_reason": reasons_l[u, j]})
    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["routine_score"])]
    write_export(df, path("app_data") / "returning_scores.parquet")

    price_map = products_raw.set_index("product_id")["price_usd"]
    info = {}
    for c in clients:
        if c["type"] == "returning":
            info[c["id"]] = {
                "usual_spend": float(np.median([price_map[h["product_id"]] for h in c["history"]])),
                "routine_model": routine_m, "routine_model_label": model_label(routine_m),
                "launch_model": launch_m, "launch_model_label": model_label(launch_m),
            }
    log(f"Returning clients scored with {routine_m} (routine) and {launch_m} (launch slot)")
    return info
