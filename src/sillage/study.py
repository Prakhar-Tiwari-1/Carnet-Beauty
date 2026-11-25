"""Extended model study over rolling validation folds, all before the test period C.

Protocol
1. Tune each model family on the three folds (mean catalogue NDCG@10).
2. For each fold, fit every tuned model at the fold's cutoff and score its clients.
3. Rankers are trained on the window just before the fold (features before that
   window's cutoff), with a fixed number of rounds, never early-stopped inside a period.
4. Compare every model with paired bootstrap CIs per fold.
5. Apply the pre-registered rule (config.study.selection_rule) to pick the
   returning-client model (catalogue slice) and the launch-slot model (launch slice).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from sillage import device
from sillage.config import load_config
from sillage.data.ingest import read_products
from sillage.data.splits import Period
from sillage.eval.context import build_context
from sillage.eval.metrics import evaluate_scores
from sillage.eval.report import compare
from sillage.models import hand_score, popularity, ranker
from sillage.models.cf import BPR, EASE, IALS, LikedMatrix, RP3beta
from sillage.models.content import ContentTFIDF
from sillage.models.itemknn import ItemKNN

CF_FAMILIES = ["itemknn", "rp3beta", "ease", "ials", "bpr"]

# Client-centric ranker: no static product-identity features (absolute price, brand
# prior, concern-tag count, price tier). Diagnosis on the first test run showed the
# original ranker learned "last season's winners" through those.
CLIENT_CENTRIC = [
    "pop_log", "rating_shrunk", "log_n_reviews", "reviews_90d_log", "days_since_first",
    "incentivised_share", "is_launch", "step_code", "log_price_ratio",
    "log_hist_len", "days_since_last", "mean_rating_given", "skin_code",
    "like_rate_skin", "concern_overlap", "brand_affinity", "skin_guard",
    "knn_sum", "knn_max", "cf_rp3beta", "cf_ease", "cf_ials", "cf_bpr", "cf_content",
]


def _jitter(shape, seed):
    return np.random.default_rng(seed).random(shape).astype(np.float32) * 1e-6


def _grid(spec: dict) -> list[dict]:
    keys = list(spec)
    return [dict(zip(keys, v)) for v in itertools.product(*(spec[k] for k in keys))]


class Scorer:
    """Fits CF models at a cutoff (cached liked matrix) and scores a context's clients."""

    def __init__(self, reviews, content: ContentTFIDF, seed: int):
        self.reviews, self.content, self.seed = reviews, content, seed
        self._liked: dict = {}

    def liked(self, cutoff) -> LikedMatrix:
        if cutoff not in self._liked:
            self._liked[cutoff] = LikedMatrix(self.reviews, cutoff)
        return self._liked[cutoff]

    def cf(self, family: str, params: dict, ctx) -> dict[str, np.ndarray]:
        cutoff, users, products = ctx.period.cutoff, ctx.users, ctx.products
        if family == "itemknn":
            knn = ItemKNN(params["shrink"]).fit(self.reviews, cutoff)
            s, mx = knn.score(knn.liked_indices(users), products)
            return {"itemknn": s, "itemknn_max": mx}
        m = self.liked(cutoff)
        model = {
            "rp3beta": lambda: RP3beta(params["alpha"], params["beta"]),
            "ease": lambda: EASE(params["lam"]),
            "ials": lambda: IALS(params["factors"], params["reg"], params["alpha"], iterations=10, seed=self.seed),
            "bpr": lambda: BPR(params["factors"], params["lr"], params["weight_decay"], epochs=20, seed=self.seed),
        }[family]()
        return {family: model.fit(m).score(users, products)}

    def all_scores(self, ctx, tuned: dict) -> dict[str, np.ndarray]:
        cfg = load_config()
        hs = cfg["hand_score"]
        out = {
            "random": popularity.random_scores(ctx, self.seed),
            "popularity": popularity.score(ctx, self.seed),
            "hand_v2_consultation": hand_score.score_v2(ctx, hs["m"], hs["guard_gap"], hs["guard_min_n"], self.seed),
            "content": self.content.score(self.reviews, ctx.period.cutoff, ctx.users, ctx.products),
        }
        for fam in CF_FAMILIES:
            out.update(self.cf(fam, tuned[fam], ctx))
        # Same random tie-break for every model, so "no information" ranks like random,
        # not like catalogue order (e.g. CF models on launches all score 0).
        jit = _jitter(ctx.seen.shape, self.seed)
        for name in out:
            if name not in ("random", "popularity", "hand_v2_consultation"):
                out[name] = out[name] + jit
        return out


def _candidates(ctx, scores: dict, k: int) -> np.ndarray:
    mask = np.zeros(ctx.seen.shape, bool)
    for name in ("popularity", "itemknn", "ease", "content"):
        s = np.where(ctx.seen | ctx.is_launch[None, :], -np.inf, scores[name] + _jitter(ctx.seen.shape, 1))
        np.put_along_axis(mask, np.argpartition(-s, kth=k, axis=1)[:, :k], True, axis=1)
    return (mask | ctx.is_launch[None, :]) & ~ctx.seen


def _ranker_frame(ctx, scores: dict, k: int):
    cfg = load_config()
    hs = cfg["hand_score"]
    cand = _candidates(ctx, scores, k)
    rows, cols = np.nonzero(cand)
    extra = {n: scores[n] for n in ("rp3beta", "ease", "ials", "bpr", "content")}
    X = ranker.build_features(ctx, scores["itemknn"], scores["itemknn_max"], rows, cols,
                              hs["m"], hs["guard_gap"], hs["guard_min_n"], extra=extra)
    return cand, rows, cols, X, ctx.target[rows, cols].astype(int)


VARIANTS = {
    "original": (ranker.ALL_FEATURES, "lambdarank"),
    "client_centric": (CLIENT_CENTRIC, "lambdarank"),
    "client_centric_binary": (CLIENT_CENTRIC, "binary"),
}


def train_rankers(train_frame, seed: int) -> dict:
    rc = load_config()["study"]["ranker"]
    _, rows, _, X, y = train_frame
    out = {}
    for v in rc["variants"]:
        feats, obj = VARIANTS[v]
        p = ranker.params(rc["num_leaves"], rc["min_child_samples"], seed, obj, device.LIGHTGBM_DEVICE)
        p["num_threads"] = 16
        out[v] = ranker.train((X, y, rows), p, feats, fixed_rounds=max(rc["rounds"]))
    return out


def ranker_scores(models: dict, eval_frame, shape) -> dict[str, np.ndarray]:
    rc = load_config()["study"]["ranker"]
    _, rows, cols, X, _ = eval_frame
    out = {}
    for v, model in models.items():
        feats = VARIANTS[v][0]
        for r in rc["rounds"]:
            s = np.full(shape, -np.inf, np.float32)
            s[rows, cols] = model.predict(X[feats], num_iteration=r)
            out[f"ranker_{v}@{r}"] = s
    return out


def tune(scorer: Scorer, fold_ctxs: list, log) -> tuple[dict, dict]:
    grids, k = load_config()["study"]["grids"], load_config()["eval"]["k"]
    tuned, report = {}, {}
    for fam in CF_FAMILIES:
        rows = []
        for params in _grid(grids[fam]):
            t = time.time()
            vals = []
            for ctx in fold_ctxs:
                s = scorer.cf(fam, params, ctx)[fam]
                vals.append(evaluate_scores(ctx, s + _jitter(s.shape, 0), "catalogue", k)["ndcg"].mean())
            rows.append({**params, "fold_ndcg": vals, "mean_ndcg": float(np.mean(vals))})
            log(f"  tune {fam} {params}: mean catalogue NDCG@10 {np.mean(vals):.4f} ({time.time() - t:.0f}s)")
        best = max(rows, key=lambda r: r["mean_ndcg"])
        tuned[fam] = {k2: best[k2] for k2 in grids[fam]}
        # Flag a winner on the edge of a grid that has 3+ values (optimum may lie outside).
        edges = [k2 for k2, vals in grids[fam].items() if len(vals) >= 3 and best[k2] in (min(vals), max(vals))]
        report[fam] = {"grid": rows, "best": tuned[fam], "best_on_grid_edge": edges}
        if edges:
            log(f"  NOTE {fam}: best value on grid edge for {edges}")
    return tuned, report


def select(per_fold: dict, slice_name: str, baseline: str, exclude: set) -> dict:
    """Pre-registered rule: CI of the difference vs baseline above zero in every fold; then best mean."""
    models = next(iter(per_fold.values()))[slice_name]["models"]
    table = []
    for name in models:
        if name in exclude or name == baseline:
            continue
        diffs = [f[slice_name]["models"][name]["ndcg"] for f in per_fold.values()]
        table.append({
            "model": name,
            "mean_ndcg": float(np.mean([d["mean"] for d in diffs])),
            "worst_diff_ci_low": float(min(d["diff_ci"][0] for d in diffs)),
            "passes_every_fold": all(d["diff_ci"][0] > 0 for d in diffs),
        })
    table.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    passing = [r for r in table if r["passes_every_fold"]]
    return {"baseline": baseline, "chosen": passing[0]["model"] if passing else baseline, "table": table}


def run(reviews, catalogue, prices, log=print) -> dict:
    cfg = load_config()
    seed, k = cfg["seed"], cfg["study"]["candidates_per_model"]
    log(f"Devices: {device.describe()}; LightGBM runs on {device.LIGHTGBM_DEVICE} (measured choice)")
    skincare = read_products()
    skincare = skincare[skincare["primary_category"] == "Skincare"]
    scorer = Scorer(reviews, ContentTFIDF(skincare), seed)

    folds = []
    for name, start, end, train_start in cfg["study"]["folds"]:
        ev = build_context(reviews, catalogue, prices, Period(name, pd.Timestamp(start), pd.Timestamp(end)))
        tr = build_context(reviews, catalogue, prices, Period(f"{name}_train", pd.Timestamp(train_start), pd.Timestamp(start)))
        folds.append((name, ev, tr))
        log(f"{name}: evaluate {start}..{end} ({len(ev.users):,} clients); ranker trained on {train_start}..{start} ({len(tr.users):,})")

    tuned, tuning = tune(scorer, [f[1] for f in folds], log)
    log(f"Tuned: {tuned}")

    per_fold, coverage = {}, {}
    for name, ev, tr in folds:
        t = time.time()
        s_ev, s_tr = scorer.all_scores(ev, tuned), scorer.all_scores(tr, tuned)
        models = train_rankers(_ranker_frame(tr, s_tr, k), seed)
        frame = _ranker_frame(ev, s_ev, k)
        s_ev.update(ranker_scores(models, frame, ev.seen.shape))
        cand = frame[0]
        coverage[name] = {sl: float((ev.slice_target(sl) & cand).sum() / ev.slice_target(sl).sum()) for sl in ("catalogue", "launch")}
        scores = {n: v for n, v in s_ev.items() if n != "itemknn_max"}
        per_fold[name] = {
            "catalogue": compare(ev, scores, slices=("catalogue",))["catalogue"],
            "launch": compare(ev, scores, slices=("launch",), baseline="random")["launch"],
        }
        log(f"{name} done ({time.time() - t:.0f}s); candidate coverage {coverage[name]}")

    selection = {
        "returning_established": select(per_fold, "catalogue", "popularity", exclude={"random", "hand_v2_consultation"}),
        "launch_slot": select(per_fold, "launch", "random", exclude={"popularity"}),
    }
    return {"tuned": tuned, "tuning": tuning, "per_fold": per_fold, "candidate_coverage": coverage,
            "selection": selection, "rule": cfg["study"]["selection_rule"]}


def test_run(reviews, catalogue, prices, tuned: dict, log=print) -> dict:
    """Evaluate every study model on the test period C (rankers trained on B labels)."""
    from sillage.data.splits import get_periods
    from sillage.eval.analysis import error_slices

    cfg = load_config()
    seed, k = cfg["seed"], cfg["study"]["candidates_per_model"]
    skincare = read_products()
    skincare = skincare[skincare["primary_category"] == "Skincare"]
    scorer = Scorer(reviews, ContentTFIDF(skincare), seed)
    periods = get_periods()

    def run(include_incentivised: bool):
        ev = build_context(reviews, catalogue, prices, periods["C"], include_incentivised)
        tr = build_context(reviews, catalogue, prices, periods["B"], include_incentivised)
        s_ev, s_tr = scorer.all_scores(ev, tuned), scorer.all_scores(tr, tuned)
        models = train_rankers(_ranker_frame(tr, s_tr, k), seed)
        frame = _ranker_frame(ev, s_ev, k)
        s_ev.update(ranker_scores(models, frame, ev.seen.shape))
        cover = {sl: float((ev.slice_target(sl) & frame[0]).sum() / ev.slice_target(sl).sum()) for sl in ("catalogue", "launch")}
        return ev, {n: v for n, v in s_ev.items() if n != "itemknn_max"}, cover

    ev, scores, cover = run(False)
    log(f"C: {len(ev.users):,} clients; candidate coverage {cover}")
    results = {
        "catalogue": compare(ev, scores, slices=("catalogue",))["catalogue"],
        "launch": compare(ev, scores, slices=("launch",), baseline="random")["launch"],
    }
    main = [n for n in scores if n not in ("random",) and not n.endswith("@100")]
    slices = error_slices(ev, {n: scores[n] for n in main})
    ev_s, scores_s, _ = run(True)
    sensitivity = {
        "catalogue": compare(ev_s, scores_s, slices=("catalogue",))["catalogue"],
        "launch": compare(ev_s, scores_s, slices=("launch",), baseline="random")["launch"],
    }
    return {"results": results, "candidate_coverage": cover, "error_slices": slices.to_dict("records"),
            "sensitivity_sampled_kept": sensitivity}
