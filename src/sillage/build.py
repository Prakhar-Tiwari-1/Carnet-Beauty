"""One command rebuilds everything: python -m sillage.build [--stage NAME]."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from sillage.config import load_config

STAGES = ["step0", "data", "eval_val", "train", "study", "test", "export", "review_sheet"]


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


def _model_inputs(ctx, cfg):
    """Shared by train and test: kNN fitted before the period cutoff, candidates, features."""
    from sillage.models import popularity, ranker
    from sillage.models.itemknn import ItemKNN

    knn = ItemKNN(cfg["itemknn"]["shrink"]).fit(ctx.reviews, ctx.period.cutoff)
    knn_sum, knn_max = knn.score(knn.liked_indices(ctx.users), ctx.products)
    pop = popularity.score(ctx, cfg["seed"])
    cand = ranker.candidates(ctx, pop, knn_sum, cfg["ranker"]["candidates_per_model"])
    rows, cols = ranker.pairs(cand)
    hs = cfg["hand_score"]
    X = ranker.build_features(ctx, knn_sum, knn_max, rows, cols, hs["m"], hs["guard_gap"], hs["guard_min_n"])
    y = ctx.target[rows, cols].astype(int)
    return knn, knn_sum, pop, cand, rows, cols, X, y


def stage_train() -> None:
    """Item-kNN and LightGBM ranker, validated across periods; chooses the returning-client model.

    Ranker: labels from T1, early stopping on T2, selection and ablation on B.
    Final ranker for the test: trained on B labels with the selected settings.
    """
    import numpy as np
    import pandas as pd

    from sillage.config import path
    from sillage.data.splits import Period, get_periods
    from sillage.eval.context import build_context
    from sillage.eval.metrics import evaluate_scores
    from sillage.eval.report import compare, to_table
    from sillage.models import hand_score, popularity, ranker

    cfg = load_config()
    seed, hs, k = cfg["seed"], cfg["hand_score"], cfg["eval"]["k"]
    reviews, catalogue, prices = _load_inputs()
    rc = cfg["ranker"]
    ctxs = {
        "T1": build_context(reviews, catalogue, prices, Period("T1", *map(pd.Timestamp, rc["t1"]))),
        "T2": build_context(reviews, catalogue, prices, Period("T2", *map(pd.Timestamp, rc["t2"]))),
        "B": get_periods()["B"],
    }
    ctxs["B"] = build_context(reviews, catalogue, prices, ctxs["B"])
    inputs = {name: _model_inputs(ctx, cfg) for name, ctx in ctxs.items()}
    data = {name: (v[6], v[7], v[4]) for name, v in inputs.items()}       # X, y, rows
    ctx = ctxs["B"]
    knn, knn_sum, pop, cand, rows, cols, X, y = inputs["B"]

    for name, c in ctxs.items():
        cand_n = inputs[name][3]
        ct = c.slice_target("catalogue")
        print(f"{name}: {len(c.users):,} clients, candidates/client {cand_n.sum(1).mean():.0f}, "
              f"catalogue positives covered by candidates {(ct & cand_n).sum() / ct.sum():.1%}")

    def b_scores(model, features):
        return ranker.to_matrix(model, X, rows, cols, ctx.seen.shape, features)

    grid = []
    for nl in rc["grid"]["num_leaves"]:
        for mcs in rc["grid"]["min_child_samples"]:
            p = ranker.params(nl, mcs, seed)
            model = ranker.train(data["T1"], p, ranker.ALL_FEATURES, data_stop=data["T2"])
            e = evaluate_scores(ctx, b_scores(model, ranker.ALL_FEATURES), "catalogue", k)
            grid.append({"num_leaves": nl, "min_child_samples": mcs, "rounds": model.best_iteration,
                         "ndcg@10": e["ndcg"].mean(), "recall@10": e["recall"].mean(), "_model": model})
            print(f"  ranker num_leaves={nl} min_child={mcs} rounds={model.best_iteration} "
                  f"B catalogue ndcg {e['ndcg'].mean():.4f}")
    best = max(grid, key=lambda g: g["ndcg@10"])
    best_model, best_p = best["_model"], ranker.params(best["num_leaves"], best["min_child_samples"], seed)

    ablation = [{"dropped": "none", "ndcg@10": best["ndcg@10"], "recall@10": best["recall@10"]}]
    for group, feats in ranker.GROUPS.items():
        keep = [f for f in ranker.ALL_FEATURES if f not in feats]
        m = ranker.train(data["T1"], best_p, keep, data_stop=data["T2"])
        e = evaluate_scores(ctx, b_scores(m, keep), "catalogue", k)
        ablation.append({"dropped": group, "ndcg@10": e["ndcg"].mean(), "recall@10": e["recall"].mean()})
    ablation = pd.DataFrame(ablation)
    ablation["Δndcg"] = ablation["ndcg@10"] - ablation["ndcg@10"].iloc[0]
    print("Ablation (trained on T1, evaluated on B, catalogue slice):")
    print(ablation.round(4).to_string(index=False))

    jitter = np.random.default_rng(seed).random(ctx.seen.shape) * 1e-6
    scores = {
        "random": popularity.random_scores(ctx, seed),
        "popularity": pop,
        "hand_v2_consultation": hand_score.score_v2(ctx, hs["m"], hs["guard_gap"], hs["guard_min_n"], seed),
        "itemknn": knn_sum + jitter,
        "ranker": b_scores(best_model, ranker.ALL_FEATURES),
    }
    results = compare(ctx, scores)
    print(to_table(results).to_string(index=False))
    head = {n: m["ndcg"]["mean"] for n, m in results["catalogue"]["models"].items() if n != "random"}
    returning = max(head, key=head.get)
    print(f"Returning-client model (best B catalogue NDCG@10): {returning}")

    final = ranker.train(data["B"], best_p, ranker.ALL_FEATURES, fixed_rounds=best["rounds"])
    (path("results") / "models").mkdir(parents=True, exist_ok=True)
    final.save_model(str(path("results") / "models" / "ranker.txt"))
    importance = pd.Series(final.feature_importance("gain"), index=ranker.ALL_FEATURES)
    out = {
        "protocol": "ranker trained on T1 labels, early-stopped on T2, compared on all B clients",
        "candidate_coverage_B": {
            sl: float((ctx.slice_target(sl) & cand).sum() / ctx.slice_target(sl).sum()) for sl in ("catalogue", "launch")
        },
        "ranker_grid": [{k2: v for k2, v in g.items() if k2 != "_model"} for g in grid],
        "ranker_params": {"num_leaves": best["num_leaves"], "min_child_samples": best["min_child_samples"],
                          "rounds": best["rounds"]},
        "ablation": ablation.to_dict("records"),
        "feature_gain_share": (importance / importance.sum()).round(4).to_dict(),
        "results": results,
        "returning_model": returning,
    }
    (path("results") / "val_models.json").write_text(json.dumps(out, indent=2, default=float))
    print("Wrote results/val_models.json and results/models/ranker.txt")


def stage_study() -> None:
    """Extended model study on rolling validation folds; writes results/model_study.json."""
    from sillage import study
    from sillage.config import path

    reviews, catalogue, prices = _load_inputs()
    out = study.run(reviews, catalogue, prices, log=lambda m: print(m, flush=True))
    (path("results") / "model_study.json").write_text(json.dumps(out, indent=2, default=float))
    for key, sel in out["selection"].items():
        print(f"Selection [{key}] (baseline {sel['baseline']}): chosen = {sel['chosen']}")
        print(pd_table(sel["table"]))
    print("Wrote results/model_study.json")


def pd_table(rows):
    import pandas as pd

    return pd.DataFrame(rows).round(4).to_string(index=False)


def stage_test(rerun_reason: str | None = None) -> None:
    """Touch the test period C. A lock file refuses reruns unless a reason is given; every run is logged.

    Evaluates every model from the study (tuned on validation folds) and records the
    pre-registered selection made on those folds next to the test outcome.
    """
    import datetime as dt
    import shutil

    from sillage import study
    from sillage.config import path

    lock, log = path("results") / "TEST_LOCK", path("results") / "test_runs.log"
    if lock.exists() and not rerun_reason:
        raise SystemExit(f"Test period already evaluated ({lock.read_text().strip()}). "
                         "Rerun only with --rerun-reason \"...\"; the reason is logged.")
    res_file = path("results") / "results.json"
    first = path("results") / "results_first_run.json"
    if res_file.exists() and not first.exists():
        shutil.copy(res_file, first)

    sel = json.loads((path("results") / "model_study.json").read_text())
    reviews, catalogue, prices = _load_inputs()
    out = study.test_run(reviews, catalogue, prices, sel["tuned"], log=lambda m: print(m, flush=True))
    out["selection_on_validation"] = {k: v["chosen"] for k, v in sel["selection"].items()}
    out["test_period"] = "2022-12-01 to 2023-03-21"
    out["runs"] = (log.read_text().splitlines() if log.exists() else [])
    for sl in ("catalogue", "launch"):
        base = "popularity" if sl == "catalogue" else "random"
        rows = [{"model": n, "ndcg": m["ndcg"]["mean"], "diff": m["ndcg"]["diff_vs_baseline"],
                 "diff_ci": m["ndcg"]["diff_ci"], "recall": m["recall"]["mean"]}
                for n, m in out["results"][sl]["models"].items()]
        print(f"Test, {sl} slice (vs {base}):")
        print(pd_table(sorted(rows, key=lambda r: -r["ndcg"])))
    res_file.write_text(json.dumps(out, indent=2, default=float))
    stamp = f"evaluated {dt.datetime.now().isoformat(timespec='seconds')}"
    lock.write_text(stamp)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{stamp} | reason: {rerun_reason or 'first run'}\n")
    print("Wrote results/results.json (first run kept in results_first_run.json); test period locked.")


def _rows(res_slice: dict, labels) -> list[dict]:
    out = []
    for n, m in res_slice["models"].items():
        out.append({
            "model": n, "label": labels(n),
            "ndcg": m["ndcg"]["mean"], "ci": m["ndcg"]["ci"],
            "diff": m["ndcg"]["diff_vs_baseline"], "diff_ci": m["ndcg"]["diff_ci"],
            "recall": m["recall"]["mean"], "recall_ci": m["recall"]["ci"],
            "coverage": m["coverage"], "bestseller_share": m["bestseller_share"],
        })
    return sorted(out, key=lambda r: -r["ndcg"])


def _write_results_md(s: dict) -> None:
    """docs/results.md, generated from the same summary the app reads (no hand-typed numbers)."""
    from sillage.config import ROOT

    def table(rows, base):
        out = ["| Model | NDCG@10 [95% CI] | Difference vs " + base + " [95% CI] | Recall@10 | Coverage |",
               "|---|---|---|---|---|"]
        for r in rows:
            if r["model"].endswith("@300"):
                continue
            out.append(f"| {r['label']} | {r['ndcg']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | "
                       f"{r['diff']:+.4f} [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}] | "
                       f"{r['recall']:.4f} | {r['coverage']:.1%} |")
        return "\n".join(out)

    def fold_table(rows):
        names = s["fold_names"]
        out = ["| Model | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for r in sorted(rows, key=lambda r: -sum(r.get(n, 0) for n in names)):
            out.append(f"| {r['label']} | " + " | ".join(f"{r.get(n, float('nan')):.4f}" for n in names) + " |")
        return "\n".join(out)

    md = [
        "# Results",
        "",
        "*Generated by `python -m sillage.build --stage export` from `results/`. Do not edit by hand.*",
        "",
        f"Test period: **{s['test_period']}**. Target: a product a returning client reviewed 4-5 stars and had "
        "not reviewed before (a proxy for a purchase; no sales impact is claimed). Free-sample reviews excluded.",
        "",
        "## Selection (made on validation folds before the final test run)",
        f"Rule: *{s['rule']}*.",
        f"- Established products for returning clients: **{s['selection']['returning_established']}**",
        f"- Launch slot for returning clients: **{s['selection']['launch_slot']}**",
        "",
        "## Test period: established products (baseline: popularity)",
        table(s["test"]["catalogue"], "popularity"),
        "",
        "## Test period: launches (baseline: random)",
        table(s["test"]["launch"], "random"),
        "",
        "## Validation folds: NDCG@10, established products",
        fold_table(s["folds"]["catalogue"]),
        "",
        "## Validation folds: NDCG@10, launches",
        fold_table(s["folds"]["launch"]),
        "",
        "## Sensitivity: free-sample reviews kept as positives (test period)",
        table(s["sensitivity"]["catalogue"], "popularity"),
        "",
        "## First test run (single-period selection, kept on purpose)",
        table(s["first_run"]["catalogue"], "popularity"),
        "",
        "Test runs log:",
        *[f"- {r}" for r in s["runs"]],
        "",
        "## Error analysis (recall@10, test period; groups under 20 clients hidden)",
        "| Dimension | Group | Clients | " + " | ".join(k for k in s["error_slices"][0] if k not in ("dimension", "group", "clients")) + " |",
        "|---|---|---|" + "---|" * (len(s["error_slices"][0]) - 3),
        *[f"| {r['dimension']} | {r['group']} | {r['clients']} | " + " | ".join(
            "hidden" if r[k] is None else f"{r[k]:.3f}" for k in r if k not in ("dimension", "group", "clients")) + " |"
          for r in s["error_slices"]],
        "",
        "## Ablation of the first ranker (trained on T1, evaluated on B)",
        "| Dropped group | NDCG@10 | Change |", "|---|---|---|",
        *[f"| {a['dropped']} | {a['ndcg@10']:.4f} | {a['Δndcg']:+.4f} |" for a in s["ablation_first_ranker"]],
        "",
        f"Tuned settings: `{json.dumps(s['tuned'])}`. Candidate coverage on the test period: "
        f"{s['candidate_coverage']['catalogue']:.1%} of established-product positives, "
        f"{s['candidate_coverage']['launch']:.1%} of launch positives.",
    ]
    (ROOT / "docs" / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def stage_export() -> None:
    """Write app_data/ (checked by export.check_export) for the deployed app."""
    from sillage import app_data
    from sillage.config import path

    reviews, catalogue, prices = _load_inputs()
    clients = app_data.load_clients()
    prod, stats = app_data.base_exports(reviews, catalogue)
    info = app_data.build_returning(reviews, catalogue, prices, clients)
    app_data.write_meta({"returning_clients": info})

    res = json.loads((path("results") / "results.json").read_text())
    study = json.loads((path("results") / "model_study.json").read_text())
    first = json.loads((path("results") / "results_first_run.json").read_text())
    val = json.loads((path("results") / "val_models.json").read_text())
    label = app_data.model_label
    folds = {}
    for sl in ("catalogue", "launch"):
        rows = {}
        for fname, f in study["per_fold"].items():
            for n, m in f[sl]["models"].items():
                rows.setdefault(n, {"model": n, "label": label(n)})[fname] = m["ndcg"]["mean"]
        folds[sl] = list(rows.values())
    summary = {
        "test_period": res["test_period"],
        "runs": res["runs"],
        "selection": res["selection_on_validation"],
        "selection_tables": {k: v["table"] for k, v in study["selection"].items()},
        "rule": study["rule"],
        "test": {sl: _rows(res["results"][sl], label) for sl in ("catalogue", "launch")},
        "sensitivity": {sl: _rows(res["sensitivity_sampled_kept"][sl], label) for sl in ("catalogue", "launch")},
        "folds": folds,
        "fold_names": list(study["per_fold"]),
        "first_run": {sl: _rows(first["results"][sl], label) for sl in ("catalogue", "launch")},
        "error_slices": res["error_slices"],
        "candidate_coverage": res["candidate_coverage"],
        "ablation_first_ranker": val["ablation"],
        "tuned": study["tuned"],
    }
    (path("app_data") / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_results_md(summary)
    for f in sorted(path("app_data").iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KB")


def stage_review_sheet() -> None:
    """30 follow-up drafts for manual labelling: consenting clients x EN/FR x 3 budget variants."""
    import dataclasses

    import pandas as pd

    from sillage import app_core
    from sillage.config import ROOT
    from sillage.messaging import llm
    from sillage.messaging.compose import compose

    d = app_core.load(ROOT)
    names = d.products["product_name"].tolist()
    use_llm = llm.is_configured()
    rows = []
    for client in [c for c in d.clients if c["consent"]]:
        variants, seen = [], set()
        for mult in [1, 1.5, 2, 3, 5, 8]:
            c = app_core.consultation(client, budget=float(client["budget_usd"]) * mult)
            view = app_core.build_view(d, client, c)
            if view.routine.picks is None:
                continue
            key = tuple(view.routine.picks["product_id"])
            if key in seen:
                continue
            seen.add(key)
            variants.append((mult, view))
            if len(variants) == 3:
                break
        for lang in ["en", "fr"]:
            for mult, view in variants:
                facts = app_core.message_facts(d, client, view, lang)
                draft = compose(facts, client["consent"], names, use_llm=use_llm)
                rows.append({
                    "id": len(rows) + 1, "client": client["name"], "language": lang, "budget_multiplier": mult,
                    "source": draft.source, "draft": draft.text,
                    "facts": json.dumps(dataclasses.asdict(facts), ensure_ascii=False),
                    "auto_check": "pass" if draft.check.ok else "blocked",
                    "auto_problems": " | ".join(draft.check.problems + draft.notices),
                    "factual_ok": "", "tone_ok": "", "compliant": "", "would_send": "", "notes": "",
                })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "docs" / "review_sheet.csv", index=False)
    print(f"Wrote docs/review_sheet.csv: {len(df)} drafts, sources {df['source'].value_counts().to_dict()}, "
          f"auto-check {df['auto_check'].value_counts().to_dict()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sillage.build")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--rerun-reason", default=None, help="required to evaluate the test period again")
    args = parser.parse_args(argv)
    _seed()
    for name in STAGES if args.stage == "all" else [args.stage]:
        print(f"== stage: {name}")
        if name == "test":
            stage_test(args.rerun_reason)
        else:
            globals()[f"stage_{name}"]()


if __name__ == "__main__":
    main()
