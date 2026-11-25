"""Results for a non-technical manager: which method, how much better, where it fails, limits."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sillage · Results", page_icon="✦", layout="centered")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import data, footer  # noqa: E402

S = data().results


def lookup(rows, model):
    return next((r for r in rows if r["model"] == model), None)


def chart(rows, baseline_name: str, title: str):
    df = pd.DataFrame(rows)
    df = df[~df["model"].str.endswith("@100")]
    df["low"], df["high"] = df["ci"].str[0], df["ci"].str[1]
    df["kind"] = df["model"].map(lambda m: "baseline" if m in (baseline_name, "random") else "model")
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar().encode(
        x=alt.X("ndcg:Q", title="NDCG@10 (higher is better)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["model", "baseline"], range=["#7a4b5e", "#b9b1ad"]), legend=None),
        tooltip=["label", alt.Tooltip("ndcg:Q", format=".4f"), alt.Tooltip("recall:Q", format=".3f", title="recall@10")],
    )
    err = base.mark_errorbar(ticks=True).encode(x="low:Q", x2="high:Q")
    st.altair_chart((bars + err).properties(title=title, height=28 * len(df)), use_container_width=True)


cat, lau = S["test"]["catalogue"], S["test"]["launch"]
sel_r, sel_l = S["selection"]["returning_established"], S["selection"]["launch_slot"]
dep = S.get("deployed", S["selection"])
pop, rnd_l = lookup(cat, "popularity"), lookup(lau, "random")
picked_r, best_l = lookup(cat, sel_r), lookup(lau, dep["launch_slot"])
top_r = max((r for r in cat if not r["model"].endswith("@100")), key=lambda r: r["ndcg"])

st.title("Results")
st.markdown(f"Test period **{S['test_period']}**, never used to choose or tune anything except where stated below. "
            "The target is a product a returning client reviewed 4-5 stars and had not reviewed before "
            "(a proxy for a purchase; no sales impact is claimed).")

st.subheader("In one minute")
lines = []
if dep["returning_established"] == "popularity" and picked_r:
    lines.append(f"**Established products:** the model picked on validation (*{picked_r['label']}*) did **not** beat "
                 f"popularity on the test period (NDCG@10 {picked_r['ndcg']:.3f} vs {pop['ndcg']:.3f}; difference "
                 f"{picked_r['diff']:+.3f}, 95% CI {picked_r['diff_ci'][0]:+.3f} to {picked_r['diff_ci'][1]:+.3f}). "
                 "**The app therefore uses popularity** for returning clients' routines.")
    if top_r["model"] not in ("popularity", sel_r):
        lines.append(f"**Open finding:** *{top_r['label']}* was best on the test period (NDCG@10 {top_r['ndcg']:.3f}, "
                     f"{top_r['ndcg'] / pop['ndcg']:.1f}× popularity) but not on validation. Choosing it now would "
                     "mean the test period picked the model, so it is not deployed; it is the first thing a pilot should test.")
elif picked_r:
    lines.append(f"**Established products:** *{picked_r['label']}* NDCG@10 {picked_r['ndcg']:.3f} vs popularity "
                 f"{pop['ndcg']:.3f} (difference {picked_r['diff']:+.3f}, 95% CI {picked_r['diff_ci'][0]:+.3f} to "
                 f"{picked_r['diff_ci'][1]:+.3f}).")
if best_l and dep["launch_slot"] != "random":
    lines.append(f"**New launches:** *{best_l['label']}* (picked on validation) finds the launches a client will like "
                 f"**{best_l['recall'] / rnd_l['recall']:.1f}× more often than random** on the test period (recall@10 "
                 f"{best_l['recall']:.3f} vs {rnd_l['recall']:.3f}). Popularity and purchase-history models "
                 "know nothing about a product with no reviews.")
lines.append("**New clients** (no history) get the consultation score: popularity, minus products their skin type "
             "clearly dislikes. Ranking by average rating instead performs no better than random, because almost "
             "every product is rated 80-90% positive.")
for l in lines:
    st.markdown("- " + l)

st.subheader("Established products (test period)")
chart(cat, "popularity", "Returning clients, products already on sale")
st.caption("Bars: mean over all returning clients in the test period; lines: 95% bootstrap CI over clients. "
           "Grey = baselines.")

st.subheader("New launches (test period)")
chart(lau, "random", "Returning clients, products launched during the test period")

st.subheader("How the model was chosen")
st.markdown(
    f"Every method was tuned and compared on **three earlier periods** ({', '.join(S['fold_names'])}), "
    f"then checked once on the test period. Pre-registered rule: *{S['rule']}*. "
    "Methods tried: popularity, item-kNN, RP3beta, EASE, iALS and BPR matrix factorisation (trained on GPU), "
    "content similarity (TF-IDF), and LightGBM learning-to-rank in three variants."
)
st.dataframe(pd.DataFrame(S["selection_tables"]["returning_established"]).rename(columns={
    "mean_ndcg": "mean NDCG@10 (3 periods)", "worst_diff_ci_low": "worst CI low vs popularity",
    "passes_every_fold": "beats popularity in all 3"}), hide_index=True, use_container_width=True)

st.subheader("What went wrong first (kept on purpose)")
fr = S["first_run"]["catalogue"]
r1, p1 = lookup(fr, "ranker"), lookup(fr, "popularity")
if r1 and p1:
    st.markdown(
        f"A LightGBM ranker chosen on a single validation period lost to popularity on the test period "
        f"(NDCG@10 {r1['ndcg']:.3f} vs {p1['ndcg']:.3f}). Diagnosis: about half of its learning came from price "
        "and brand-level popularity, so it learned *last season's winners* rather than the client. That led to "
        "the rolling three-period protocol, the client-centric ranker variant, and a second, logged test run. "
        "In that second run the rankers again fell to popularity level on the test period, although they led on all "
        "three validation periods, while simpler models (EASE, RP3beta) rose. The cause is not established. "
        "A leakage check (refitting with every test-period row deleted gives identical scores) rules out a data bug.")
for run in S["runs"]:
    st.caption("• " + run)

st.subheader("Where it works and where it does not")
es = pd.DataFrame(S["error_slices"])
show = [c for c in dict.fromkeys(["popularity", top_r["model"]]) if c in es.columns]
if show:
    es = es[["dimension", "group", "clients", *show]]
    st.caption("Recall@10 by client and product group (groups with fewer than 20 clients hidden).")
    st.dataframe(es, hide_index=True, use_container_width=True)

st.subheader("Limits")
st.markdown(
    "- A review is a proxy for a purchase; free-sample reviews are excluded (a text rule; lower bound).\n"
    "- Prices and stock are March 2023 snapshots; products delisted before then are missing.\n"
    "- Skin-type figures are self-reported by reviewers. Concern tags are Sephora's marketing tags.\n"
    "- The ingredient filter is text matching, not an allergy or safety guarantee.\n"
    "- Results come from one retailer's US reviews; a luxury maison's clientele may behave differently. "
    "An A/B test is needed before any business claim (see the business case in the repository)."
)
footer()
