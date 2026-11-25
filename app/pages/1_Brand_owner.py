"""Brand-owner view: the same engine restricted to one brand, and the brand's routine gaps."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Sillage · Brand owner", page_icon="✦", layout="centered")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import data, footer, money  # noqa: E402

from sillage import app_core  # noqa: E402
from sillage.routine import STEPS, Consultation, build_routine  # noqa: E402

D = data()
P = D.products
counts = P[~P["out_of_stock"]].groupby("brand_name")["step"].nunique()
full_brands = sorted(counts[counts == 3].index)

st.title("Brand owner view")
st.markdown(
    "A single maison sees the same recommendation engine **restricted to its own products**, "
    "and where its range cannot serve a client profile. Ranking uses the new-client consultation score."
)
brand = st.selectbox(f"Brand ({len(full_brands)} brands have an in-stock product in all three steps)", full_brands,
                     index=full_brands.index("CLINIQUE") if "CLINIQUE" in full_brands else 0)

st.subheader("Routine gaps")
gaps = app_core.brand_gaps(D, brand)
pivot = gaps.pivot(index="skin type", columns="step", values="products")[STEPS]
evid = gaps.pivot(index="skin type", columns="step", values="with skin-type evidence")[STEPS]
st.caption("In-stock products not flagged as disliked by that skin type. 0 = the brand cannot fill that step.")
st.dataframe(pivot, use_container_width=True)
st.caption(f"Of those, products with at least {D.min_n} reviews from that skin type (evidence the advisor can cite):")
st.dataframe(evid, use_container_width=True)
holes = gaps[gaps["products"] == 0]
thin = gaps[(gaps["products"] > 0) & (gaps["with skin-type evidence"] == 0)]
if len(holes):
    st.warning("Gaps: " + "; ".join(f"no {r.step} for {r._1} skin" for r in holes.itertuples()))
if len(thin):
    st.info("Thin evidence: " + "; ".join(f"{r.step} for {r._1} skin" for r in thin.itertuples()))
if not len(holes) and not len(thin):
    st.success("No gaps: every skin type can get a full routine with skin-type evidence.")

st.subheader("Try a profile")
skin = st.selectbox("Skin type", ["dry", "normal", "combination", "oily"])
budget = st.number_input("Total budget (USD)", 10.0, 2000.0, 200.0, 10.0)
c = Consultation(skin_type=skin, concerns=[], budget=budget, brand=brand)
est = P[~P["is_new"]]
r = build_routine(est, est.set_index("product_id")["pop_score"], c, app_core.guarded(D, skin), D.min_n)
if r.picks is None:
    st.error("No routine from this brand passes every filter:\n\n" + "\n\n".join(r.failure))
    st.dataframe(r.funnel, hide_index=True, use_container_width=True)
else:
    for _, p in r.picks.iterrows():
        with st.container(border=True):
            st.markdown(f"**{p['step'].capitalize()}** · {money(p['price_usd'])}  \n*{p['product_name']}*")
            st.caption("• " + app_core.skin_reason(D, p["product_id"], skin).text)
    st.markdown(f"**Total {money(r.total)}**")

footer()
