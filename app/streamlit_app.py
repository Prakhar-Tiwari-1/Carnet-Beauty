"""Sillage: advisor view (home page)."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Sillage", page_icon="✦", layout="centered")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data, footer, money  # noqa: E402

from sillage import app_core  # noqa: E402
from sillage.messaging import llm  # noqa: E402
from sillage.messaging.compose import compose  # noqa: E402
from sillage.messaging.guardrails import catalogue_names, check_draft  # noqa: E402
from sillage.routine import INGREDIENT_DISCLAIMER, STEPS  # noqa: E402

D = data()
CONCERNS = sorted({c for cs in D.products["concerns"] for c in cs})
STEP_TITLE = {"cleanser": "1 · Cleanser", "treatment": "2 · Treatment", "moisturiser": "3 · Moisturiser"}

st.title("Sillage")
st.markdown(
    "**For beauty advisors:** pick a client, get an explained three-step routine and a follow-up draft "
    "that is checked before you approve it. Behind it: a model comparison on about a million real reviews "
    "(see *Results*)."
)

labels = {c["id"]: f"{c['name']} · {'returning' if c['type'] == 'returning' else 'new'} client" for c in D.clients}
cid = st.selectbox("Client", list(labels), format_func=labels.get)
client = D.client(cid)
st.caption("🔖 SYNTHETIC PROFILE: invented for this demo. No real person is represented.")

with st.expander("Consultation (editable)", expanded=client["type"] == "new"):
    skin = st.selectbox("Skin type", ["dry", "normal", "combination", "oily"],
                        index=["dry", "normal", "combination", "oily"].index(client["skin_type"]), key=f"skin_{cid}")
    concerns = st.multiselect("Stated concerns", CONCERNS, default=client["concerns"], key=f"conc_{cid}")
    budget = st.number_input("Total budget for the three steps (USD)", 5.0, 2000.0,
                             float(client["budget_usd"]), 5.0, key=f"bud_{cid}")
    avoid = st.text_input("Ingredients to avoid (comma separated)", ", ".join(client.get("avoid", [])), key=f"av_{cid}")
    usual_default = client.get("usual_spend_usd")
    usual = st.number_input(
        "Usual spend per product (USD, used to match new launches; 0 = not given)", 0.0, 1000.0,
        float(usual_default or 0), 5.0, key=f"us_{cid}",
        help="For returning clients, left at 0 the app uses the median price of her past purchases.")
    if client.get("history"):
        hist = D.products.set_index("product_id")
        names = [f"{hist.loc[h['product_id'], 'brand_name']}, {hist.loc[h['product_id'], 'product_name']}"
                 if h["product_id"] in hist.index else h["product_id"] for h in client["history"]]
        st.caption("Invented purchase history: " + "; ".join(names))

c = app_core.consultation(client, skin_type=skin, concerns=concerns, budget=budget,
                          avoid=[a.strip() for a in avoid.split(",") if a.strip()],
                          usual_spend=usual or None)
view = app_core.build_view(D, client, c)

st.info(view.path_label, icon="🧭")

st.subheader("Routine")
r = view.routine
for note in r.notes:
    st.caption(("⚠️ " if note == INGREDIENT_DISCLAIMER else "ℹ️ ") + note)
if r.picks is None:
    st.error("No routine passes every filter. Nothing was relaxed.\n\n" + "\n\n".join(r.failure))
    st.caption("Products left per step after each filter:")
    st.dataframe(r.funnel, hide_index=True, use_container_width=True)
else:
    for _, p in r.picks.iterrows():
        with st.container(border=True):
            st.markdown(f"**{STEP_TITLE[p['step']]}** · {money(p['price_usd'])}")
            st.markdown(f"{p['brand_name']}, *{p['product_name']}*")
            for reason in view.reasons.get(p["product_id"], []):
                st.caption("• " + reason.text)
    st.markdown(f"**Total {money(r.total)}** of a {money(c.budget)} budget. "
                "Chosen by exact search over every combination that fits the budget.")

st.subheader("New this season")
st.caption(view.launch_note)
if view.launches.empty:
    st.caption("No new product passes the same filters.")
for _, p in view.launches.head(3).iterrows():
    with st.container(border=True):
        st.markdown(f"**{p['step'].capitalize()}** · {money(p['price_usd'])} · not included in the routine budget")
        st.markdown(f"{p['brand_name']}, *{p['product_name']}*")
        for reason in view.reasons.get(p["product_id"], []):
            st.caption("• " + reason.text)

st.subheader("Follow-up message")
if not client["consent"]:
    st.warning("No marketing consent recorded for this client: no message is drafted and no AI call is made.")
elif r.picks is None:
    st.caption("No routine, so no message.")
else:
    lang = st.radio("Language", ["en", "fr"], index=["en", "fr"].index(client["language"]),
                    format_func={"en": "English", "fr": "Français"}.get, horizontal=True, key=f"lang_{cid}")
    use_ai = st.toggle("Draft with AI", value=False, disabled=not llm.is_configured(),
                       help=None if llm.is_configured() else "No AI key configured: templates only.")
    facts = app_core.message_facts(D, client, view, lang)
    names = catalogue_names(D.products)
    st.session_state.setdefault("ai_calls", 0)
    draft = compose(facts, client["consent"], names, use_llm=use_ai, session_calls=st.session_state["ai_calls"])
    if draft.source == "llm":
        st.session_state["ai_calls"] += 1
    for n in draft.notices:
        st.caption("ℹ️ " + n)
    st.caption(f"Draft source: {draft.source}. Edit freely; every edit is checked again.")
    key = f"msg_{cid}_{lang}_{hash(draft.text)}"
    text = st.text_area("Draft", draft.text, height=300, key=key)
    check = check_draft(text, facts, names)
    if check.ok:
        st.success("Checks passed: every product, price and figure matches the records; "
                   "no medical or environmental claims.")
    else:
        st.error("Blocked until fixed:\n\n" + "\n\n".join(f"- {p}" for p in check.problems))
    if st.button("Approve draft", disabled=not check.ok, type="primary"):
        st.success("Approved. This demo never sends anything.")

footer()
