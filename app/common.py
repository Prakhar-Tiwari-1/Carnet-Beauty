"""Shared helpers for the Streamlit pages."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from sillage import app_core  # noqa: E402

FOOTER = (
    "Independent portfolio project. Not affiliated with, endorsed by or connected to Sephora, "
    "LVMH or L'Oréal. All clients are synthetic. Data: *Sephora Products and Skincare Reviews* by "
    "nadyinky (Kaggle, CC BY 4.0), scraped March 2023. A 4-5 star review is used as a proxy for a "
    "purchase; no sales impact is claimed."
)


@st.cache_resource
def data() -> app_core.AppData:
    return app_core.load(ROOT)


def footer() -> None:
    st.divider()
    st.caption(FOOTER)


def money(x: float) -> str:
    return f"${x:,.2f}"
