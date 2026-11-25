"""End-to-end check of the Streamlit pages with the real app_data (skipped if not built)."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "app_data" / "products.parquet").exists(), reason="app_data not built")

from streamlit.testing.v1 import AppTest  # noqa: E402

CLIENTS = ["claire", "ines", "sophie", "emma", "margaux", "hannah"]


def _advisor(cid: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    at.selectbox[0].set_value(cid).run()
    assert not at.exception, at.exception[0].message
    return at


@pytest.mark.parametrize("cid", CLIENTS)
def test_every_client_runs_end_to_end(cid):
    at = _advisor(cid)
    assert any("SYNTHETIC PROFILE" in c.value for c in at.caption)
    assert any("Not affiliated" in c.value for c in at.caption)


def test_no_consent_client_gets_no_message():
    at = _advisor("margaux")
    assert any("No marketing consent" in w.value for w in at.warning)
    assert len(at.text_area) == 0


def test_impossible_budget_names_the_constraint():
    at = _advisor("hannah")
    assert any("above the budget" in e.value for e in at.error)


def test_consenting_client_gets_checked_draft():
    at = _advisor("claire")
    assert len(at.text_area) == 1
    assert any("Checks passed" in s.value for s in at.success)


def test_edited_draft_with_claim_is_blocked():
    at = _advisor("claire")
    at.text_area[0].set_value(at.text_area[0].value + "\nThis cream heals your skin, only $5.00.").run()
    assert any("Blocked until fixed" in e.value for e in at.error)


@pytest.mark.parametrize("page", ["1_Brand_owner.py", "2_Results.py"])
def test_other_pages_render(page):
    if page == "2_Results.py" and (ROOT / "app_data" / "results_summary.json").read_text().strip() == "{}":
        pytest.skip("results summary not built yet")
    at = AppTest.from_file(str(ROOT / "app" / "pages" / page), default_timeout=120)
    at.run()
    assert not at.exception, at.exception[0].message
