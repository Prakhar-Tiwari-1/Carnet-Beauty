from __future__ import annotations

import datetime as dt

import pytest

from sillage.messaging import compose as compose_mod
from sillage.messaging import llm
from sillage.messaging.compose import compose
from sillage.messaging.guardrails import Facts


def _facts():
    return Facts(
        client_name="Inès", language="fr", skin_type="oily",
        products=[{"product_name": "Daily Gel", "brand_name": "X", "step": "cleanser", "price_usd": 20.0}],
        total=20.0,
    )


def _exploding_factory():
    raise AssertionError("An LLM client must never be created without consent")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(compose_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "settings", lambda: {"LLM_API_KEY": "k", "LLM_MODEL": "m", "LLM_BASE_URL": None})


def test_no_consent_no_message_no_llm(tmp_path):
    d = compose(_facts(), consent=False, catalogue_names=[], use_llm=True, client_factory=_exploding_factory)
    assert d.text is None and d.source == "none"
    assert "no AI call" in d.notices[0]
    assert list(tmp_path.iterdir()) == []          # nothing cached, nothing counted


def test_consent_without_llm_uses_template():
    d = compose(_facts(), consent=True, catalogue_names=[], use_llm=False, client_factory=_exploding_factory)
    assert d.source == "template" and d.check.ok
    assert d.text.startswith("Bonjour Inès")


class _FakeClient:
    def __init__(self, reply):
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls += 1
                msg = type("M", (), {"content": reply})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_llm_draft_cached_and_capped():
    fake = _FakeClient("Bonjour Inès, voici X, Daily Gel (20,00 $). Total : 20,00 $.")
    day = dt.date(2026, 1, 1)
    d1 = compose(_facts(), True, [], use_llm=True, client_factory=lambda: fake, today=day)
    d2 = compose(_facts(), True, [], use_llm=True, client_factory=lambda: fake, today=day)
    assert d1.source == "llm" and d2.source == "llm (cached)" and fake.calls == 1


def test_session_cap_falls_back_to_template():
    d = compose(_facts(), True, [], use_llm=True, session_calls=99, client_factory=_exploding_factory)
    assert d.source == "template" and "Session limit" in d.notices[0]


def test_blocked_llm_draft_falls_back_with_reasons():
    fake = _FakeClient("Bonjour Inès, ce gel guérit la peau, seulement 9,99 $.")
    d = compose(_facts(), True, [], use_llm=True, client_factory=lambda: fake, today=dt.date(2026, 1, 2))
    assert d.source == "template"
    assert any("guérit" in n for n in d.notices) and any("9,99" in n for n in d.notices)
