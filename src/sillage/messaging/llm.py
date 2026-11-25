"""Optional live drafting through any OpenAI-compatible endpoint.

Configuration (environment variables or Streamlit secrets, never in the repo):
  LLM_API_KEY, LLM_MODEL, LLM_BASE_URL (optional; default OpenAI)
Without LLM_API_KEY the app uses templates.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from sillage.messaging.guardrails import Facts

PROMPT_VERSION = "v1"

SYSTEM = """You write short follow-up messages from a beauty advisor to a client after an in-store consultation.
Rules:
- Use ONLY the facts in the JSON. Do not add any product, price, number, ingredient, benefit or claim.
- Write prices exactly as given in the JSON field "price_text" and the total as "total_text".
- Mention every product once, with its brand and exact name.
- No medical words (cure, heal, treat a condition, eczema, rosacea...).
- No environmental claims (clean, green, eco-friendly, sustainable...).
- Warm, discreet, luxury tone. At most 120 words. Sign with the advisor name.
- Write in {language_name}."""


def settings() -> dict:
    get = os.environ.get
    secrets = {}
    try:  # Streamlit secrets take precedence when a secrets file exists (never shows a warning)
        from streamlit.runtime.secrets import secrets_singleton

        if secrets_singleton.load_if_toml_exists():
            secrets = dict(secrets_singleton)
    except Exception:
        pass
    return {k: secrets.get(k, get(k)) for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL")}


def is_configured() -> bool:
    s = settings()
    return bool(s["LLM_API_KEY"] and s["LLM_MODEL"])


def facts_payload(facts: Facts) -> dict:
    from sillage.messaging.templates import price

    d = asdict(facts)
    for p in d["products"]:
        p["price_text"] = price(p["price_usd"], facts.language)
    d["total_text"] = price(facts.total, facts.language)
    return d


def make_client():
    from openai import OpenAI

    s = settings()
    return OpenAI(api_key=s["LLM_API_KEY"], base_url=s["LLM_BASE_URL"] or None)


def draft(facts: Facts, client=None, temperature: float = 0.2) -> str:
    s = settings()
    client = client or make_client()
    language_name = "French" if facts.language == "fr" else "English"
    resp = client.chat.completions.create(
        model=s["LLM_MODEL"],
        temperature=temperature,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM.format(language_name=language_name)},
            {"role": "user", "content": json.dumps(facts_payload(facts), ensure_ascii=False)},
        ],
    )
    return resp.choices[0].message.content.strip()
