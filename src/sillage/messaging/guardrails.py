"""Checks every draft (template, LLM or advisor-edited) before it can be approved.

1. Facts: every number in the draft (prices, totals, percentages, counts) must be
   one of the supplied fact numbers; list markers "1." "2." "3." at line start are allowed.
   Every catalogue product name mentioned must be one of the supplied products.
2. Lexicon: medical and unsupported environmental wording is blocked in EN and FR.
   Supplied product names and Sephora tags are masked first, so the real product
   "Green Clean Makeup Meltaway Cleansing Balm" does not trigger "green" or "clean".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from sillage.config import ROOT

_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
_LIST_MARKER = re.compile(r"(?m)^\s*\d\.\s")


@dataclass
class Facts:
    """Everything a draft may state. Built from the profile, chosen products and shown aggregates."""
    client_name: str
    language: str
    skin_type: str
    products: list[dict]                 # each: product_name, brand_name, step, price_usd
    total: float
    numbers: list[float] = field(default_factory=list)   # extra displayed figures (e.g. 87, 412)
    tags: list[str] = field(default_factory=list)        # Sephora tags that may be quoted
    advisor_name: str = "Your beauty advisor"

    def allowed_numbers(self) -> set[float]:
        vals = [p["price_usd"] for p in self.products] + [self.total] + list(self.numbers)
        return {round(float(v), 2) for v in vals}

    def protected_phrases(self) -> list[str]:
        names = [p["product_name"] for p in self.products] + [p["brand_name"] for p in self.products]
        return sorted(set(names) | set(self.tags), key=len, reverse=True)


@dataclass
class CheckResult:
    ok: bool
    problems: list[str]


@lru_cache(maxsize=1)
def lexicon() -> dict:
    with open(ROOT / "guardrails" / "lexicon.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mask(text: str, phrases: list[str]) -> str:
    for ph in phrases:
        if ph:
            text = re.sub(re.escape(ph), lambda m: " " * len(m.group(0)), text, flags=re.IGNORECASE)
    return text


def _to_float(token: str) -> float:
    return round(float(token.replace(",", ".")), 2)


def check_numbers(text: str, facts: Facts) -> list[str]:
    allowed = facts.allowed_numbers()
    body = _LIST_MARKER.sub(" ", _mask(text, facts.protected_phrases()))
    problems = []
    for m in _NUMBER.finditer(body):
        if _to_float(m.group(0)) not in allowed:
            problems.append(f"Number '{m.group(0)}' (position {m.start()}) is not in the supplied facts.")
    return problems


def catalogue_names(products) -> list[str]:
    """Names to scan drafts for. A one-word name ("Cleanser") is also a generic word, so it only
    counts as a product mention together with its brand ("Dr. Barbara Sturm Cleanser")."""
    return [n if len(n.split()) > 1 else f"{b} {n}" for b, n in zip(products["brand_name"], products["product_name"])]


def check_product_names(text: str, facts: Facts, catalogue_names: list[str]) -> list[str]:
    supplied = {p["product_name"].lower() for p in facts.products}
    supplied |= {f'{p["brand_name"]} {p["product_name"]}'.lower() for p in facts.products}
    lowered = text.lower()

    def mentioned(name: str) -> bool:
        return re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", lowered) is not None

    return [
        f"Mentions '{name}', which is not one of the chosen products."
        for name in catalogue_names
        if name.lower() not in supplied and len(name) >= 8 and mentioned(name)
        and not any(name.lower() in s for s in supplied)
    ]


def check_lexicon(text: str, facts: Facts) -> list[str]:
    body = _mask(text, facts.protected_phrases())
    problems = []
    for kind, by_lang in lexicon().items():
        for lang, terms in by_lang.items():
            for term in terms:
                for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", body, flags=re.IGNORECASE):
                    label = "medical" if kind == "medical" else "unsupported environmental"
                    problems.append(f"Blocked {label} wording '{m.group(0)}' ({lang}, position {m.start()}).")
    return problems


def check_draft(text: str, facts: Facts, catalogue_names: list[str]) -> CheckResult:
    problems = (
        check_numbers(text, facts)
        + check_product_names(text, facts, catalogue_names)
        + check_lexicon(text, facts)
    )
    return CheckResult(ok=not problems, problems=problems)
