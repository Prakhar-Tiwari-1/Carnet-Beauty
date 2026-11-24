"""One entry point for drafting a follow-up.

Order matters:
1. Consent gate: without marketing consent, return immediately. No template is
   rendered, no LLM client is created, nothing is cached.
2. Cache: identical facts + language + model + prompt version reuse the last draft.
3. Caps: live calls are limited per session and per day; above a cap, template.
4. Guardrails: every draft is checked; a blocked LLM draft falls back to the template
   and the reasons are shown.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from sillage.config import ROOT, load_config
from sillage.messaging import llm, templates
from sillage.messaging.guardrails import CheckResult, Facts, check_draft

CACHE_DIR = ROOT / "llm_cache"


@dataclass
class Draft:
    text: str | None
    source: str                       # "template", "llm", "llm (cached)", "none"
    check: CheckResult | None
    notices: list[str] = field(default_factory=list)


def cache_key(facts: Facts, model: str | None) -> str:
    raw = json.dumps({"facts": asdict(facts), "model": model, "prompt": llm.PROMPT_VERSION}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _daily_count_file(today: dt.date) -> Path:
    return CACHE_DIR / f"calls_{today.isoformat()}.txt"


def daily_calls(today: dt.date) -> int:
    f = _daily_count_file(today)
    return int(f.read_text()) if f.exists() else 0


def _bump_daily(today: dt.date) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    _daily_count_file(today).write_text(str(daily_calls(today) + 1))


def compose(
    facts: Facts,
    consent: bool,
    catalogue_names: list[str],
    use_llm: bool = False,
    session_calls: int = 0,
    client_factory: Callable | None = None,
    today: dt.date | None = None,
) -> Draft:
    if not consent:
        return Draft(None, "none", None, ["No marketing consent: no message drafted and no AI call made."])

    template_text = templates.render(facts)
    notices: list[str] = []
    if use_llm:
        caps = load_config()["llm"]
        today = today or dt.date.today()
        model = llm.settings()["LLM_MODEL"]
        key = cache_key(facts, model)
        cached = CACHE_DIR / f"{key}.txt"
        if cached.exists():
            text = cached.read_text(encoding="utf-8")
            check = check_draft(text, facts, catalogue_names)
            if check.ok:
                return Draft(text, "llm (cached)", check)
            notices.append("Cached AI draft failed the checks; showing the template.")
        elif not llm.is_configured():
            notices.append("No AI key configured: showing the template.")
        elif session_calls >= caps["session_cap"]:
            notices.append(f"Session limit of {caps['session_cap']} AI drafts reached: showing the template.")
        elif daily_calls(today) >= caps["daily_cap"]:
            notices.append(f"Daily limit of {caps['daily_cap']} AI drafts reached: showing the template.")
        else:
            client = (client_factory or llm.make_client)()
            text = llm.draft(facts, client=client)
            _bump_daily(today)
            check = check_draft(text, facts, catalogue_names)
            if check.ok:
                CACHE_DIR.mkdir(exist_ok=True)
                cached.write_text(text, encoding="utf-8")
                return Draft(text, "llm", check)
            notices.append("The AI draft was blocked by the checks below; showing the template instead.")
            notices.extend(check.problems)

    return Draft(template_text, "template", check_draft(template_text, facts, catalogue_names), notices)
