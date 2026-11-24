"""Template follow-ups in English and French, built only from supplied facts."""
from __future__ import annotations

from sillage.messaging.guardrails import Facts

STEP_LABEL = {
    "en": {"cleanser": "Cleanser", "treatment": "Treatment", "moisturiser": "Moisturiser"},
    "fr": {"cleanser": "Nettoyant", "treatment": "Traitement", "moisturiser": "Hydratant"},
}
SKIN_LABEL = {
    "en": {"dry": "dry", "oily": "oily", "combination": "combination", "normal": "normal"},
    "fr": {"dry": "sèche", "oily": "grasse", "combination": "mixte", "normal": "normale"},
}


def price(value: float, lang: str) -> str:
    """$39.00 in English, 39,00 $ in French (prices are USD in the source data)."""
    return f"${value:.2f}" if lang == "en" else f"{value:.2f}".replace(".", ",") + " $"


def render(facts: Facts) -> str:
    lang = facts.language
    steps = STEP_LABEL[lang]
    lines = [
        f"{i}. {steps[p['step']]}: {p['brand_name']}, {p['product_name']} ({price(p['price_usd'], lang)})"
        for i, p in enumerate(facts.products, start=1)
    ]
    skin = SKIN_LABEL[lang].get(facts.skin_type)
    if lang == "fr":
        intro = "Merci pour votre visite. Comme convenu, voici la routine que nous avons composée"
        intro += f" pour votre peau {skin} :" if skin else " :"
        return "\n".join([
            f"Bonjour {facts.client_name},", "", intro, "", *lines, "",
            f"Total : {price(facts.total, lang)}.", "",
            "N'hésitez pas à me dire comment votre peau se sent après quelques semaines. "
            "Je reste à votre disposition pour ajuster la routine.", "",
            "Bien à vous,", facts.advisor_name,
        ])
    intro = "Thank you for your visit. As discussed, here is the routine we put together"
    intro += f" for your {skin} skin:" if skin else ":"
    return "\n".join([
        f"Dear {facts.client_name},", "", intro, "", *lines, "",
        f"Total: {price(facts.total, lang)}.", "",
        "I would love to hear how your skin feels after a few weeks, "
        "and I am happy to adjust the routine at any time.", "",
        "Warm regards,", facts.advisor_name,
    ])
