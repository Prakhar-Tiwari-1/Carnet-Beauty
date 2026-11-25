from __future__ import annotations

import pytest

from sillage.messaging.guardrails import Facts, check_draft
from sillage.messaging.templates import render

CATALOGUE = [
    "Green Clean Makeup Meltaway Cleansing Balm",
    "Niacinamide 10% + Zinc 1% Oil Control Serum",
    "Ultra Repair Cream Intense Hydration",
    "The True Cream Aqua Bomb",
]


@pytest.fixture
def facts():
    return Facts(
        client_name="Claire",
        language="en",
        skin_type="dry",
        products=[
            {"product_name": "Green Clean Makeup Meltaway Cleansing Balm", "brand_name": "Farmacy", "step": "cleanser", "price_usd": 36.0},
            {"product_name": "Niacinamide 10% + Zinc 1% Oil Control Serum", "brand_name": "The Ordinary", "step": "treatment", "price_usd": 6.0},
            {"product_name": "Ultra Repair Cream Intense Hydration", "brand_name": "First Aid Beauty", "step": "moisturiser", "price_usd": 38.0},
        ],
        total=80.0,
        numbers=[87, 412],
        tags=["Good for: Dryness"],
    )


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_templates_pass_their_own_checks(facts, lang):
    facts.language = lang
    text = render(facts)
    result = check_draft(text, facts, CATALOGUE)
    assert result.ok, result.problems


def test_green_clean_product_name_is_masked(facts):
    text = "Your cleanser: Green Clean Makeup Meltaway Cleansing Balm ($36.00)."
    assert check_draft(text, facts, CATALOGUE).ok


def test_invented_price_is_blocked(facts):
    r = check_draft("The serum is only $5.00 today.", facts, CATALOGUE)
    assert not r.ok and "'5.00'" in r.problems[0]


def test_invented_percentage_is_blocked(facts):
    r = check_draft("Loved by 95% of clients.", facts, CATALOGUE)
    assert not r.ok


def test_supplied_figures_pass_in_both_formats(facts):
    assert check_draft("Liked by 87% of 412 dry-skin reviewers. Total 80,00 $.", facts, CATALOGUE).ok


def test_list_markers_allowed_but_other_small_numbers_blocked(facts):
    assert check_draft("1. Cleanser\n2. Treatment\n3. Moisturiser", facts, CATALOGUE).ok
    assert not check_draft("Use it 2 times a day.", facts, CATALOGUE).ok


def test_unsupplied_product_is_blocked(facts):
    r = check_draft("You might also enjoy The True Cream Aqua Bomb.", facts, CATALOGUE)
    assert not r.ok and "The True Cream Aqua Bomb" in r.problems[0]


@pytest.mark.parametrize("text", [
    "This serum heals breakouts.",
    "It cures dryness.",
    "Great for eczema.",
    "Ce soin guérit les rougeurs.",
    "Il soigne la peau.",
    "Clinically proven results.",
])
def test_medical_wording_blocked(facts, text):
    assert not check_draft(text, facts, CATALOGUE).ok


@pytest.mark.parametrize("text", [
    "A clean formula.",
    "Our eco-friendly packaging.",
    "Une routine écologique.",
    "Un choix éco-responsable et durable.",
    "Good for the planet.",
])
def test_environmental_wording_blocked(facts, text):
    assert not check_draft(text, facts, CATALOGUE).ok


@pytest.mark.parametrize("text", [
    "Your cleanser and treatment are ready.",
    "Votre nettoyant et votre traitement sont prêts.",
    "It is Good for: Dryness according to the Sephora tag.",
])
def test_step_names_and_supplied_tags_allowed(facts, text):
    assert check_draft(text, facts, CATALOGUE).ok, check_draft(text, facts, CATALOGUE).problems


def test_generic_step_word_is_not_an_invented_product():
    """Regression: a catalogue product named just 'Cleanser' blocked every draft saying 'cleanser'."""
    import pandas as pd

    from sillage.messaging.guardrails import Facts, catalogue_names, check_product_names

    cat = pd.DataFrame({"brand_name": ["Dr. Barbara Sturm", "CeraVe"],
                        "product_name": ["Cleanser", "Hydrating Facial Cleanser"]})
    facts = Facts(client_name="S", language="en", skin_type="dry", total=17.0,
                  products=[{"brand_name": "CeraVe", "product_name": "Hydrating Facial Cleanser", "price_usd": 17.0}])
    names = catalogue_names(cat)
    assert check_product_names("Start with your cleanser: CeraVe Hydrating Facial Cleanser.", facts, names) == []
    assert check_product_names("Or try the Dr. Barbara Sturm Cleanser.", facts, names)
