# Sillage

**Problem:** a beauty advisor has ten minutes to turn a consultation into the right routine and a relevant follow-up, without pushing the wrong products or misusing client data.
**What it is:** a web demo that gives the advisor an explained three-step routine and a consent-checked follow-up draft, backed by an honest comparison of recommendation methods on about a million real skincare reviews.
**Who it's for:** advisors and clienteling managers in luxury and prestige beauty (independent portfolio project).

<!-- LIVE LINK, SCREENSHOT, RESULTS TABLE AND KEY FINDING ARE FILLED FROM results/ -->

## How it works
1. **Consultation.** The advisor picks a client (all synthetic) and confirms skin type, concerns, budget and ingredients to avoid.
2. **Routine.** Hard filters first (in stock, ingredients to avoid, products clearly disliked by that skin type, stated concerns), then an exact search for the best cleanser + treatment + moisturiser within the total budget. If nothing passes, the app names the constraint and never relaxes a filter silently.
3. **Two paths.** New clients get the consultation score (what clients liked most, minus products their skin type dislikes). Returning clients get the model chosen by a pre-registered rule on three validation periods. The page says which path is used.
4. **Reasons.** Real figures with sample sizes (hidden under 20 reviews), and explanations that come from the model that ranked the product.
5. **Follow-up.** A draft in English or French, built only from the routine and the figures shown. No consent: no draft and no AI call. Invented products, prices or figures, medical wording and unsupported environmental claims ("clean", "green", "eco-friendly", EN/FR) are blocked. The advisor edits and approves; nothing is sent.
6. **Brand owner view.** One brand's own routine and its gaps by skin type and step.

## Run it
```bash
# 1. Data (not in the repo): download nadyinky/sephora-products-and-skincare-reviews from Kaggle into data/raw/
# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
# 3. Rebuild everything with one command (step0, data, baselines, study, test, export)
python -m sillage.build
#    The test period is evaluated once; a rerun needs --rerun-reason "..." and is logged.
# 4. Tests and app
pytest
streamlit run app/streamlit_app.py
```
- **Optional AI drafting:** set `LLM_API_KEY`, `LLM_MODEL` and optionally `LLM_BASE_URL` (any OpenAI-compatible endpoint) as environment variables or Streamlit secrets. Without them the app uses templates. Calls are cached and capped per session and per day.
- **GPU:** EASE, RP3beta, iALS and BPR train on CUDA when PyTorch sees a GPU. LightGBM runs on CPU by measured choice (see `docs/decisions.md`).

## Repository
```
src/sillage/    data rules, features, models, evaluation, routine engine, messaging + guardrails
app/            Streamlit pages (advisor, brand owner, results); reads app_data/ only
app_data/       small derived tables for the app: no review text, no author IDs, counts >= 20
clients/        synthetic client profiles
guardrails/     blocked wording, EN/FR
tests/          leakage, filters, budget search, consent, guardrails, export rules, app pages
docs/           step0.md, decisions.md, results.md, business_case.md, demo_script.md, review_sheet.csv
```

## Data and licence
- **Dataset:** *Sephora Products and Skincare Reviews* by nadyinky on Kaggle, declared CC BY 4.0, scraped from Sephora in March 2023. Not redistributed here.
- **Data rules:** skin tone, eye colour and hair colour are never loaded. Review text is read once to flag free-sample reviews, then dropped. The deployed app contains no review text and no author IDs, and hides any figure based on fewer than 20 reviews.
- **Disclaimer:** independent portfolio project, not affiliated with Sephora, LVMH or L'Oréal. No company logos.
