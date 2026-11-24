# Decision log

Short entries: what was decided, why, and what it costs. Newest last.

## Scope
- **Catalogue = three routine steps** (cleanser, treatment, moisturiser; 1,102 products). The study measures exactly what the app recommends. Cost: 37% of skincare reviews (masks, eye care, SPF) are out of scope.
- **Out of scope:** deep learning, fine-tuning, accounts, real client data, sending messages, scraping, Docker, a separate API.

## Data
- **Appearance columns never loaded.** skin_tone, eye_color and hair_color are not in the read allowlist (`data/ingest.py`). Not needed, and sensitive.
- **Review text read once, then dropped.** Used only to flag free / sampled products; the cached table has no text.
- **Duplicate (author, product) pairs keep the earliest review** (5,525 pairs). The first experience is the decision-relevant one.
- **Snapshot fields banned.** product_info `rating`, `reviews`, `loves_count` are March 2023 values, so they leak the future. All product statistics are recomputed from reviews before each cutoff. Tested.
- **Price and stock are snapshots too.** Price is used as a feature (it rarely moves much); a limitation. Stock is used only as a live filter in the app.
- **Concern tags only from Sephora "Good for" highlights**, kept if 20+ routine products carry them (drops "Damage", 1 product). Nothing inferred from review text.

## Evaluation
- **Split by time.** A < 2022-07-01 (history), B Jul–Nov 2022 (ranker labels, tuning), C 2022-12-01 → 2023-03-21 (test, touched once).
- **Positive = 4–5 stars, not sampled, not seen before, routine product.** A review is a proxy for a purchase; no sales impact is claimed.
- **Sampled reviews excluded from positives.** 13.7% of reviews mention a free or gifted product (a lower bound); they rate higher (4.53 vs 4.26) and are not purchases. A sensitivity run keeps them.
- **Two slices: catalogue and launch.** 49% of returning clients' test positives are products with no review before C. History-based models cannot score those, so pooling would hide where each model works. Headline = catalogue slice.
- **Features filter `t < cutoff` internally**, so the leakage test can inject future rows and assert nothing changes. A review dated exactly at the cutoff belongs to the target period.
- **Price tier = quartile of price within the step**, by percentile rank (robust to small groups).
