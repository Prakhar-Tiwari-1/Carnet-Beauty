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

## Baselines and the new-client score (validation period B only; C untouched)
- **Brief's hand score (v1) failed.** Shrunk like rate for the client's skin type + concern match: catalogue NDCG@10 0.007, equal to random (popularity 0.041). Almost every product has an 80–90% like rate, so it measures quality, not what a client picks next. Kept in the validation table as the record.
- **Mixing skin type or concerns into the ranking always lowered accuracy** (all tested variants on B). Skin-type-specific counts split the data into noisier groups.
- **v2 for established products = popularity + skin-type guard.** The guard drops a product when 20+ reviewers of her skin type rate it 10+ points below the step median (shrinkage m=20). It excludes 6.6% of client–product pairs at zero accuracy cost. Gap and m chosen by you as a caution policy: accuracy was identical for every gap ≥ 0.10.
- **Stated concerns are an eligibility filter in the app, not a weight.** History-derived concern profiles are too noisy to evaluate them (27% of clients have none, 26% have 5+ of 9).
- **Popularity window = 365 days**, chosen by best worst case over four backtests inside A (NDCG: 30d avg 0.043 / worst 0.014; 365d avg 0.038 / worst 0.030). No window won consistently, because popularity recommends the same ~15 products to everyone.
- **Launches: no review-history signal reliably beats random.** Across four backtests, brand prior (all variants), concern match and "Best for" skin tags each lost to random in some period. In B, one fresh launch took 1,378 likes with a brand prior of 51. An earlier "+48% for brand prior" was one lucky period and is withdrawn.
- **Launch slot = price fit, shown as a match, not a prediction.** Closeness to the client's usual spend was the only signal never clearly worse than random (mean +0.024 recall@10, worst −0.003, 3 of 4 periods). Launches never compete with established products in one list.
- **Ties broken at random (fixed seed)**, so popularity scores "no information" on launches instead of catalogue order.

## Assumptions register
| Assumption | Status | Evidence / owner |
|---|---|---|
| Free-sample text rule flags real free products | Spot-checked, not measured | 32/32 matches read for the 4 most ambiguous patterns were free products ("gifted" = brand PR in every case). Recall unknown, so 13.7% is a lower bound. |
| A client's skin type = her most common reported type | Verified | 4.4% of 188,526 clients with 2+ reviews ever report a different type. |
| Scrape is complete per product | Verified | Scraped / listed reviews per product: median 1.00. |
| Routine-step mapping (oils and night creams = moisturiser; toners, exfoliators, SPF excluded) | Decided by you | Domain choice, not testable. |
| Popularity window, guard policy, launch slot | Decided by you on B evidence | See entries above. |
| A 4–5★ review stands in for a purchase | Unverifiable here | Stated limitation; no sales impact claimed. |
| Price and stock as of March 2023 apply to earlier periods | Unverifiable here | Snapshot; stated limitation. Stock used only as a live filter. |
| Sephora "Good for" tags describe the product correctly | Unverifiable here | They are retailer marketing tags; shown as "Sephora tag", never as our claim. |
| Delisted products don't change the conclusions | Unverifiable here | Survivorship limitation stated. |
