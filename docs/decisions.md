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
- **v2 for established products = popularity + skin-type guard.** The guard drops a product when 20+ reviewers of her skin type rate it 10+ points below the step median (shrinkage m=20). It excludes 6.6% of client–product pairs at zero accuracy cost. Gap and m are a caution policy: accuracy was identical for every gap ≥ 0.10.
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
| Routine-step mapping (oils and night creams = moisturiser; toners, exfoliators, SPF excluded) | Product decision | Domain choice, not testable. |
| Popularity window, guard policy, launch slot | Product decision on B evidence | See entries above. |
| A 4–5★ review stands in for a purchase | Unverifiable here | Stated limitation; no sales impact claimed. |
| Price and stock as of March 2023 apply to earlier periods | Unverifiable here | Snapshot; stated limitation. Stock used only as a live filter. |
| Sephora "Good for" tags describe the product correctly | Unverifiable here | They are retailer marketing tags; shown as "Sephora tag", never as our claim. |
| Delisted products don't change the conclusions | Unverifiable here | Survivorship limitation stated. |

## Learned models and the test period
- **A within-period hold-out leaked.** The first ranker was validated on held-out users from the same months it was trained on. Static features (price, step, brand prior) identify products, so it memorised which products were hot that season: NDCG@10 0.177 and 75% launch recall, far too good. Fixed by validating across periods: train on Jan–Apr 2022 labels, early-stop on May–Jun, compare on B.
- **First test run (pre-registered rule "best on validation"): the ranker failed on established products.** NDCG@10 0.036 vs popularity 0.042 (CI of the difference −0.011 to −0.001). Item-kNN, untuned, won (+0.024, CI +0.020 to +0.029). The ranker held on launches (recall@10 2.2× random). Kept in `results/results_first_run.json` and on the Results page.
- **Post-hoc diagnosis (analysis only, no model change):** 47% of the ranker's gain came from absolute price and brand prior. It learned last season's winners, not the client; heavy users, who already own those winners, were hit hardest (recall 0.047 vs 0.199 for popularity).
- **Extended study before a second, logged test run.** Every family a reviewer would expect: item-kNN, RP3beta, EASE, iALS, BPR-MF, content TF-IDF, and LightGBM in three variants (original; client-centric without static identity features and with all CF scores; client-centric with a binary objective). Tuned on three rolling folds (Jul–Nov 2021, Feb–Jun 2022, Jul–Nov 2022), rankers trained on the window just before each fold, fixed rounds (no early stopping inside a training period).
- **Selection rule written before the second test run:** a model replaces popularity only if its NDCG@10 beats popularity with a 95% CI above zero in every fold; then the highest mean wins. Launch slot: same rule against random. The second test run's reason is logged in `results/test_runs.log`; both runs are reported.
- **Grid edges checked.** EASE (λ) and RP3beta (α, β) first won at the edge of their grids; the grids were widened and the study re-run. The tuning report flags any remaining edge winner.
- **Bug caught:** item-kNN with shrink 0 divided 0 by 0 for never co-liked pairs (NaN scores). Fixed with a guarded division and a regression test. Earlier runs used shrink 10 and were unaffected.
- **Ties broken at random for every model**, so "no information" (e.g. CF on launches) ranks like random, not like catalogue order.

## Compute
- **GPU where it helps.** EASE, RP3beta, iALS and BPR run in PyTorch on CUDA (RTX 4000 Ada). iALS was rewritten with batched matrix products instead of atomic scatter-adds so reruns are bit-identical; all four were checked twice with the same seed.
- **LightGBM on CPU, by measurement.** A CUDA build (`-DUSE_CUDA=ON`) was benchmarked on this ranking task: 112 s vs 1.2 s per fit, and two runs with the same seed differed (max prediction gap 0.84). Reproducibility is a requirement, so CPU.
- **PyTorch environment.** The home quota was full (30 of 30 GB), so the project venv links the CUDA PyTorch already installed in a local conda env instead of downloading 3 GB again. The deployed app does not use PyTorch. Clean alternative: free ~3 GB and `pip install torch` into `.venv`.

## Product
- **New clients: consultation score = popularity + skin-type guard;** launches matched on price fit to the stated usual spend (a match, not a prediction).
- **Returning clients: the models chosen by the pre-registered rule** (routine and launch slot separately). Reasons are faithful to each model: co-like counts for item-kNN, top contributing history product for EASE/RP3beta, closest liked product for content, top feature contributions for LightGBM, and "no per-product reason" for latent-factor models.
- **Deployment refit.** Once selection is final, the chosen configuration is refitted on the newest labels (the test period) to score the synthetic clients "today" (March 2023). Standard practice; selection never used this refit.
- **Concern filter applies to a step only when 20+ products in that step carry the tag;** otherwise the app says the filter was skipped and why (e.g. 2 cleansers tagged "Dark spots").
- **App launch slot uses Sephora's own "new" flag** (a March 2023 snapshot, i.e. current for the app), not an invented launch window.
- **Ingredient filter = whole-word text match, with the INCI synonym fragrance ↔ parfum.** It also excludes "fragrance-free" when avoiding fragrance (the safe direction); stated in the on-screen disclaimer.
- **Budget = exact search** over all cleanser × treatment pairs with the best moisturiser that fits (sorted prefix maximum), tested against brute force.

## Extended study outcome (validation folds only)
- **Selected for both slots: the client-centric LightGBM ranker (100 rounds).** Established products: mean NDCG@10 0.077 over the three folds, beating popularity in every fold (worst lower CI bound +0.024). Launches: 0.116, beating random in every fold. Item-kNN and RP3beta also passed the rule but scored lower (0.053, 0.054); EASE, iALS, BPR and content did not pass on established products. Content TF-IDF passed on launches but scored below the ranker.
- **Grid edges checked beyond the grid** (mean NDCG@10 over the folds): EASE plateaus (0.0497 at λ = 30k, 0.0499 at 300k, 0.0498 at 1M); RP3beta peaks at α = 1.0 (0.0539; 0.0523 at 1.5; 0.0465 at 2.0); BPR peaks at lr = 0.01 (0.0407; 0.0369 at 0.03; 0.0255 at 0.1); iALS peaks between reg 10 and 100 (0.0417 at 10, 0.0420 at 30, 0.0402 at 100: a 0.0003 gain, inside fold noise, so the study settings were kept).
- **The GPU is shared with another user's jobs.** One side check ran out of GPU memory while the study was running and was re-run afterwards.
