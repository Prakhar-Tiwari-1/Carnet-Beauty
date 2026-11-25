# Business case: explained routines and consented follow-ups for beauty advisors

*Independent portfolio project. All company-specific figures below are labelled assumptions, not data.*

## 1. User and journey
**User:** a beauty advisor (clienteling role) in a multi-brand store or a maison boutique.

| | Today | With Sillage |
|---|---|---|
| Consultation | 5–10 min, recommendations from memory and current push products | Same conversation; the advisor records skin type, concerns, budget and ingredients to avoid |
| Routine | Often one hero product; budget and stock checked by hand | Three steps within budget, in stock, with the evidence behind each pick (e.g. "liked by 87% of 412 dry-skin reviewers") |
| Launches | Pushed to everyone | Shown in a separate slot, filtered on the client's concerns and skin type, labelled as low-evidence |
| Follow-up | Generic message, or none; consent checked inconsistently | Draft in French or English from the routine; blocked without consent; invented prices, medical and "green" claims blocked; advisor edits and approves |

**Business question:** how does an advisor turn a short consultation into the right routine and a relevant follow-up, without pushing the wrong products or misusing client data?

## 2. KPI tree
**Primary KPI: 90-day repeat purchase rate of consulted clients** (bought again within 90 days of the consultation).

- Repeat purchase rate
  - Follow-up reach = consulted clients with consent × follow-up approved and sent
  - Follow-up response = clients who return or buy online after the follow-up
  - Routine completion = clients who own all three steps within 90 days
  - Fit = share of recommended products not returned and rated well

**Guardrail metrics** (a pilot stops if any degrades beyond a pre-agreed threshold):
- Opt-out and unsubscribe rate after follow-ups
- Product return rate on recommended products
- Complaints mentioning the message or a recommendation
- Advisor approval rate and edit distance of drafts (low approval = low trust)
- Brand mix of recommendations versus the store's sales mix (multi-brand fairness)
- Messages blocked by the guardrails (should be near zero after approval)

## 3. A/B test design
- **Unit of randomisation: the store.** Advisors in one store share clients and talk to each other, so randomising advisors or clients would contaminate the control group.
- **Duration:** 12 weeks of consultations plus 90 days of follow-up for the primary KPI. Seasonal effects (holidays, launches) hit both arms equally because both run at the same time.
- **Analysis:** difference in repeat purchase rate with cluster-robust standard errors; guardrails monitored weekly.

**Sample size.** Two-proportion test, α = 5% two-sided, power 80%. Design effect = 1 + (m − 1) × ICC. Every input is an assumption:

| Baseline repeat rate (assumed) | Minimum detectable effect | Clients per arm, individual randomisation | ICC (assumed) | Clients per arm, by store | Stores per arm at 300 consulted clients each (assumed) |
|---|---|---|---|---|---|
| 20% | +1 pt | 25,580 | 0.01 – 0.02 | 102,063 – 178,545 | 341 – 596 |
| 20% | +2 pt | 6,507 | 0.01 – 0.02 | 25,962 – 45,417 | 87 – 152 |
| 20% | +3 pt | 2,940 | 0.01 – 0.02 | 11,730 – 20,521 | 40 – 69 |
| 30% | +2 pt | 8,391 | 0.01 – 0.02 | 33,478 – 58,566 | 112 – 196 |
| 30% | +3 pt | 3,760 | 0.01 – 0.02 | 15,001 – 26,243 | 51 – 88 |

**Implication:** a store-level pilot can realistically detect about +3 points, not +1. A smaller effect would need several hundred stores per arm. If the true effect is small, say so up front, or measure a closer proxy first (follow-up response rate).

<!-- RESULTS-DEPENDENT SECTIONS BELOW ARE FILLED FROM results/results.json -->
