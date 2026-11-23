# FraudLens P4 evaluation summary

**Status: provisional.** This report uses held-out 2022 and 2023 provider-year cohorts. The team still needs to confirm the outcome, candidate population, and score-direction/test-split handoff before publication.

## Decision evaluated

At the end of a reporting year, rank eligible providers for audit. A hit is a provider with a later Scenario A LEIE exclusion within the two-year label window (`y_A_h2`). This is a proxy for audit prioritisation, not proven billing fraud or a measured recovery.

Every model and baseline uses the same scored pool within each year. K = 100, 500, and 1,000 are audit-capacity scenarios, not an operationally selected capacity. The 2020 data was used for validation, not for this held-out table.

## Held-out results

| Year | Method | K=100 hits | K=500 hits | K=1,000 hits |
|---|---|---:|---:|---:|
| 2022 | LightGBM | 1 | 3 | 3 |
| 2022 | Logistic regression | 1 | 2 | 3 |
| 2022 | Isolation Forest | 0 | 0 | 0 |
| 2022 | Highest total payment | 0 | 0 | 0 |
| 2022 | Highest payment per beneficiary | 0 | 0 | 0 |
| 2023 | LightGBM | 0 | 0 | 1 |
| 2023 | Logistic regression | 0 | 0 | 0 |
| 2023 | Isolation Forest | 0 | 1 | 1 |
| 2023 | Highest total payment | 0 | 0 | 0 |
| 2023 | Highest payment per beneficiary | 0 | 0 | 0 |

The payment baselines found zero hits at every displayed capacity. The seeded random shortlists also found zero in most individual draws; their expected hit count is below 0.1 even at K=1,000. The results CSV contains precision, recall, lift, and audits per hit for each method and capacity.

## What the comparison shows

- In 2022, LightGBM found 3 hits at K=500 versus 2 for logistic regression and 0 for Isolation Forest. Logistic regression's 2 hits were already on LightGBM's list.
- In 2023, Isolation Forest found 1 hit at K=500 while the supervised models found 0. At K=1,000, LightGBM and Isolation Forest each found 1, but they were different providers.
- All 18 paired 95% bootstrap intervals for model hit differences include zero. The smallest exact paired p-value is 0.25. These fixed-shortlist checks do not establish a reliable winner or account for model retraining and future-year drift.
- At the cutoffs used for the three models in 2022 and 2023, deterministic tie handling did not change the observed hit counts.

## Coverage and operational limits

| Year | Full eligible pool | Scored by all models | Later exclusions in scored pool | Later exclusions with no score |
|---|---:|---:|---:|---:|
| 2022 | 1,230,281 | 1,165,640 (94.7%) | 82 | 4 |
| 2023 | 1,259,333 | 1,196,525 (95.0%) | 58 | 3 |

Providers without a score cannot enter any model shortlist. This limits full-pool recall even if every scored provider were reviewed. Shortlist Medicare payment totals describe historical billing exposure only; no audit cost, adjudicated fraud, recoverable payment, or savings measure is available.

## Provisional recommendation

Use LightGBM as a **candidate lead ranking for a prospective audit pilot**, because its observed hit counts are at least as high as the other models at most displayed capacities. Keep Isolation Forest as a comparator because it found a different 2023 hit. Do not deploy a combined list or call LightGBM a proven winner: the paired differences are highly uncertain. Report K=100, 500, and 1,000 as alternative workload levels. Select a single operational K only after an audit team states its review capacity or audit cost/value rule; do not tune K on the 2022/2023 test outcomes.

A prospective pilot should record audited providers, confirmed findings, audit cost, and recoveries. Those observations are needed to estimate actual business value. Historical Medicare payments are not a savings estimate.

## Presentation figures and source tables

- [Hits by audit capacity](../presentation_figures/p4_hits_by_k.png)
- [Precision by audit capacity](../presentation_figures/p4_precision_by_k.png)
- [Scoring coverage](../presentation_figures/p4_scoring_coverage.png)
- [P4 PowerPoint section](../presentation_figures/FraudLens_P4_Evaluation.pptx)
- [Interactive dashboard](p4_dashboard.html)
- [Consolidated results](p4_results.csv), [top-K shortlists](p4_top_k_shortlists.csv), [model comparison](p4_model_comparison.csv)
- [Pairwise uncertainty](p4_model_paired_uncertainty.csv), [random benchmark](p4_seeded_random_baseline.csv), [coverage](p4_scoring_coverage_summary.csv)

## Team confirmation before publication

1. P1: confirm Scenario A, two-year outcome windows, and eligibility rules.
2. P2: confirm that feature-covered individual practitioners represent the intended audit population.
3. P3: confirm score direction and that 2022/2023 outcomes were not used to choose model settings.
4. Audit sponsor: state review capacity or costs so a single K and financial decision rule can be chosen.
