# FraudLens P4 evaluation summary

**Status: provisional.** P1 confirmed Scenario A at a two-year horizon, complete uncensored 2022/2023 labels, and an individual-practitioner evaluation pool. P3 confirmed the prediction handoff, score direction, and 2020-only setting selection. P2's final feature handoff and an operational audit capacity remain to be confirmed.

## Decision evaluated

At the end of a reporting year, rank eligible providers for audit. A hit is a provider with a later Scenario A LEIE exclusion within the two-year label window (`y_A_h2`). This is a proxy for audit prioritisation, not proven billing fraud or a measured recovery.

Every model and baseline uses the same P1 evaluation pool within each year: eligible individuals with P2 features. P1's `eval_pool.parquet` was not supplied locally, so P4 reconstructed its published rule from labels and feature keys; the reconstructed keys exactly match P4's test cohorts. All providers in this pool have all three scores. K = 100, 500, and 1,000 are audit-capacity scenarios, not an operationally selected capacity. The 2020 data was used for validation, not for this held-out table.

## Model scores and tie rule

LightGBM and logistic regression scores are predicted probabilities of later exclusion. Isolation Forest is fitted separately to each decision year's billing features, which P3 says are available at the decision date, without exclusion labels. Its anomaly score was negated so more unusual billing ranks higher; that score is not a probability. P3 chose supervised settings on 2020 validation only, kept Isolation Forest settings fixed, and used matured labels from 2013–2020 for the supervised 2022 models and 2013–2021 for the supervised 2023 models. P3 confirms neither test year was used to choose model settings.

Every ranking sorts by score descending, then NPI ascending when scores tie. This fixed tie rule uses no outcome information. P3 reports that final LightGBM has 13 trees, so tied scores can occur. P4 checked the minimum and maximum possible hit counts across cutoff tie orders; all held-out 2022/2023 cutoffs have the same hit count under every tie order. The large 2020 validation ties remain a separate sensitivity issue and did not determine K.

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
- Across both test years there are 140 positive provider-year records but only 113 distinct positive NPIs. A supplementary pooled comparison resamples whole NPIs and also finds no reliable model difference; the primary metrics remain separate by year.
- At the cutoffs used for the three models in 2022 and 2023, alternative ordering within tied scores would not change the observed hit counts.

## Evaluation pool and upstream coverage

| Year | Upstream eligible candidates | Individual evaluation pool (all scored) | Upstream candidates outside pool (unscored) | Positive records in evaluation pool | Positive records outside evaluation pool |
|---|---:|---:|---:|---:|---:|
| 2022 | 1,230,281 | 1,165,640 (94.7%) | 64,641 | 82 | 4 |
| 2023 | 1,259,333 | 1,196,525 (95.0%) | 62,808 | 58 | 3 |

The broader candidate pool includes organisations and is **not** the test denominator. The 4 and 3 positive records outside the individual evaluation pool were not scored because they had no P2 feature row; their entity type is not established here. Within the evaluation pool, score coverage is 100%. Shortlist Medicare payment totals describe historical billing exposure only; no audit cost, adjudicated fraud, recoverable payment, or savings measure is available.

## Provisional recommendation

Use LightGBM as a **candidate lead ranking for a prospective audit pilot**, because its observed hit counts are at least as high as the other models at most displayed capacities. Keep Isolation Forest as a comparator because it found a different 2023 hit. Do not deploy a combined list or call LightGBM a proven winner: the paired differences are highly uncertain. Report K=100, 500, and 1,000 as alternative workload levels. Select a single operational K only after an audit team states its review capacity or audit cost/value rule; do not tune K on the 2022/2023 test outcomes.

A prospective pilot should record audited providers, confirmed findings, audit cost, and recoveries. Those observations are needed to estimate actual business value. Historical Medicare payments are not a savings estimate.

## Presentation figures and source tables

- [Hits by audit capacity](../presentation_figures/p4_hits_by_k.png)
- [Precision by audit capacity](../presentation_figures/p4_precision_by_k.png)
- [Evaluation-pool coverage within upstream candidates](../presentation_figures/p4_scoring_coverage.png)
- [P4 PowerPoint section](../presentation_figures/FraudLens_P4_Evaluation.pptx)
- [Interactive dashboard](p4_dashboard.html)
- [Consolidated results](p4_results.csv), [top-K shortlists](p4_top_k_shortlists.csv), [model comparison](p4_model_comparison.csv)
- [Per-year paired uncertainty](p4_model_paired_uncertainty.csv), [NPI-clustered pooled check](p4_pooled_npi_uncertainty.csv), [random benchmark](p4_seeded_random_baseline.csv), [coverage](p4_scoring_coverage_summary.csv)

## Team confirmation before publication

1. P1's outcome and evaluation-population definitions are confirmed in the team reply. Obtain the actual generated `eval_pool.parquet` if a direct file-to-file key comparison is required; the published construction has already been reconciled exactly.
2. P2: confirm the final feature file and individual-practitioner filter.
3. P3's score direction, 2020-only selection and mature training-year split are confirmed. If predictions change, revalidate that handoff before rerunning P4.
4. Audit sponsor: state review capacity or costs so a single K and financial decision rule can be chosen.
