# FraudLens P4 evaluation summary

**Status: P4 calculations complete; P1, P2 and P3 handoffs confirmed.** P1 confirmed Scenario A at a two-year horizon, complete uncensored 2022/2023 labels, and an individual-practitioner evaluation pool. P2 confirmed the individual-practitioner feature file and both payment baseline fields. P3 confirmed the prediction handoff, score direction, and 2020-only setting selection. P1 confirmed its checker used `np.argsort(-s)` without a secondary key for equal scores; it will rerun with P4's deterministic NPI tie rule. The team supplied no fixed audit budget or cost, so P4 reports audit yield at all three K values without selecting one.

## Decision evaluated

At the end of a reporting year, rank eligible providers for audit. A hit is a provider with a later Scenario A LEIE exclusion within the two-year label window (`y_A_h2`). This is a proxy for audit prioritisation, not proven billing fraud or a measured recovery.

Every model and baseline uses the same P1 evaluation pool within each year: eligible individuals with P2 features. P1's `eval_pool.parquet` was not supplied locally, so P4 reconstructed its published rule from labels and feature keys; the reconstructed keys exactly match P4's test cohorts. All providers in this pool have all three scores. K = 100, 500, and 1,000 are audit-capacity scenarios, not an operationally selected capacity. The 2020 data was used for validation, not for this held-out table.

## Model scores and tie rule

LightGBM and logistic regression scores are predicted probabilities of later exclusion. Isolation Forest is fitted separately to each decision year's billing features, which P3 says are available at the decision date, without exclusion labels. Its anomaly score was negated so more unusual billing ranks higher; that score is not a probability. P3 chose supervised settings on 2020 validation only, kept Isolation Forest settings fixed, and used matured labels from 2013–2020 for the supervised 2022 models and 2013–2021 for the supervised 2023 models. P3 confirms neither test year was used to choose model settings.

Every ranking sorts by score descending, then NPI ascending when scores tie. NPIs are ten-digit strings, so ascending string order matches ascending numeric order. This fixed tie rule uses no outcome information. P3 reports that final LightGBM has 13 trees, so tied scores can occur. P4 checked the minimum and maximum possible hit counts across cutoff tie orders. At fixed K=100/500/1,000, the held-out hit counts do not depend on tie order; at 2022's 1% cutoff, the LightGBM hit count can be 18 or 19. The large 2020 validation ties remain a separate sensitivity issue and did not determine K.

## Held-out results

| Year | Method | K=100 hits | K=500 hits | K=1,000 hits |
|---|---|---:|---:|---:|
| 2022 | LightGBM | 1 | 3 | 3 |
| 2022 | Logistic regression | 1 | 2 | 3 |
| 2022 | Isolation Forest | 0 | 0 | 0 |
| 2022 | Highest total payment | 0 | 0 | 0 |
| 2022 | Highest payment per beneficiary | 0 | 0 | 0 |
| 2022 | P1 peer-adjusted payment per beneficiary | 0 | 0 | 0 |
| 2022 | P1 peer-adjusted services per beneficiary | 1 | 1 | 1 |
| 2023 | LightGBM | 0 | 0 | 1 |
| 2023 | Logistic regression | 0 | 0 | 0 |
| 2023 | Isolation Forest | 0 | 1 | 1 |
| 2023 | Highest total payment | 0 | 0 | 0 |
| 2023 | Highest payment per beneficiary | 0 | 0 | 0 |
| 2023 | P1 peer-adjusted payment per beneficiary | 0 | 0 | 0 |
| 2023 | P1 peer-adjusted services per beneficiary | 0 | 0 | 0 |

The two raw-payment baselines found zero hits at every displayed capacity. P1's peer-adjusted services-per-beneficiary baseline found one 2022 hit at each K; its peer-adjusted payment-per-beneficiary baseline found none. The seeded random shortlists found zero in most individual draws; their expected hit count is below 0.1 even at K=1,000. The results CSV contains precision, recall, lift, and audits per hit for every method and capacity.

## Reconciliation with P1's independent 1% and 5% evaluation

P1's cutoffs inspect much larger lists than fixed K: the top 1% selects 11,657 providers in 2022 and 11,966 in 2023; the top 5% selects 58,282 and 59,827. P4 uses exactly `ceil(share × evaluation-pool size)` rows and its fixed score-descending/NPI-ascending tie rule. ROC-AUC uses all scored rows. The two additional P1 baselines rank peer-standardised payment per beneficiary and services per beneficiary; missing peer scores are placed at the observed minimum, matching P1's baseline script.

| Year | Approach | ROC-AUC | Recall@1% | Recall@5% |
|---|---|---:|---:|---:|
| 2022 | Isolation Forest | 0.678 | 7.3% (6/82) | 15.9% (13/82) |
| 2022 | P1 peer-adjusted payment per beneficiary | 0.541 | 8.5% (7/82) | 19.5% (16/82) |
| 2022 | P1 peer-adjusted services per beneficiary | 0.570 | 7.3% (6/82) | 14.6% (12/82) |
| 2022 | LightGBM | 0.790 | 23.2% (19/82) | 40.2% (33/82) |
| 2022 | Logistic regression | 0.819 | 14.6% (12/82) | 42.7% (35/82) |
| 2023 | Isolation Forest | 0.687 | 6.9% (4/58) | 24.1% (14/58) |
| 2023 | P1 peer-adjusted payment per beneficiary | 0.588 | 5.2% (3/58) | 19.0% (11/58) |
| 2023 | P1 peer-adjusted services per beneficiary | 0.612 | 5.2% (3/58) | 19.0% (11/58) |
| 2023 | LightGBM | 0.745 | 10.3% (6/58) | 34.5% (20/58) |
| 2023 | Logistic regression | 0.756 | 10.3% (6/58) | 37.9% (22/58) |

All P1 AUCs and 1%/5% recalls reconcile after rounding except 2022 LightGBM Recall@1%: P1 reported 18/82 = 22.0%, while the fixed NPI tie rule selects 19/82 = 23.2%. The cutoff splits a 77-provider equal-score group, of which 49 are selected and one has a positive label. P1 confirmed its checker used `np.argsort(-s)` without a secondary tie key and will align it to P4's score-descending/NPI-ascending rule. This explains the tie-dependent difference; its Recall@5% agrees at 33/82 = 40.2%.

For these broader cutoffs, P4 resampled the same 113 distinct positive NPIs 20,000 times across both years, retaining every record of a sampled NPI together. The paired bootstrap intervals for LightGBM minus logistic regression include zero for year-weighted AUC, Recall@1%, and Recall@5% across both years. Thus the broader-cutoff evidence does not establish either supervised model as the unique winner. Some pooled supervised-versus-Isolation-Forest intervals exclude zero, conditional on the observed negative scores and fixed rankings. These intervals do not account for negative-provider sampling, retraining, or future-year drift.

The supplemental pooled AUC is the positive-record-weighted average of the two year-specific AUCs, not a ROC-AUC from mixing scores across years. The primary results remain separate by year.

## What the comparison shows

- In 2022, LightGBM found 3 hits at K=500 versus 2 for logistic regression and 0 for Isolation Forest. Logistic regression's 2 hits were already on LightGBM's list.
- In 2023, Isolation Forest found 1 hit at K=500 while the supervised models found 0. At K=1,000, LightGBM and Isolation Forest each found 1, but they were different providers.
- All 18 paired 95% bootstrap intervals for fixed-K model hit differences include zero. The smallest exact paired p-value is 0.25. These narrow-shortlist checks do not establish a reliable winner or account for model retraining and future-year drift.
- Across both test years there are 140 positive provider-year records but only 113 distinct positive NPIs. A supplementary pooled comparison resamples whole NPIs and also finds no reliable model difference; the primary metrics remain separate by year.
- At the cutoffs used for the three models in 2022 and 2023, alternative ordering within tied scores would not change the observed hit counts.

## Evaluation pool and upstream coverage

| Year | Upstream eligible candidates | Individual evaluation pool (all scored) | Upstream candidates outside pool (unscored) | Positive records in evaluation pool | Positive records outside evaluation pool |
|---|---:|---:|---:|---:|---:|
| 2022 | 1,230,281 | 1,165,640 (94.7%) | 64,641 | 82 | 4 |
| 2023 | 1,259,333 | 1,196,525 (95.0%) | 62,808 | 58 | 3 |

The broader candidate pool includes organisations and is **not** the test denominator. P2 says candidates without features are mainly organisations. The 4 and 3 positive records outside the individual evaluation pool were not scored because they had no P2 feature row; their specific entity types cannot be verified because the local candidate file has no entity-type column. Within the evaluation pool, score coverage is 100%. Shortlist Medicare payment totals describe historical billing exposure only; no audit cost, adjudicated fraud, recoverable payment, or savings measure is available.

## Provisional recommendation

Carry **both LightGBM and logistic regression** as candidates into a prospective audit pilot; the NPI-clustered percentage-cutoff comparisons do not establish a unique supervised-model winner. Keep Isolation Forest as an additional comparator because it found a different 2023 hit at K=500. Do not deploy a combined list or claim a proven operational winner from these retrospective results. Report K=100, 500, and 1,000 as alternative narrow-list workloads; separately report P1's 1% and 5% capacities. Select a single operational capacity only after an audit team states its review capacity or audit cost/value rule; do not tune capacity on the 2022/2023 test outcomes.

A prospective pilot should record audited providers, confirmed findings, audit cost, and recoveries. Those observations are needed to estimate actual business value. Historical Medicare payments are not a savings estimate.

## Presentation figures and source tables

- [Hits by audit capacity](../presentation_figures/p4_hits_by_k.png)
- [Precision by audit capacity](../presentation_figures/p4_precision_by_k.png)
- [Evaluation-pool coverage within upstream candidates](../presentation_figures/p4_scoring_coverage.png)
- [P4 PowerPoint section](../presentation_figures/FraudLens_P4_Evaluation.pptx)
- [Interactive dashboard](p4_dashboard.html)
- [Consolidated results](p4_results.csv), [top-K shortlists](p4_top_k_shortlists.csv), [model comparison](p4_model_comparison.csv)
- [P1 percentage-cutoff reconciliation](p4_p1_percentage_reconciliation.csv), [P1 peer baselines at fixed K](p4_p1_peer_baseline_topk.csv), [NPI bootstrap intervals](p4_p1_percentage_bootstrap.csv), [paired bootstrap differences](p4_p1_percentage_paired_bootstrap.csv)
- [Per-year paired uncertainty](p4_model_paired_uncertainty.csv), [NPI-clustered pooled check](p4_pooled_npi_uncertainty.csv), [random benchmark](p4_seeded_random_baseline.csv), [coverage](p4_scoring_coverage_summary.csv)

## Team confirmation before publication

1. P1's outcome and evaluation-population definitions are confirmed in the team reply. Obtain the actual generated `eval_pool.parquet` if a direct file-to-file key comparison is required; the published construction has already been reconciled exactly.
2. P2 confirmed the final individual-practitioner feature file, `medicare_payment` for total-payment volume and `pymt_per_bene` for payment-per-beneficiary intensity.
3. P3's score direction, 2020-only selection and mature training-year split are confirmed. If predictions change, revalidate that handoff before rerunning P4.
4. The team requested audit-yield reporting without a fixed budget or cost. P4 has completed that comparison; a future operational sponsor would need to state capacity or costs before selecting one K.
