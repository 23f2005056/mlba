# FraudLens P4 evaluation protocol

Status: **P4 handoff confirmed**. P1 confirmed the primary
outcome, uncensored test windows, and individual-practitioner evaluation pool.
P3 confirmed the prediction handoff, higher-is-higher priority score direction,
and validation-only model selection on 2020. P2 confirmed that the final feature
file covers individual practitioners and that the two payment fields are the
intended baselines.
This document records the rules
used for the held-out P4 test results in `reports/p4_evaluation_summary.md`.
The outcome, test years, and score direction
follow the real files and P3's prediction handoff. No single primary audit
capacity has been established, so P4 reports all three K values without
selecting one. All rows in
the individual evaluation pool have all three model scores.

## Decision and outcome

- Decision unit: one eligible provider (`npi`) in one reporting year (`year`).
- Primary outcome: `labels.y_A_h2`, a Scenario A exclusion within the next two
  years. A row is evaluable only when `censored_h2 == False`.
- P1 confirmed that 2022 and 2023 have complete, uncensored two-year windows.
  The 2023 three-year label is censored and is not used.
- Primary held-out test years: **2022 and 2023**, reported separately. The 2020
  predictions are validation data and are not a held-out test result.
- P3's three approaches are `supervised_lgbm`, `supervised_logreg`, and
  `anomaly_iforest`. Rank each approach by descending `score`.
  LightGBM and logistic regression scores rank providers by exclusion risk.
  They are not calibrated probabilities because training down-sampled
  negatives and weighted positives heavily. P3 negated the
  Isolation Forest anomaly score, so more unusual billing ranks higher; it is
  not an exclusion probability. Isolation Forest is fitted on each decision
  year's billing features, which P3 says are available at that decision date,
  without labels or tuning on test outcomes.
- P3 confirmed that LightGBM tree count and logistic-regression C were selected
  on 2020 only; Isolation Forest settings were fixed. The supervised models
  for 2022 used only matured 2013–2020 labels, and the supervised models for
  2023 used matured 2013–2021 labels.
- The evaluation concerns later LEIE exclusions and audit prioritisation; it
  does not identify proven billing fraud.

## Inputs and keys

| Local input | Required columns | Use |
| --- | --- | --- |
| `stubs/candidate_pool.parquet` | `npi`, `year` | Broader upstream eligible candidates, including organisations; not the primary test denominator |
| `eval_pool.parquet` (P1 handoff) | `npi`, `year` | Final individual-practitioner evaluation denominator; reconstructed from eligible labels and P2 feature keys until the actual file is supplied |
| `stubs/labels.parquet` | `npi`, `year`, `eligible`, `y_A_h2`, `censored_h2` | Outcome and observability |
| `stubs/provider_features.parquet` | `npi`, `year`, `medicare_payment`, `pymt_per_bene`, `pymt_per_bene_peer_z`, `srvcs_per_bene_peer_z`, `specialty` | Raw payment and P1 peer baselines; shortlist context |
| `stubs/predictions.parquet` | `NPI`, `year`, `approach`, `score` | Model rankings; rename `NPI` to `npi` for joining |

Keep NPI as a ten-digit string and year as an integer. The key for candidates,
labels, and features is `(npi, year)`; the prediction key is
`(npi, year, approach)`. A duplicate or conflicting key is an error, never
silently dropped. The old synthetic generator and README in `stubs/` do not
describe the large real Parquet files now stored there; do not run the
generator against these paths.

## Comparison populations

1. **Primary fair-comparison population:** use P1's `eval_pool.parquet`: eligible
   individual practitioners with P2 feature rows. Require an uncensored
   `y_A_h2` label and a finite score from **all three** P3 approaches. Rank
   both payment baselines and all models over these same NPI-year keys.
   P1's published `eligible AND in_features` construction was reproduced from
   local inputs. Its 2022 and 2023 keys exactly match P4's existing test
   cohorts: 1,165,640/82 positives and 1,196,525/58 positives. The actual
   P1-generated Parquet file has not yet been supplied for direct comparison.
2. **Upstream coverage report:** separately describe P1's broader eligible
   candidate pool, which includes organisations. It is not the test-metric
   denominator. In 2022 and 2023, 64,641 and 62,808 upstream eligible rows
   lack P2 features/model scores and are outside the evaluation pool. These
   include 4 and 3 positive records. P2 says the missing rows are mainly
   organisations; the local candidate file has no entity-type column, so their
   individual entity types have not been independently verified. Within the
   evaluation pool, all three scores cover 100%.
3. Do not assign a made-up score to upstream candidates outside the evaluation
   pool. Report any upstream sensitivity separately from primary test recall.

## Ranking and metrics

- Payment baselines: descending `medicare_payment` (total-payment volume) and
  descending `pymt_per_bene` (payment per beneficiary) from P2's feature
  table, as confirmed by P2. P1's additional peer baselines use descending
  `pymt_per_bene_peer_z` and `srvcs_per_bene_peer_z`; missing values are placed
  at the observed minimum as in P1's baseline script. Use ascending `npi` as a
  deterministic tie-breaker for **every** ranking, including model scores.
  This fixed NPI order is P4's tie rule; it does not depend on outcomes.
- Random baseline: generate reproducible seeded random shortlists on the same
  population and report their average and spread. Check the average against
  the analytical expectation (`expected_hits = K * P / N`,
  `expected_precision = P / N`, `expected_recall = K / N`,
  `expected_lift = 1`). Here `N` is the number of eligible providers in the
  comparison population and `P` is its positive count.
- For each year, approach, and `K`, select exactly the highest-ranked `K`
  distinct NPIs. Report `hits@K`, `precision@K = hits / K`,
  `recall@K = hits / P`, and `lift@K = precision@K / (P / N)`.
- Report `N`, `P`, `K`, and hits alongside rates. Mark recall and lift
  undefined if `P == 0`; reject `K > N`. Do not pool provider-years across
  test years for the primary result.
- Compare rankings with paired bootstrap uncertainty and shortlist overlap
  after the basic top-K results pass input checks.
- P3 reports the final LightGBM model has 13 trees. LightGBM has tied scores
  at some held-out cutoffs (for example, 29 providers
  at 2022 K=500 and 96 at 2023 K=1,000). P4 reports the tie sizes and the
  minimum/maximum possible hits across within-tie orders. At the displayed
  2022/2023 cutoffs, these hit bounds equal the observed counts.
- Keep 2022 and 2023 primary metrics separate. The combined 140 positive
  provider-year records correspond to 113 distinct NPIs. Any supplementary
  two-year bootstrap resamples whole NPIs so repeated providers stay together.
- Reconcile P1's independent ROC-AUC and Recall@1%/5% by sorting each year's
  full evaluation pool, taking `ceil(q × N)` rows for q = 0.01 and 0.05, and
  reporting cutoff-tie hit bounds. At 2022 LightGBM 1%, a 77-provider tie
  straddles the cutoff: P1's 18 and P4's 19 hits are both within the bound.
  P1 confirmed its checker sorted descending scores with `np.argsort(-s)`
  but had no secondary tie key; P1 will adopt P4's ascending ten-digit NPI key.
- For percentage-cutoff uncertainty, resample the same 113 outcome-positive
  NPIs across both years, retaining repeated provider records together. The
  paired bootstrap uses 20,000 shared draws. AUC bootstrap contributions use
  the fixed observed negative-score distribution, so the resulting intervals
  do not cover negative sampling, model retraining, or future-year drift.

## Audit capacity exploration on 2020 validation

The team supplied no fixed audit capacity or cost per audit. We compared
`K = 100`, `500`, and `1000` on the **2020 validation cohort only** in
`reports/p4_k_validation.csv`, without using 2022 or 2023 test outcomes to
choose K. LightGBM found 0, 4, and 5 later Scenario A exclusions at
these capacities, respectively. The highest-total-payment baseline found 0
at all three capacities. At `K = 500`, observed LightGBM precision was
`4/500 = 0.8%`; at `K = 1000`, it was `5/1000 = 0.5%`.

These observations do **not** establish a best K for the other models or a
budget that an audit team would actually use. Evaluate every model and
baseline at **all three K values in both test years**, using the same K for
every method within each comparison. Report the capacity-versus-yield tradeoff.
Choose a single operational K only from a stated audit capacity or cost/value
decision rule, not from whichever K makes one model look strongest. `K = 500`
may be shown as an illustrative midpoint but is not a selected primary K.

LightGBM has a 704-provider equal-score group at the cutoffs for both
`K = 500` and `K = 1000`, containing three positives. The NPI tie-break
gives reproducible observed hits, but any ordering within that tied group
could yield 3 to 6 hits at either cutoff. Show this tie sensitivity when
interpreting the validation comparison. Do not label any K operationally
optimal without a capacity or cost/value decision rule.

## Handoff and operational decisions

1. **Audit capacity `K`:** the team supplied no fixed budget or cost and asked
   for audit-yield comparisons. P4 evaluates `100`, `500`, and `1000` equally
   for all approaches and leaves operational K unselected. This is the agreed
   reporting approach, not an unresolved P4 computation.
2. **P2 handoff:** P2 confirmed that the final feature file covers individual
   practitioners and that `medicare_payment` and `pymt_per_bene` are the final
   volume and intensity baseline fields.
3. **P1 file:** if available, provide the generated `eval_pool.parquet` for a
   direct file comparison; its published formula and counts already reconcile.
4. **Predictions:** P3's score direction, validation-only selection and mature
   training-year split are confirmed in the team handoff. Keep that contract
   fixed if predictions are regenerated.

The P4 evaluation and P1's additional metric reconciliation are complete.
P1 has identified its differing tie convention and plans to rerun its checker
with P4's deterministic rule. The P1 file
comparison is optional because its published key rule and counts already
reconcile. The pilot recommendation is not an operational model or budget
decision. Changes to the outcome horizon require matching predictions from P3.
