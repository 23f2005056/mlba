# FraudLens P4 evaluation protocol

Status: **provisional pending team confirmation**. This document records the rules
used for the held-out P4 test results in `reports/p4_evaluation_summary.md`.
The outcome, test years, and score direction
follow the real files and `stubs/P3_HANDOFF_FOR_P4.md`. No single primary audit
capacity has been established; treatment of candidates without model scores
also needs team confirmation.

## Decision and outcome

- Decision unit: one eligible provider (`npi`) in one reporting year (`year`).
- Primary outcome: `labels.y_A_h2`, a Scenario A exclusion within the next two
  years. A row is evaluable only when `censored_h2 == False`.
- Primary held-out test years: **2022 and 2023**, reported separately. The 2020
  predictions are validation data and are not a held-out test result.
- P3's three approaches are `supervised_lgbm`, `supervised_logreg`, and
  `anomaly_iforest`. Rank each approach by descending `score`; the anomaly
  score is not interpreted as an exclusion probability.
- The evaluation concerns later LEIE exclusions and audit prioritisation; it
  does not identify proven billing fraud.

## Inputs and keys

| Local input | Required columns | Use |
| --- | --- | --- |
| `stubs/candidate_pool.parquet` | `npi`, `year` | Eligible audit population |
| `stubs/labels.parquet` | `npi`, `year`, `eligible`, `y_A_h2`, `censored_h2` | Outcome and observability |
| `stubs/provider_features.parquet` | `npi`, `year`, `medicare_payment`, `pymt_per_bene`, `specialty` | Baselines and shortlist context |
| `stubs/predictions.parquet` | `NPI`, `year`, `approach`, `score` | Model rankings; rename `NPI` to `npi` for joining |

Keep NPI as a ten-digit string and year as an integer. The key for candidates,
labels, and features is `(npi, year)`; the prediction key is
`(npi, year, approach)`. A duplicate or conflicting key is an error, never
silently dropped. The old synthetic generator and README in `stubs/` do not
describe the large real Parquet files now stored there; do not run the
generator against these paths.

## Comparison populations

1. **Primary fair-comparison population:** within each test year, start from
   P1's eligible candidate pool, require an uncensored label and P2 features,
   then require a finite score from **all three** P3 approaches. Rank the two
   payment baselines and all three models over exactly these same providers.
2. **Full-pool coverage report:** separately show the size and positive count
   of P1's entire eligible, uncensored pool, how many have P2 features, and
   how many receive every P3 score. Count positive candidates lost at each
   step. This prevents the common-pool metrics from hiding model coverage.
3. Do not assign a made-up score to unscored candidates. If the team requests
   operational recall over the full eligible pool, report it separately and
   count unscored positive candidates as misses.

## Ranking and metrics

- Payment baselines: descending `medicare_payment` and descending
  `pymt_per_bene` from P2's feature table. Use ascending `npi` as a
  deterministic tie-breaker for every ranking.
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

## Awaiting team confirmation

1. **Audit capacity `K`:** evaluate `100`, `500`, and `1000` equally for all
   approaches. A single operational capacity remains unselected because no
   audit budget or cost/value rule has been supplied.
2. **Population:** P1 and P2 should confirm that restricting the primary
   comparison to feature-covered, scored individual practitioners matches the
   intended audit decision. The full candidate-pool coverage will be shown
   regardless.
3. **Outcome:** P1 should confirm that Scenario A, horizon two, and the 2022
   and 2023 outcome windows are the final primary test definition.
4. **Predictions:** P3 should confirm that higher scores always mean higher
   audit priority and that 2022 and 2023 were not used to choose settings.

Do not label this protocol final or treat the provisional audit recommendation
as an operational decision until these items are resolved. Changes to the
outcome horizon require matching predictions from P3.
