# P4 evaluation handoff

## What is ready

- Held-out 2022 and 2023 evaluation of LightGBM, logistic regression and Isolation Forest at K = 100, 500 and 1,000.
- Payment and seeded-random baselines, top-K metrics, uncertainty, shortlist overlap, specialty summaries and scoring coverage.
- `reports/p4_evaluation_summary.md`: provisional conclusions and limitations.
- `reports/p4_dashboard.html`: local interactive view of year and K results.
- `reports/p4_results.csv`, `reports/p4_top_k_shortlists.csv` and `reports/p4_model_comparison.csv`: consolidated deliverables.
- `presentation_figures/FraudLens_P4_Evaluation.pptx`: six-slide **P4 section** for the team deck. Two results charts are editable PowerPoint charts.
- `presentation_figures/*.png`: standalone figures for slides or report reuse.

The P4 slides cover only evaluation. P1, P2 and P3 should add their own data, feature and modelling sections to the complete project presentation.

## Inputs needed to reproduce

The real files are currently placed under `stubs/`, although that folder name originally referred to a tiny synthetic example:

| File | Used for |
| --- | --- |
| `stubs/candidate_pool.parquet` | Eligible provider-year candidates |
| `stubs/labels.parquet` | Future exclusion outcome and censoring |
| `stubs/provider_features.parquet` | Payment baselines and shortlist context |
| `stubs/predictions.parquet` | P3's three approach scores |

The large local Parquet replacements and model pickle files are **not** part of the P4 branch commit. A fresh clone requires the matching P1/P2/P3 real input handoff to rerun the analysis. P4 consumes predictions; it does not need the pickle files to evaluate them.

## Rebuild order

Use Python 3.11 with pandas, pyarrow, NumPy, SciPy and matplotlib installed. From the repository root:

```powershell
py -3.11 src/p4_prepare_cohort.py
foreach ($model in 'supervised_lgbm','supervised_logreg','anomaly_iforest') {
  py -3.11 src/p4_rank.py --approach $model
  py -3.11 src/p4_select_topk.py --approach $model --k 100 500 1000
  py -3.11 src/p4_evaluate_topk.py --approach $model
}
py -3.11 src/p4_compare_baselines.py
py -3.11 src/p4_confidence_intervals.py
py -3.11 src/p4_shortlist_analysis.py
py -3.11 src/p4_model_followup.py --approach supervised_logreg
py -3.11 src/p4_model_followup.py --approach anomaly_iforest
py -3.11 src/p4_compare_models.py
py -3.11 src/p4_paired_uncertainty.py
py -3.11 src/p4_random_coverage.py
py -3.11 src/p4_build_final_outputs.py
```

The 2020 capacity exploration is a separate validation-only check in `src/p4_compare_k_validation.py`. It does not determine a single operational K.

## Confirmations before final submission

**P1:** Confirm that Scenario A (`y_A_h2`), the two-year outcome window, and candidate eligibility are the intended primary evaluation target. Confirm that 2022 and 2023 labels are fully observed under the supplied censor rule.

**P2:** Confirm that the feature-covered, scored individual-practitioner pool is the intended audit decision population. P4 separately reports the eligible providers and later exclusions missing scores.

**P3:** Confirm that higher score means higher audit priority for all three approaches and that 2022/2023 outcome information was not used to choose model settings. Confirm the supplied predictions remain the final handoff.

**Audit sponsor or team:** State the number of audits possible, or an audit cost and value rule, before selecting one K. No observed audit costs, confirmed findings or recoveries are available, so the project cannot claim monetary savings.

Until those confirmations arrive, the evaluation report and deck remain **provisional**. The historical test hit counts are descriptive; paired uncertainty does not establish a reliable winner.
