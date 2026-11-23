# P4 evaluation handoff

## What is ready

- Held-out 2022 and 2023 evaluation of LightGBM, logistic regression and Isolation Forest at K = 100, 500 and 1,000.
- Payment and seeded-random baselines, top-K metrics, uncertainty, shortlist overlap, specialty summaries and evaluation-pool coverage within upstream candidates.
- `reports/p4_evaluation_summary.md`: final P4 analysis and limitations.
- `reports/p4_dashboard.html`: local interactive view of year and K results.
- `reports/p4_results.csv`, `reports/p4_top_k_shortlists.csv` and `reports/p4_model_comparison.csv`: consolidated deliverables.
- `presentation_figures/FraudLens_P4_Evaluation.pptx`: six-slide **P4 section** for the team deck. Two results charts are editable PowerPoint charts.
- `presentation_figures/*.png`: standalone figures for slides or report reuse.
- `reports/p4_pooled_npi_uncertainty.csv`: supplementary paired comparison
  across both years, resampled by distinct provider NPI.

The P4 slides cover only evaluation. P1, P2 and P3 should add their own data, feature and modelling sections to the complete project presentation.

## Inputs needed to reproduce

The real files are currently placed under `stubs/`, although that folder name originally referred to a tiny synthetic example:

| File | Used for |
| --- | --- |
| `stubs/candidate_pool.parquet` | Broader upstream candidate population, including organisations; **not** the test denominator |
| `stubs/eval_pool.parquet` or `out/eval_pool.parquet` (P1 output) | Final eligible individual-practitioner test denominator. The actual file is not present locally; P4 reconstructs P1's published `eligible AND in_features` rule and checks exact NPI-year equality with its cohort. If supplied at either path, the P4 preparation script also compares its keys directly. |
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
py -3.11 src/p4_pooled_npi_uncertainty.py
py -3.11 src/p4_random_coverage.py
py -3.11 src/p4_build_final_outputs.py
```

The 2020 capacity exploration is a separate validation-only check in `src/p4_compare_k_validation.py`. It does not determine a single operational K.

## Confirmations before final submission

**P1 confirmed:** Scenario A (`y_A_h2`) is primary, both 2022 and 2023 two-year windows are complete and uncensored, and the final evaluation denominator is `eval_pool.parquet` (individual practitioners with features). Counts are 82 and 58 positive provider-year records, 113 distinct positive NPIs across both years. The actual generated Parquet file would enable a direct comparison; P4 has already recreated P1's published formula and verified exact keys against its cohorts.

**P2 confirmed:** The final feature file covers individual practitioners as the feature-building code specifies. P2 also confirmed `medicare_payment` as the total-payment volume baseline and `pymt_per_bene` as the payment-per-beneficiary intensity baseline. P4 reports broader upstream candidates separately. All individuals in the evaluation pool have all three model scores.

**P3 confirmed:** The comparison uses the same eligible, feature-covered pool for all three models and payment baselines. Higher score means higher audit priority for every approach. LightGBM and logistic regression scores are predicted exclusion probabilities; Isolation Forest's score is sign-reversed so more unusual billing ranks higher. P3 selected supervised settings using 2020 validation only; Isolation Forest settings were fixed. The supervised 2022 models used matured 2013–2020 labels, and the supervised 2023 models used matured 2013–2021 labels. Isolation Forest uses decision-year billing features but no labels. P3 confirmed the supplied predictions are the final handoff. P4 breaks tied scores by NPI ascending and reports tie sensitivity.

The 64,641 (2022) and 62,808 (2023) upstream eligible rows without features/scores are reported separately, with 4 and 3 positive records. P2 says these missing rows are mainly organisations. The local candidate file has no entity-type column, so P4 cannot verify the type of every missing row. Under P1's definition they are outside the individual evaluation denominator, so they do not enter primary Recall@K.

**Audit capacity:** The team supplied no fixed budget or cost and requested audit-yield metrics. P4 therefore reports K = 100, 500 and 1,000 without selecting a single operational K. If an audit sponsor later needs one shortlist size, it must provide review capacity or an audit cost/value rule. No observed audit costs, confirmed findings or recoveries are available, so the project cannot claim monetary savings.

P1, P2 and P3 have confirmed their respective handoffs, and the **P4 evaluation deliverables are complete**. The historical test hit counts are descriptive; paired uncertainty does not establish a reliable winner. The actual generated P1 `eval_pool.parquet` remains optional for a direct file comparison; its published construction and cohort keys already reconcile.
