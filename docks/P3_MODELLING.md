# FraudLens - P3 Modelling: what was done and how

**Owner:** Vishrut (P3, machine learning)
**Code:** `src/p3_models.py`, `src/p3_utils.py`
**Run:** `python src/p3_models.py --data real [--label y_A_h2|y_B_h2|y_A_h3]` (or `--data stub` for the synthetic smoke test)

## 1. Goal
Rank Medicare Part B providers by audit priority. For each decision year T, score every eligible provider so that those later excluded by the OIG (LEIE) rank as high as possible.

The project identifies elevated audit risk. It does not identify proven fraud (work plan rule 4).

Two kinds of ranking were built, as the work plan specifies:
- **Supervised (Approach A):** learn from past exclusions.
  - Logistic Regression
  - Gradient Boosting (LightGBM)
- **Unsupervised (Approach B):** Isolation Forest. It flags unusual billing without using any labels.

## 2. Inputs
| Input | From | Used for |
| --- | --- | --- |
| `labels.parquet` (labels v1, frozen) | P1 | Target `y_A_h2` (primary), `y_B_h2`, `y_A_h3`; `eligible`; `censored_h{h}` |
| `candidate_pool.parquet` | P1 | Cross-check: eligible row count must equal it (12,232,035) |
| `split_plan.csv` | P1 | Allowed training years per decision year; asserted exactly |
| `eval_pool.parquet` | P1 | Verified equal to the pool P3 scores (11,551,956 rows) |
| `provider_features.parquet` | P2 | 30 numeric features |

The SHA256 of every P1 file was checked against `labels_v1_hashes.txt`, and all match.

## 3. Pool, label and decision timing
- **Pool:** eligible provider-years (`eligible == True`) that also have a CMS feature row, i.e. individual Part B practitioners. This is identical to P1's `eval_pool.parquet`.
  - About 60k eligible rows per year have no features and are not scored. Counts are in `reports/p3_scoring_coverage.csv`.
- **Primary label:** `y_A_h2` = Scenario A exclusion (codes 1128a1, 1128a3, 1128b1, 1128b6, 1128b7) in the 2 years after `year`.
- **Decision time:** 31 December of `year`. Features use CMS data from `year` and earlier only.
- **Positives** (distinct providers, primary label): train 461, validation 64, test 113. Base rate about 5.8e-05.

## 4. Time-based splits and leakage guards
| Role | Decision year | Training years (h=2) |
| --- | --- | --- |
| Validation | 2020 | 2013-2018 |
| Test | 2022 | 2013-2020 |
| Test | 2023 | 2013-2021 |

- **Maturity rule:** a row from year T' may train a model for year T only if T' + h <= T, so its whole outcome window has closed by the decision date (`allowed_train_years`).
- Guards in the code, all enforced with `assert`:
  - `check_split_plan`: our allowed years and decision years must match P1's `split_plan.csv` exactly.
  - `make_split`: no immature or censored rows in training, and no row both trained on and scored.
  - `check_no_leakage`: features exclude `npi`, `year`, `specialty`, `state` and any `y_*`, `censored_*`, `excluded_*` or `label_*` column.
  - `load_real`: eligible rows only, and the count must equal `candidate_pool`. There must be no duplicate (npi, year).
- 2022 and 2023 were never used for any choice, and P3 computed no metric on them. P4 does the test evaluation.
- P1 independently reviewed `p3_models.py` and found no leakage.

## 5. Features
The 30 numeric columns from `docks/FEATURE_DICTIONARY.md`:
- **Volume and money:** `medicare_participating`, `tot_benes`, `tot_srvcs`, `submitted_charges`, `allowed_amount`, `medicare_payment`
- **Ratios:** `pymt_per_bene`, `srvcs_per_bene`, `pymt_per_srvc`, `submitted_per_bene`, `charge_to_allowed`, `allowed_to_pymt`, `pymt_to_charge`
- **Logs:** `log_medicare_payment`, `log_tot_benes`, `log_tot_srvcs`
- **Specialty peer robust z-scores:** `pymt_per_bene_peer_z`, `srvcs_per_bene_peer_z`, `charge_to_allowed_peer_z`
- **Year-over-year changes** (using only t-1): `*_yoy_diff`, `*_yoy_growth`, `is_new_billing_provider`

Year-over-year values are missing for new providers (about 1.9M rows). The `is_new_billing_provider` flag carries that information.

## 6. Handling extreme class imbalance
- **Negative sampling (training only):** keep every positive plus 100,000 random negatives (seeded) per training fit. Scoring always uses every pool row.
- **Class weighting:**
  - LightGBM uses `scale_pos_weight = negatives / positives` (about 216 for the primary label).
  - Logistic Regression uses `class_weight="balanced"`.
- **Consequence:** scores rank providers but are **not calibrated probabilities**.

## 7. The three approaches
| | LightGBM (`supervised_lgbm`) | Logistic Regression (`supervised_logreg`) | Isolation Forest (`anomaly_iforest`) |
| --- | --- | --- | --- |
| Uses labels | yes | yes | **no** |
| Preprocessing | none (handles NaN and skew) | signed log `sign(x)*log(1+abs(x))`, median imputation, standardisation | same as Logistic Regression |
| Fixed settings | objective binary, lr 0.05, subsample 0.8, colsample 0.8, L2 = 5, up to 2,000 trees, deterministic | lbfgs, max_iter 2000 | 300 trees, max_samples 256 |
| Chosen on 2020 | grid: num_leaves {7, 15, 31} x min_child_samples {50, 200}; trees by early stopping on 2020 PR-AUC (patience 100) | C in {0.01, 0.1, 1} | nothing chosen |
| Selection rule | highest 2020 PR-AUC | highest 2020 PR-AUC | none |
| Fit for test years | refit on that year's allowed years with the chosen settings | same | fit on 200k sampled rows of the scored year's own features; scores every row |
| Score | P(y=1), weighted | P(y=1), weighted | `-score_samples` (higher = more anomalous) |

**Why three:**
- Logistic Regression is the plan's simple, transparent supervised model.
- LightGBM tests whether non-linear patterns and feature interactions help.
- Isolation Forest tests whether labels are needed at all. It is also the plan's fallback if the labels proved too weak: only about 60 positives a year, and about 10% of LEIE records have an NPI.

`signed_log` lives in `src/p3_utils.py` so that the saved `.pkl` models can be loaded outside the training script.

## 8. Tuning history and the freeze rule
All four rounds were judged on 2020 only (`reports/p3_val_metrics_v1..v4*.csv`):

| Round | Class weight | Early stopping on | Negatives | Best 2020 PR-AUC |
| --- | --- | --- | --- | --- |
| v1 | full | PR-AUC | 1,000,000 | 0.00016 (stopped after 3-21 trees) |
| v2 | sqrt | ROC-AUC | 1,000,000 | 0.00065 |
| v3 | sqrt | ROC-AUC | 100,000 | 0.00096 |
| v4 | full | PR-AUC | 100,000 | 0.00118 |

- **Pre-set rule:** keep sqrt weighting / AUC stopping only if 2020 PR-AUC >= 0.0010. v3 reached 0.00096, so the config fell back to full weighting with PR-AUC stopping at 100k negatives, and was **frozen**. No further tuning.
- Most of the gain came from 100k negatives, not from the weighting scheme.
- **Reproducibility fix:** re-running v4 gave different numbers. The pyarrow join returns rows in arbitrary order, so the seeded negative sample changed between runs.
  - Fixed by sorting rows after the join and setting LightGBM `deterministic=True, force_row_wise=True`.
  - Two back-to-back runs are now byte-identical.
  - The official frozen-config result is the run after this fix (section 10).
- **Noise check:** the same frozen config scored between 0.00088 and 0.00118 depending only on which negatives were sampled, which is as wide as the gap between v3 and v4. With 64 validation positives, 2020 cannot reliably separate these configs. This is the main reason not to tune further.

## 9. Sensitivity runs
`--label` reruns the identical frozen procedure on P1's robustness labels:
- same grids, seeds and selection rule
- outputs in `out/<label>/` and `reports/sensitivity/<label>/`
- the primary results are untouched (verified by SHA256)

| Label | Meaning | Decision years | Training positives (distinct) |
| --- | --- | --- | --- |
| `y_B_h2` | Scenario B: all exclusion types, 2 years | val 2020; test 2022, 2023 | 1,064 (matches P1) |
| `y_A_h3` | Scenario A, 3-year window | val 2020; test 2022 | 534 |

## 10. Validation results (2020)
| Label (base rate) | LightGBM PR-AUC | Logistic Regression PR-AUC | Isolation Forest PR-AUC | LightGBM ROC / recall@1% / recall@5% |
| --- | --- | --- | --- | --- |
| `y_A_h2` (5.8e-05) | **0.00097** (17x) | 0.00034 | 0.00009 | 0.806 / 0.14 / 0.33 |
| `y_B_h2` (1.15e-04) | **0.00245** (21x) | 0.00069 | 0.00018 | 0.815 / 0.18 / 0.41 |
| `y_A_h3` (1.16e-04) | **0.00219** (19x) | 0.00065 | 0.00019 | 0.822 / 0.20 / 0.41 |

Chosen LightGBM setting: `num_leaves=31, min_child_samples=200` for all three labels, with 13 / 201 / 138 trees.

**Findings:**
1. **The ordering is the same under every label:** LightGBM > Logistic Regression > Isolation Forest. The conclusion holds across label definition and horizon.
2. **Learning from past exclusions works; unusual billing alone does not.** Isolation Forest is barely above random (recall@1% about 0-2%). The patterns that precede exclusion are specific, not just extreme.
3. **Top LightGBM features are stable across labels:**
   - year-over-year change in Medicare payment (`medicare_payment_yoy_diff` / `_growth`, about 0.2-0.4 of gain)
   - specialty-adjusted payment per beneficiary (`pymt_per_bene_peer_z`)
   - charge-to-allowed ratio
4. **Logistic Regression coefficients should not be read one by one.** Its coefficients are large and switch sign (e.g. `allowed_amount` vs `tot_srvcs`) because the volume and money features are strongly correlated. Use the LightGBM importance to describe what drives risk.

Validation is noisy (section 8). The honest performance estimate is P4's evaluation on 2022/2023 with bootstrap confidence intervals, against P1's payment baselines (Recall@1% 5-8.5%, AUC 0.54-0.61).

## 11. Outputs
| File | Content |
| --- | --- |
| `out/predictions.parquet` | `NPI, year, approach, score`: 10,388,865 rows (3 approaches x 2020/2022/2023); higher = higher priority |
| `out/supervised_model.pkl` | `{"supervised_lgbm", "supervised_logreg"}` bundles: one model per year, settings, label, horizon |
| `out/anomaly_model.pkl` | Isolation Forest bundle |
| `out/y_B_h2/*`, `out/y_A_h3/*` | The same three files for the sensitivity runs |
| `reports/p3_split_summary.csv` | Training years, rows and positives per decision year |
| `reports/p3_scoring_coverage.csv` | Eligible rows with and without features, per year |
| `reports/p3_val_metrics.csv` | 2020 metrics for every setting tried, all approaches |
| `reports/p3_feature_importance.csv` | LightGBM share of gain (mean over test-year models) |
| `reports/p3_logreg_coefficients.csv` | Standardised Logistic Regression coefficients |
| `reports/sensitivity/<label>/` | The same reports for each sensitivity run |

`out/` is gitignored because the files are large. They are shared through the team drive.

## 12. Verification performed
- `split_plan.csv` match, eligibility count, leakage and maturity assertions pass on every run.
- The P1 file SHA256 values match `labels_v1_hashes.txt`.
- The scored pool equals `eval_pool.parquet` row for row, for every approach and year.
- Determinism: repeated runs give identical `p3_val_metrics.csv`, `p3_feature_importance.csv` and `predictions.parquet` (SHA256 `9AB2713B...3031958`).
- All `.pkl` files load with `joblib.load` from the repo root.
- P1 code review found no leakage.
- The stub run (`--data stub`) exercises the full pipeline on synthetic data.

## 13. Limitations
- **Very few positives:** about 60 per year, and 64 in validation. All validation metrics are noisy, and model selection on 2020 is close to a coin flip between similar configs.
- **Labels capture only providers who were caught,** and only about 10% of LEIE records carry an NPI. Undetected cases look like negatives.
- **Scores are not calibrated probabilities,** because of negative sampling and class weighting.
- **Providers without a CMS feature row cannot be scored** (about 5% of eligible rows per year).
- **The final primary LightGBM has only 13 trees**, chosen by early stopping on a noisy year. That produces tied scores, so top-K needs a fixed tie-break.
- **Isolation Forest is fit on each scored year's own features.** These are known at the decision date, so there is no leakage, but its scores are relative to that year's population.
