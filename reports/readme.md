# FraudLens — Data Preprocessing, Engineering and Validation Report

## 1. Project Overview

### 1.1 Problem Statement

FraudLens aims to prioritise individual Medicare Part B healthcare practitioners for potential audit using historical Medicare billing patterns.

The project combines two major public datasets:

1. **CMS Medicare Part B Provider Utilization and Payment Data:** Contains annual provider-level billing information, including services, beneficiaries, submitted charges, allowed amounts and Medicare payments.
2. **OIG List of Excluded Individuals/Entities (LEIE):** Contains healthcare providers excluded from participating in federally funded healthcare programmes.

The central objective is to examine whether historical CMS billing patterns can help identify providers who subsequently receive selected OIG exclusions.

**Important:** An OIG exclusion is an enforcement outcome and is not automatically proof of fraud. FraudLens is an audit-prioritisation system, not an automated fraud-determination system.

### 1.2 Project Observation Period

- CMS data: 2013–2023.
- LEIE main dataset: Updated through November 2026 in the downloaded snapshot.
- Reinstatement supplements: January 2025–October 2026.
- Primary modelling outcome: Scenario A, two-year horizon (`y_A_h2`).
- Validation year: 2020.
- Test years: 2022 and 2023.

### 1.3 Team Responsibilities

| Member | Responsibility |
|---|---|
| P1 — Harsha | LEIE preprocessing, outcome labels, NPI matching, population construction, validation, baseline models and independent prediction checks |
| P2 | CMS data engineering, schema auditing and feature engineering |
| P3 | Machine-learning model development and prediction generation |
| P4 | Model evaluation, performance metrics, statistical comparisons and reporting |

---

# 2. Data Sources and Initial Dataset Findings

## 2.1 OIG LEIE Dataset

The LEIE dataset contains information about healthcare providers excluded from participating in federal healthcare programmes.

Initial profiling identified:

| Measurement | Result |
|---|---:|
| Original records | 84,224 |
| Total columns | 18 |
| Records with NPI = 0 | 75,201 |
| Records with valid 10-digit NPIs | 9,023 |
| Unique valid NPIs | 8,844 |
| Repeated NPI groups | 175 |
| Excess repeated-NPI rows | 179 |
| Exact duplicate exclusion events | 2 |

### Important findings

**1. High proportion of unusable NPIs**

Approximately 89.3% of the original LEIE records contained NPI = 0, which was a placeholder rather than a usable provider identifier.

Consequently, only a relatively small portion of LEIE records could be directly linked to CMS using NPI.

**2. NPI validation was necessary**

A valid NPI must satisfy the 10-digit format and its check-digit validation.

The cleaning process used:
- Ten-digit format validation.
- Healthcare identifier prefix `80840`.
- Luhn check-digit validation.

The reference NPI `1234567893` was used to check the validation implementation.

**3. Repeated provider records**

Repeated NPIs were examined separately from exact duplicate exclusion events because one provider can have multiple exclusion records.

**4. Reinstatement-date placeholder**

The `REINDATE` column contained `"0"` in all 84,224 original records. This was treated as a placeholder rather than an actual reinstatement date.

## 2.2 Reinstatement Supplement Integration

A major limitation of using only the current LEIE file is that reinstated providers may no longer appear in the current exclusion list.

To improve historical coverage, 1,199 monthly reinstatement supplement records covering January 2025–October 2026 were downloaded and incorporated.

### Results after integration

| Measurement | Before supplements | After supplements |
|---|---:|---:|
| Valid-NPI exclusion events | 9,021 | 9,288 |
| Unique excluded NPIs | 8,844 | 9,105 |
| Scenario A unique NPIs | 4,320 | 4,461 |

The integration resulted in a net increase of 267 valid-NPI exclusion events.

**Important distinction:** The 1,199 supplement rows are reinstatement records, not 1,199 additional exclusion events.

Historical reinstatement records before 2025 may still be incomplete.

## 2.3 CMS Provider-Year Dataset

The CMS annual provider files covered 2013–2023.

| Measurement | Result |
|---|---:|
| Annual CMS files | 11 |
| Observation period | 2013–2023 |
| Total provider-year rows | 12,232,194 |
| Unique NPIs | 1,825,405 |
| Duplicate NPI-year combinations | 0 |

The NPI and year were extracted from each annual file to create a consistent provider-year reference population.

All annual record counts were checked against the original source files.

Result: `YEAR COUNTS: ALL MATCH`.

---

# 3. LEIE and CMS Integration Findings

## 3.1 NPI-Based Matching

The two datasets were connected using provider NPI and the relevant year.

Final matching results:

| Population | Unique valid-NPI excluded providers | Found in CMS |
|---|---:|---:|
| All exclusion types | 9,105 | 2,746 |
| Scenario A | 4,461 | 1,236 |

Matching coverage:

- All exclusion types: approximately 30.2%.
- Scenario A: approximately 27.7%.

### Interpretation

Only around 30% of the unique excluded providers with valid NPIs appeared in the available CMS Part B provider population.

This established an important feasibility limitation.

The project could only model providers for whom both historical CMS billing information and usable exclusion information were available.

Unmatched providers should not automatically be interpreted as providers who never billed Medicare. They may fall outside the available programme, observation period or dataset coverage.

## 3.2 Exploratory Name-Based Matching

Additional matching strategies were investigated:

| Method | Additional provider-year matches |
|---|---:|
| Name + State | 397 |
| Name + State + ZIP | 6 |

These matches were not automatically included in the final labels because name-based matching can incorrectly connect different providers with similar names.

The decision prioritised matching reliability over increasing the positive sample size.

---

# 4. Outcome Definition and Label Construction

## 4.1 Scenario A and Scenario B

Not all OIG exclusion categories represent the same type of conduct.

Two outcome scenarios were constructed.

### Scenario A — Primary Outcome

Selected exclusion codes:

`1128a1`, `1128a3`, `1128b1`, `1128b6`, `1128b7`.

This represents selected programme-integrity and billing-related exclusions.

### Scenario B — Sensitivity Outcome

Includes all OIG exclusion types.

Scenario B provides a broader outcome definition to examine whether results change when all exclusion categories are considered.

Neither scenario should be described as a database of proven fraud cases.

## 4.2 Prediction Horizons

Three outcome horizons were created.

| Label | Meaning |
|---|---|
| `y_A_h1` | Scenario A exclusion within the next year |
| `y_A_h2` | Scenario A exclusion within the next two years |
| `y_A_h3` | Scenario A exclusion within the next three years |
| `y_B_h1` | Scenario B exclusion within the next year |
| `y_B_h2` | Scenario B exclusion within the next two years |
| `y_B_h3` | Scenario B exclusion within the next three years |

A positive label is represented by 1 and a negative observed outcome by 0.

## 4.3 Eligibility and Exclusion History

The label-building process also generated fields such as:

- `eligible`
- `excluded_before_year`
- `excluded_in_year`
- `censored_h1`
- `censored_h2`
- `censored_h3`

These fields support eligibility filtering and outcome-window validation.

Providers who were already excluded by the relevant decision year were not treated as ordinary eligible candidates.

## 4.4 Censoring and Temporal Validity

A prediction horizon is usable only when the complete outcome window has elapsed.

The primary outcome selected was:

**`y_A_h2` — Scenario A, two-year horizon.**

The two test years, 2022 and 2023, have complete two-year outcome windows.

However, the three-year outcome for 2023 is incomplete.

Therefore:

- 2022 two-year outcomes: complete.
- 2023 two-year outcomes: complete.
- 2023 three-year outcomes: censored.
- `y_A_h3` and `y_B_h3` for 2023 must not be used for three-year training or evaluation.

Censoring indicators were retained rather than incorrectly treating incomplete observations as fully observed negative outcomes.

---

# 5. CMS Feature Engineering and Population Construction

## 5.1 CMS Schema Audit

P2 audited 81 CMS columns across the 11 annual files.

| Finding | Result |
|---|---:|
| Total columns audited | 81 |
| Core columns consistently available | 56 |
| Chronic-condition columns introduced from 2017 | 25 |

The chronic-condition variables were excluded because they were not consistently available across the full observation period.

Fifteen NPPES-derived fields were also excluded from predictors because they were based on registry snapshots that were not necessarily representative of historical availability.

## 5.2 Individual Practitioner Filtering

The original CMS population contained both individual practitioners and organisations.

The feature pipeline retained individual practitioners using:

`Rndrng_Prvdr_Ent_Cd == 'I'`

Results:

| Measurement | Rows |
|---|---:|
| Original CMS NPI-year population | 12,232,194 |
| Organisational provider-years removed | 680,090 |
| Final individual feature rows | 11,552,104 |

This aligned the modelling population with the project objective of individual-practitioner audit prioritisation.

## 5.3 Feature Dataset

The final master feature dataset was:

`out/provider_features.parquet`

| Measurement | Result |
|---|---:|
| Feature rows | 11,552,104 |
| Total columns | 34 |
| NPI and year identifiers | 2 |
| Feature columns | 32 |
| Duplicate NPI-year combinations | 0 |

P3 used 30 numerical CMS predictors for modelling.

## 5.4 Feature Engineering Concepts

Important feature groups included:

**Volume indicators**
- `tot_benes`
- `tot_srvcs`

**Financial indicators**
- `submitted_charges`
- `allowed_amount`
- `medicare_payment`

**Intensity indicators**
- `pymt_per_bene`
- `srvcs_per_bene`
- `pymt_per_srvc`
- `charge_to_allowed`

**Year-on-year changes**
- Payment differences and growth.
- Submitted-charge differences and growth.
- Service-volume differences and growth.

**Peer-adjusted indicators**
- `pymt_per_bene_peer_z`
- `srvcs_per_bene_peer_z`

Peer comparisons were calculated using year and specialty groups.

Groups with fewer than 30 providers or insufficient variation (IQR <= 1e-4) received missing peer-adjusted values.

## 5.5 Temporal Feature Construction

P2's longitudinal feature code sorted providers by NPI and year, then used the previous observed year to calculate historical changes.

For example:

```python
df = df.sort_values(by=['npi', 'year']).reset_index(drop=True)

npi_grp = df.groupby('npi')

prev_year = npi_grp['year'].shift(1)
is_consecutive = (df['year'] - prev_year) == 1
```

Historical payment and service changes were calculated only when consecutive annual observations existed.

### `is_new_billing_provider`

The feature was calculated as:

```python
df['is_new_billing_provider'] = np.where(
    prev_year.isna(), 1, 0
).astype('int8')
```

This uses the previous observed year and does not use future-year information.

However, since CMS data begins in 2013, providers first observed in 2013 are classified as new even if they had been billing Medicare before 2013.

It should therefore be interpreted as first observed appearance in the available CMS data, not necessarily first-ever Medicare billing.

---

# 6. Final Evaluation-Pool Construction

## 6.1 Why a Separate Evaluation Pool Was Required

The broad candidate population and the final evaluation population are different.

The final evaluation population must contain providers who:

1. Are eligible.
2. Have corresponding engineered feature records.
3. Belong to the individual-practitioner population.

The evaluation-pool definition was:

`eligible == True AND feature row exists`

Final dataset sizes:

| Dataset | Provider-year rows |
|---|---:|
| `labels.parquet` | 12,232,194 |
| `provider_features.parquet` | 11,552,104 |
| `eval_pool.parquet` | 11,551,956 |

The evaluation pool is the population that P3 must score and P4 must evaluate.

## 6.2 Primary Outcome Counts

For Scenario A, two-year outcome (`y_A_h2`):

| Split | Distinct positive NPIs |
|---|---:|
| Training | 461 |
| Validation — 2020 | 64 |
| Test — 2022 and 2023 | 113 |

Test-year details:

| Year | Evaluation pool | Positive provider-year records |
|---|---:|---:|
| 2022 | 1,165,640 | 82 |
| 2023 | 1,196,525 | 58 |
| Combined | 2,362,165 | 140 |

There are 113 distinct positive NPIs across the two test years because some providers appear as positive in both years.

The 2022 positive rate was approximately 0.00703%.

This extreme class imbalance made accuracy an unsuitable primary evaluation metric.

---

# 7. P1 Python Scripts — Detailed Inventory

This section documents the major scripts used for P1's preprocessing, label construction, validation and independent evaluation.

## 7.1 LEIE Profiling and Cleaning

### `01_profile_leie.py`

**Purpose:** Profile the original OIG LEIE file before making any changes.

**What it does:**
- Reads the original LEIE CSV.
- Examines the 18 columns.
- Checks NPI validity and missing-value patterns.
- Profiles exclusion dates.
- Identifies repeated NPIs.
- Checks exact duplicate exclusion events.
- Examines reinstatement-date values.

**Why it was used:**

To understand the source-data problems before designing the cleaning process.

**Important results:**
- 84,224 original rows.
- 75,201 records with NPI = 0.
- 9,023 valid-NPI records.
- 8,844 unique valid NPIs.
- 175 repeated NPI groups.
- 2 exact duplicate exclusion events.
- `REINDATE` was zero in all original rows.

**Main output:** LEIE profiling statistics and diagnostic reports.

### `02_clean_leie.py`

**Purpose:** Clean the LEIE source data and construct usable exclusion events.

**What it does:**
- Standardises the NPI field.
- Validates ten-digit NPIs.
- Applies NPI check-digit validation.
- Handles unusable placeholder values.
- Normalises relevant exclusion fields.
- Handles repeated exclusion records.
- Produces cleaned datasets and audit reports.

**Why it was used:**

To create reliable provider identifiers and exclusion-event records before joining LEIE to CMS.

**Main outputs:**
- `out/leie_clean.parquet`
- `out/leie_npi_events.parquet`
- `reports/leie_cleaning_summary.csv`
- Duplicate and data-quality reports.

### `03_profile_exclusion_types.py`

**Purpose:** Examine the distribution of OIG exclusion categories.

**What it does:**
- Groups records by exclusion-type codes.
- Examines frequency distributions.
- Helps distinguish selected billing/programme-integrity categories from other exclusion categories.

**Why it was used:**

To design two defensible outcome definitions rather than treating all exclusion categories identically.

**Result:**
- Supported Scenario A and Scenario B construction.
- Scenario A focused on five selected exclusion codes.
- Scenario B included all exclusion types.

### `05_build_labels.py`

**Purpose:** Construct provider-year outcome labels by combining OIG exclusion events and the CMS NPI-year population.

**What it does:**
- Reads cleaned LEIE records and reinstatement supplements.
- Reads the CMS NPI-year dataset.
- Associates exclusion events with provider-years.
- Identifies providers already excluded.
- Constructs eligibility indicators.
- Constructs Scenario A and Scenario B labels.
- Creates one-, two- and three-year outcome horizons.
- Calculates censoring indicators.
- Generates split and label-audit information.

**Important generated fields:**
- `eligible`
- `excluded_before_year`
- `excluded_in_year`
- `y_A_h1`, `y_A_h2`, `y_A_h3`
- `y_B_h1`, `y_B_h2`, `y_B_h3`
- `censored_h1`, `censored_h2`, `censored_h3`

**Why it was used:**

Machine-learning models require a clearly defined target variable. This script converted exclusion events into future provider-year outcomes.

**Main outputs:**
- `out/labels.parquet`
- `out/candidate_pool.parquet`
- `out/label_audit.csv`
- `out/split_plan.csv`

### `06_npi_coverage_check.py`

**Purpose:** Examine whether excluded providers can be found in the CMS provider population.

**What it checks:**
- Valid-NPI coverage.
- Unique excluded NPIs.
- CMS matching coverage.
- Scenario-specific matching.

**Why it was used:**

To quantify whether LEIE could provide usable supervised-learning outcomes for the available CMS population.

**Important final findings:**
- All exclusion types: 2,746 of 9,105 unique valid-NPI providers found in CMS.
- Scenario A: 1,236 of 4,461 unique providers found in CMS.

The matching analysis also explored name-based matching, but those additional matches were not automatically accepted.

## 7.2 CMS Provider-Year Construction and Validation

### `07_build_cms_npi_year.py`

**Purpose:** Extract provider identifiers and reporting years from all 11 annual CMS files.

**What it does:**
- Reads CMS annual files.
- Extracts NPI and year.
- Processes large files in chunks.
- Constructs the combined provider-year dataset.
- Exports the data into Parquet format.

**Why it was used:**

The LEIE and CMS datasets needed a consistent provider-year reference population for matching and label construction.

**Main output:**
- `out/cms_npi_year.parquet`

**Final result:**
- 12,232,194 provider-year rows.
- 1,825,405 unique NPIs.
- 11 annual files from 2013–2023.

### `08_check_cms_npi_year.py`

**Purpose:** Validate the combined CMS NPI-year dataset.

**What it checks:**
- Annual record counts.
- Duplicate NPI-year combinations.
- Overall row counts.
- Year coverage.

**Why it was used:**

To establish that the extraction process had not silently omitted or duplicated annual records.

**Result:**
- Annual counts matched the original CMS files.
- No duplicate NPI-year combinations.
- `YEAR COUNTS: ALL MATCH`.

### `09_spot_check_labels.py`

**Purpose:** Manually inspect sampled label records.

**What it checks:**
- Selected provider records.
- Their associated exclusion information.
- Whether dates fall within the expected label window.

**Why it was used:**

Automated label creation can contain date or matching errors. Spot checks provide an independent validation of selected observations.

**Additional validation evidence:**
- 8 of 8 sampled exclusion dates were found in the expected outcome window.
- 3 of 3 manually checked NPIs matched the NPI Registry.

These checks support the reliability of the selected identifiers and labels, although they are sample checks rather than an exhaustive audit.

### `11_validate_labels.py`

**Purpose:** Validate label construction and outcome-window completeness.

**What it checks:**
- Label distributions.
- Eligibility.
- Positive-outcome counts.
- Censoring flags.
- Completeness of different prediction horizons.

**Why it was used:**

To ensure that incomplete outcome windows were not incorrectly treated as fully observed negative outcomes.

**Important result:**
- One- and two-year horizons were fully observed for the relevant final evaluation years.
- The 2023 three-year outcome was identified as censored.
- The primary two-year test outcomes for 2022 and 2023 were complete.

**Important distinction:** This is the label-validation script. It is not the prediction-integrity checker.

## 7.3 Feature Integration and Evaluation-Pool Construction

### `12_check_features_join.py`

**Purpose:** Check the integration between P2's feature dataset and P1's labels.

**What it checks:**
- Feature row count.
- Label row count.
- Duplicate NPI-year pairs.
- Feature coverage by year.
- Positive labels without corresponding feature rows.
- Suspicious feature-column names.
- Simple single-feature AUC diagnostics for selected years.

**Why it was used:**

To verify that the engineered features could be joined to the outcome labels without introducing duplicate records or losing important positive cases.

**Important results:**
- Feature rows: 11,552,104.
- Label rows: 12,232,194.
- Duplicate feature NPI-year pairs: 0.
- Annual feature coverage was approximately 93.8%–95.0% of the corresponding full CMS provider-year population.
- A small number of positive provider-year observations had no feature records.
- No suspicious feature-column names were identified in the scan.

The script also performed preliminary univariate AUC checks. These were diagnostic feature-signal checks, not final model-performance results.

### `13_build_eval_pool.py`

**Purpose:** Construct the final eligible individual-practitioner evaluation dataset.

**Population definition:**

`eligible == True AND feature row exists`

**What it does:**
- Reads labels.
- Reads available feature keys.
- Filters the candidate population to eligible records with features.
- Creates the final evaluation pool.
- Audits positive-case retention.

**Why it was used:**

P3 and P4 needed exactly the same population for training, scoring and evaluation.

**Main outputs:**
- `out/eval_pool.parquet`
- `out/label_audit_pool.csv`

**Final result:**
- 11,551,956 evaluation-pool rows.
- Only eligible individual practitioners with available features.
- Positive-provider counts were checked for every relevant split and outcome.

### `14b_censoring_audit.py`

**Purpose:** Independently audit the completeness of the one-, two- and three-year outcome horizons.

**Why it was used:**

To ensure that a model is not trained or tested against labels whose outcome windows have not completely elapsed.

**Important result:**
- The two-year 2022 and 2023 test outcomes were complete.
- The 2023 three-year outcomes were censored.
- The censoring indicators were retained rather than silently converting incomplete observations into fully observed negative outcomes.

### `05_recheck.py`

**Purpose:** Rebuild the label output separately as an additional reproducibility check.

**What it does:**
- Re-runs the label-building process.
- Saves a separate set of outputs for comparison.
- Avoids overwriting the frozen primary output.

**Why it was used:**

To establish whether the original label file could be reconstructed independently from the available source data and current script.

### `15_recheck_labels.py`

**Purpose:** Compare the original labels with the independently rebuilt labels.

**What it checks:**
- Dataset shapes.
- Column equality.
- Row-level content equality.

**Result:**
- Original and recheck shapes were identical: 12,232,194 rows × 14 columns.
- Columns were identical.
- `DataFrame.equals()` returned `True`.

This established that the rebuilt label dataset was identical to the original output.

## 7.4 Independent Baseline and Prediction Evaluation

### `16_p1_baseline_predictions.py`

**Purpose:** Generate independent baseline predictions without training a machine-learning model.

**Two baseline approaches:**

1. `pymt_per_bene_peer_z`
2. `srvcs_per_bene_peer_z`

**What it does:**
- Reads P2's provider features.
- Selects test years 2022 and 2023.
- Joins them to the frozen evaluation pool.
- Uses the two peer-adjusted indicators as ranking scores.
- Assigns missing scores the minimum available feature value so they rank last.
- Saves baseline predictions.

**Why it was used:**

A complex machine-learning model should be compared with a simpler reference method. This reveals whether the model adds value beyond individual billing indicators.

**Main output:**
- `out/p1_baseline_predictions.parquet`

**Result:**
- 2,362,165 test provider-year observations.
- Two baseline approaches.
- 4,724,330 prediction rows.
- 115 missing feature values per approach, handled by assigning the minimum available score.

### `14_check_predictions.py`

**Purpose:** Independently validate model predictions and recompute performance metrics.

**What it checks:**
1. Duplicate NPI-year-approach predictions.
2. Missing prediction scores.
3. Predictions outside the evaluation pool.
4. Predictions without matching label records.
5. Predictions for ineligible providers.
6. Coverage of the complete test population.

It then independently calculates:

- ROC-AUC.
- Recall@1%.
- Recall@5%.
- Lift@1%.
- Lift@5%.

**Why it was used:**

P1 wanted to independently verify P3's predictions instead of relying only on performance numbers reported by P3 or P4.

**Important update — deterministic tie-breaking:**

The checker was updated to use P4's agreed ranking rule:

1. Prediction score descending.
2. Ten-digit NPI ascending when scores are tied.
3. Select exactly `ceil(percentage × evaluation pool size)` observations.

The implementation uses:

```python
npi = d["npi"].astype(str).str.strip().str.zfill(10).to_numpy()

o = np.lexsort((npi, -s))

k1 = int(np.ceil(len(pool) * 0.01))
k5 = int(np.ceil(len(pool) * 0.05))

rec1 = yv[o[:k1]].sum() / P
rec5 = yv[o[:k5]].sum() / P
```

This ensures deterministic, reproducible top-percentage selection.

**Final integrity results:**

| Check | Result |
|---|---:|
| Duplicate predictions | 0 |
| Missing scores | 0 |
| Predictions outside evaluation pool | 0 |
| Predictions without matching labels | 0 |
| Predictions for ineligible providers | 0 |
| Evaluation coverage | 100% |

**Important final reconciliation:**

The original checker produced 18/82 positive cases for 2022 LightGBM Recall@1%. Following P4's deterministic NPI tie-breaking rule, the agreed result became 19/82 = 23.2%.

All other AUC and recall figures matched P4 after rounding.

---

# 8. P2 Python Files — CMS Feature Engineering

These files were developed by P2 and reviewed as part of the team pipeline.

### `build_provider_features.py`

**Purpose:** Convert raw annual CMS provider information into engineered numerical model features.

**Major operations:**
- Reads the annual CMS provider files.
- Filters to individual practitioners.
- Extracts financial, volume and service information.
- Creates per-beneficiary and per-service indicators.
- Calculates year-on-year differences and growth.
- Constructs peer-adjusted indicators.
- Creates the historical new-billing-provider indicator.
- Exports the master feature dataset.

**Main output:**
- `out/provider_features.parquet`

**Validation findings:**
- 11,552,104 rows.
- 34 columns: two identifiers and 32 retained/engineered features.
- Zero duplicate NPI-year pairs.
- P3 selected 30 numerical CMS predictors.

### `build_cms_npi_year.py`

**Purpose:** Build the CMS provider-year identifier dataset.

This was another version of the provider-year extraction pipeline reviewed within P2's work.

The initial review identified potential issues involving:
- Year extraction from the complete path instead of the filename.
- Zero-padding invalid short NPIs into apparent ten-digit values.
- Silently continuing when an annual file failed.

The resulting CMS NPI-year dataset was subsequently checked against annual source counts.

### `docks/FEATURE_DICTIONARY.md`

**Purpose:** Document the CMS feature names, descriptions, source fields, calculations and intended modelling use.

It helped P1 and P3 understand what each predictor represented and whether it was appropriate for historical modelling.

### `reports/cms_column_audit.csv`

**Purpose:** Document CMS column availability across the 11 annual files.

It identified the consistently available core fields and the chronic-condition fields introduced in later years.

### Additional utilities

| File | Purpose |
|---|---|
| `src/merge_features_labels.py` | Supports combining feature and label datasets |
| `src/explain_top_providers.py` | Supports explanation of high-ranked providers |

---

# 9. P3 Python File — Model Training

### `src/p3_models.py`

**Purpose:** Train and generate predictions from three model approaches.

| Model | Approach |
|---|---|
| Logistic Regression | Supervised linear classification model |
| LightGBM | Supervised gradient-boosted tree model |
| Isolation Forest | Unsupervised anomaly-detection model |

### Leakage and training checks reviewed by P1

The 634-line training script was reviewed against six important conditions:

| Check | Result |
|---|---|
| Training years conform to `split_plan.csv` | Passed |
| Only eligible population rows are used | Passed |
| No identifiers, dates or label fields used as predictors | Passed |
| Negative sampling restricted to training data | Passed |
| Preprocessing fitted on training data only | Passed |
| Tuning restricted to 2020 validation data | Passed |

P3 also confirmed:

- All six SHA256 fingerprints matched P1's frozen files.
- The scored pool matched `eval_pool.parquet`.
- All 11,551,956 eligible evaluation-pool rows were scored.
- Predictions were generated for 2020, 2022 and 2023.
- 2022 and 2023 test metrics were not used during model tuning.

### Prediction output

`out/predictions.parquet`

This file contains:
- NPI
- Year
- Approach
- Prediction score

It was subsequently independently evaluated using P1's `14_check_predictions.py`.

---

# 10. Final Model Results and Baseline Comparison

The primary outcome was Scenario A, two-year horizon (`y_A_h2`).

Test years: 2022 and 2023.

## 10.1 Baseline results

| Approach | Year | AUC | Recall@1% | Recall@5% |
|---|---:|---:|---:|---:|
| Payment per beneficiary | 2022 | 0.541 | 8.5% | 19.5% |
| Services per beneficiary | 2022 | 0.570 | 7.3% | 14.6% |
| Payment per beneficiary | 2023 | 0.588 | 5.2% | 19.0% |
| Services per beneficiary | 2023 | 0.612 | 5.2% | 19.0% |

## 10.2 Machine-learning results

| Model | Year | AUC | Recall@1% | Recall@5% |
|---|---:|---:|---:|---:|
| Isolation Forest | 2022 | 0.678 | 7.3% | 15.9% |
| Isolation Forest | 2023 | 0.687 | 6.9% | 24.1% |
| LightGBM | 2022 | 0.790 | 23.2% | 40.2% |
| LightGBM | 2023 | 0.745 | 10.3% | 34.5% |
| Logistic Regression | 2022 | 0.819 | 14.6% | 42.7% |
| Logistic Regression | 2023 | 0.756 | 10.3% | 37.9% |

### Main findings

1. Both supervised models outperformed the single-feature baselines.
2. Logistic Regression achieved the highest AUC in both test years.
3. LightGBM achieved the highest Recall@1% in 2022.
4. Logistic Regression achieved the highest Recall@5% in both years.
5. Isolation Forest identified useful billing anomalies without using exclusion labels.
6. The paired, NPI-clustered bootstrap comparison did not establish a clear winner between LightGBM and Logistic Regression.

**Final recommendation:** Pilot both supervised models rather than claiming one is statistically superior.

---

# 11. Feature-Importance Findings

P3 generated feature-importance and coefficient reports.

P1 independently inspected and ranked the reported features.

## 11.1 LightGBM — Top 10 Features

| Rank | Feature | Share of gain |
|---:|---|---:|
| 1 | `medicare_payment_yoy_diff` | 40.37% |
| 2 | `charge_to_allowed` | 11.52% |
| 3 | `pymt_per_bene_peer_z` | 11.50% |
| 4 | `pymt_per_bene` | 5.49% |
| 5 | `pymt_per_bene_yoy_growth` | 3.56% |
| 6 | `pymt_per_srvc` | 2.95% |
| 7 | `srvcs_per_bene_peer_z` | 2.90% |
| 8 | `submitted_per_bene` | 2.63% |
| 9 | `submitted_charges` | 2.53% |
| 10 | `tot_benes_yoy_diff` | 2.02% |

**Finding:** Year-on-year Medicare payment difference was the dominant LightGBM feature, accounting for approximately 40.37% of total gain.

The top ten features accounted for approximately 85.47% of total gain.

Feature gain represents model contribution to tree-split improvements, not causation.

## 11.2 Logistic Regression — Top 10 Features

| Rank | Feature | Standardised coefficient |
|---:|---|---:|
| 1 | `pymt_per_bene` | +7.243 |
| 2 | `allowed_amount` | +7.020 |
| 3 | `tot_srvcs` | -6.133 |
| 4 | `pymt_per_srvc` | -4.722 |
| 5 | `submitted_per_bene` | -4.341 |
| 6 | `charge_to_allowed` | +2.990 |
| 7 | `srvcs_per_bene` | -2.058 |
| 8 | `submitted_charges` | -1.916 |
| 9 | `log_medicare_payment` | -1.828 |
| 10 | `medicare_payment` | +1.474 |

Coefficients represent conditional associations after accounting for the other predictors. They should not be interpreted as causal relationships.

## 11.3 `is_new_billing_provider` Review

P1 checked this feature's implementation and confirmed that it uses only the previous observed year.

It was not among the displayed top ten features in either model.

Therefore, no sensitivity rerun was required under the agreed review rule.

The 2013 first-observation limitation was retained as a documentation caveat.

---

# 12. Reproducibility and Team Data Governance

P1 created SHA256 fingerprints for six frozen outputs:

1. `labels.parquet`
2. `candidate_pool.parquet`
3. `split_plan.csv`
4. `label_audit.csv`
5. `eval_pool.parquet`
6. `label_audit_pool.csv`

The fingerprints were stored in:

`out/labels_v1_hashes.txt`

P3 calculated the hashes of the files actually used for model training and confirmed that all six matched.

This established that P1 and P3 used identical frozen data versions.

Additional validation:
- Labels were rebuilt and compared successfully.
- The final evaluation pool was shared with the team.
- Predictions were independently checked.
- The deterministic ranking rule was reconciled with P4.
- P4's evaluation results matched P1's calculations after rounding and tie handling.

The large Parquet datasets and raw CMS/LEIE files were retained outside GitHub and shared through the team drive. GitHub was used for code, documentation and selected reports/figures.

---

# 13. Final Output File Inventory

## 13.1 Main Data Outputs

| File | Purpose |
|---|---|
| `out/leie_clean.parquet` | Cleaned LEIE records |
| `out/leie_npi_events.parquet` | Valid-NPI exclusion events |
| `out/cms_npi_year.parquet` | CMS NPI-year population |
| `out/labels.parquet` | Provider-year outcome labels |
| `out/candidate_pool.parquet` | Broader eligible candidate population |
| `out/provider_features.parquet` | Engineered CMS feature dataset |
| `out/eval_pool.parquet` | Final eligible individual-practitioner evaluation population |
| `out/predictions.parquet` | P3 model predictions |
| `out/p1_baseline_predictions.parquet` | P1 independent baseline predictions |

## 13.2 Validation and Audit Outputs

| File | Purpose |
|---|---|
| `out/label_audit.csv` | Label validation summary |
| `out/label_audit_pool.csv` | Evaluation-pool audit |
| `out/split_plan.csv` | Temporal training/validation/test split |
| `out/labels_v1_hashes.txt` | SHA256 fingerprints of frozen datasets |

## 13.3 Reports and Figures

| File | Purpose |
|---|---|
| `reports/leie_cleaning_summary.csv` | LEIE cleaning findings |
| `reports/cms_column_audit.csv` | CMS schema consistency audit |
| `reports/p1_baseline_results.csv` | P1 baseline performance |
| `reports/p3_feature_importance.csv` | LightGBM feature importance |
| `reports/p3_logreg_coefficients.csv` | Logistic Regression coefficients |
| `reports/fig1_split_timeline.png` | Temporal split visualisation |
| `reports/fig2_positives_per_year.png` | Positive-case distribution |
| `reports/fig3_independent_check.png` | Draft independent model-comparison figure |

P4's final evaluation results and confidence intervals are maintained separately in the team's evaluation deliverables.

---

# 14. Main Research Limitations

1. **Incomplete LEIE identifiers:** Most original records did not have usable NPIs, restricting direct matching to CMS.
2. **Historical reinstatements:** Older reinstatement information before 2025 may be missing.
3. **Limited dataset coverage:** The analysis covers Medicare fee-for-service Part B, not every healthcare programme.
4. **Outcome proxy:** OIG exclusions are not equivalent to proven fraud.
5. **Rare positive cases:** The positive-outcome rate is extremely low, producing uncertainty in performance estimates.
6. **Historical coverage:** `is_new_billing_provider` cannot identify providers who billed Medicare before 2013.
7. **COVID-related distortion:** Unusual vaccine-related fee-for-service payments in 2021 may affect payment and year-on-year growth features.
8. **NPI-level repeated observations:** Providers appearing in both test years are related observations; bootstrap comparisons therefore account for provider-level clustering.
9. **Limited test periods:** Only two later years were available for the final test evaluation.

---

# 15. Overall Conclusion

The preprocessing and validation work established a usable longitudinal dataset linking historical Medicare Part B billing behaviour with subsequent OIG exclusion outcomes.

The project addressed several major data challenges:

- Missing and invalid provider identifiers in LEIE.
- Reinstatement history.
- Cross-dataset matching limitations.
- CMS schema differences across years.
- Individual versus organisational provider populations.
- Rare positive outcomes.
- Incomplete prediction horizons.
- Potential temporal leakage.
- Reproducibility of labels and evaluation datasets.

The final evaluation population consisted of 11,551,956 eligible individual-practitioner provider-year records.

P1 independently established baseline rankings, reviewed P3's model-training methodology, verified frozen-data fingerprints and independently recomputed the final test metrics.

Both supervised models outperformed the simple billing-based baselines. Logistic Regression achieved the highest AUC point estimates, while LightGBM performed particularly well at the 2022 top-1% audit threshold.

However, paired bootstrap comparisons did not establish a statistically clear winner between the two supervised approaches.

The recommended next step is to pilot both models as decision-support tools for audit prioritisation, with human investigation remaining essential.

**Overall contribution:** The work established not only a dataset suitable for modelling, but also a validated, reproducible and temporally consistent evaluation framework for examining the usefulness of Medicare billing indicators in predicting subsequent OIG exclusion outcomes.