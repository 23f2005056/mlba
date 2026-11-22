# **FraudLens — Feature Dictionary & Provenance Specification**

**Deliverable ID:** CP-11  
**Dataset:** `out/provider_features.parquet` (11,552,104 rows, 34 columns)  
**Sample Dataset:** `out/provider_features_sample_2pct.parquet` (139,104 rows)  
**Primary Key:** `(npi, year)`  
**Population:** Individual Practitioners only (`Rndrng_Prvdr_Ent_Cd == 'I'`)  
**Temporal Coverage:** Reporting Years 2013–2023  

---

## **1. Executive Architecture & Leakage Safeguards**

1. **Strict Causal Ordering (Zero Lookahead):**
   * Features for reporting year $T$ are computed strictly using claims information from year $T$ and historical lags ($T-1$).
   * For the baseline year **2013**, all longitudinal lag and Year-over-Year (YoY) velocity features are strictly `NaN`.
   * **Consecutive Year Requirement:** YoY metrics require that $year_t - year_{t-1} = 1$. If a provider billed in 2018 and 2020 but was inactive in 2019, 2020 is **not** compared against 2018; its YoY features are safely assigned `NaN`.

2. **Specialty Provenance & Peer Calculations:**
   * `specialty` is derived from `Rndrng_Prvdr_Type`. In CMS Part B summary files, this field is derived from Medicare Part B claims billing (the primary specialty code associated with the largest service volume during that calendar year). It is **not** an NPPES registry post-hoc snapshot.
   * All peer medians and Interquartile Ranges (IQRs) are computed **strictly within each calendar year independently**.
   * **Small Group Threshold:** Specialty peer groups with fewer than 30 providers in a given year ($N < 30$) or where $\text{IQR} \le 10^{-4}$ are assigned `NaN` to prevent spurious variance artifacts.

3. **NPPES Exclusion & Demographics:**
   * All 15 static NPPES registry columns (`Rndrng_Prvdr_First_Name`, `Rndrng_Prvdr_Last_Org_Name`, `Rndrng_Prvdr_City`, `Rndrng_Prvdr_Zip5`, `Rndrng_Prvdr_RUCA`, etc.) are completely excluded from predictive modeling to prevent temporal contamination.
   * The `state` column is included for regional audit stratification and reporting only; it must **not** be used as a raw predictive feature without proper temporal modeling.

4. **Reconciliation of Development Sample (139,104 rows):**
   * The fast-development sample (`provider_features_sample_2pct.parquet`) is drawn as a **reproducible 2.000% sample strictly from the historical training split (2013–2019)**:
     $$\text{Training Provider-Years (2013–2019)} = 6,955,202$$
     $$6,955,202 \times 0.02 = 139,104 \text{ rows}$$
   * The validation (2020) and test (2022–2023) periods are deliberately excluded from this sample to prevent test contamination during baseline prototyping.

---

## **2. Complete 34-Feature Specification**

| Column Name | Category | Source Column(s) | Data Type | Mathematical Formula / Derivation | Missingness / Null Handling | Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `npi` | Key | `Rndrng_NPI` | `string` | 10-digit zero-padded string, validated against Luhn/NPI checksum. | 0% nulls | **Primary Key** |
| `year` | Key | Filename `D{YY}` | `int32` | Reporting calendar year (2013–2023). | 0% nulls | **Temporal Key** |
| `specialty` | Context | `Rndrng_Prvdr_Type` | `string` | Primary Medicare billing specialty code from Part B claims. | `Unknown` if missing | **Stratification / Context** |
| `state` | Metadata | `Rndrng_Prvdr_State_Abrvtn` | `string` | Two-letter state postal code from provider enrollment. | `XX` if missing | **Audit Metadata (Exclude)** |
| `medicare_participating` | Operational | `Rndrng_Prvdr_Mdcr_Prtcptg_Ind` | `int8` | $1$ if provider participates in Medicare assignment (`'Y'`), else $0$. | 0% nulls | **Predictor** |
| `tot_benes` | Volume | `Tot_Benes` | `int64` | Total distinct Medicare beneficiaries receiving care (minimum 11). | 0% nulls | **Predictor** |
| `tot_srvcs` | Volume | `Tot_Srvcs` | `float32` | Total number of Part B services/procedures rendered. | 0% nulls | **Predictor** |
| `submitted_charges` | Financial | `Tot_Sbmtd_Chrg` | `float32` | Aggregate dollar amount billed by provider to Medicare. | 0% nulls | **Predictor** |
| `allowed_amount` | Financial | `Tot_Mdcr_Alowd_Amt` | `float32` | Aggregate Medicare fee schedule allowed amount. | 0% nulls | **Predictor** |
| `medicare_payment` | Financial | `Tot_Mdcr_Pymt_Amt` | `float32` | Net amount paid by Medicare Part B to the provider (min \$0.00). | 0% nulls | **Predictor** |
| `pymt_per_bene` | Ratio | Calculated | `float32` | $\frac{\text{medicare\_payment}}{\text{tot\_benes}}$ | `NaN` if `tot_benes` $\le 0$ | **Predictor** |
| `srvcs_per_bene` | Ratio | Calculated | `float32` | $\frac{\text{tot\_srvcs}}{\text{tot\_benes}}$ | `NaN` if `tot_benes` $\le 0$ | **Predictor** |
| `pymt_per_srvc` | Ratio | Calculated | `float32` | $\frac{\text{medicare\_payment}}{\text{tot\_srvcs}}$ | `NaN` if `tot_srvcs` $\le 0$ | **Predictor** |
| `submitted_per_bene`| Ratio | Calculated | `float32` | $\frac{\text{submitted\_charges}}{\text{tot\_benes}}$ | `NaN` if `tot_benes` $\le 0$ | **Predictor** |
| `charge_to_allowed` | Ratio | Calculated | `float32` | $\frac{\text{submitted\_charges}}{\text{allowed\_amount}}$ (Billing markup ratio). | `NaN` if `allowed` $\le 0$ | **Predictor** |
| `allowed_to_pymt` | Ratio | Calculated | `float32` | $\frac{\text{allowed\_amount}}{\text{medicare\_payment}}$ (Reimbursement efficiency). | `NaN` if `payment` $\le 0$ | **Predictor** |
| `pymt_to_charge` | Ratio | Calculated | `float32` | $\frac{\text{medicare\_payment}}{\text{submitted\_charges}}$ (Realization rate). | `NaN` if `charge` $\le 0$ | **Predictor** |
| `log_medicare_payment`| Non-linear | Calculated | `float32` | $\log(1 + \max(0, \text{medicare\_payment}))$ | $0.0$ if payment $\le 0$ | **Predictor** |
| `log_tot_benes` | Non-linear | Calculated | `float32` | $\log(1 + \max(0, \text{tot\_benes}))$ | $0.0$ if benes $\le 0$ | **Predictor** |
| `log_tot_srvcs` | Non-linear | Calculated | `float32` | $\log(1 + \max(0, \text{tot\_srvcs}))$ | $0.0$ if services $\le 0$ | **Predictor** |
| `pymt_per_bene_peer_z`| Peer Anomaly | Calculated | `float32` | $\frac{X - \text{Median}_{\text{spec}, \text{year}}}{0.7413 \times \text{IQR}_{\text{spec}, \text{year}}}$ | `NaN` if $N < 30$ or $\text{IQR} = 0$ | **Primary Anomaly Feature** |
| `srvcs_per_bene_peer_z`| Peer Anomaly | Calculated | `float32` | $\frac{X - \text{Median}_{\text{spec}, \text{year}}}{0.7413 \times \text{IQR}_{\text{spec}, \text{year}}}$ | `NaN` if $N < 30$ or $\text{IQR} = 0$ | **Primary Anomaly Feature** |
| `charge_to_allowed_peer_z`| Peer Anomaly | Calculated | `float32` | $\frac{X - \text{Median}_{\text{spec}, \text{year}}}{0.7413 \times \text{IQR}_{\text{spec}, \text{year}}}$ | `NaN` if $N < 30$ or $\text{IQR} = 0$ | **Primary Anomaly Feature** |
| `medicare_payment_yoy_diff`| Dynamic | Calculated | `float32` | $\Delta_t = \text{payment}_t - \text{payment}_{t-1}$ ($year_t - year_{t-1} = 1$) | `NaN` for 2013 or gaps | **Predictor** |
| `medicare_payment_yoy_growth`| Dynamic | Calculated | `float32` | $\frac{\text{payment}_t - \text{payment}_{t-1}}{\lvert \text{payment}_{t-1} \rvert + 100.0}$ | `NaN` for 2013 or gaps | **Predictor** |
| `tot_benes_yoy_diff` | Dynamic | Calculated | `float32` | $\Delta_t = \text{benes}_t - \text{benes}_{t-1}$ ($year_t - year_{t-1} = 1$) | `NaN` for 2013 or gaps | **Predictor** |
| `tot_benes_yoy_growth` | Dynamic | Calculated | `float32` | $\frac{\text{benes}_t - \text{benes}_{t-1}}{\lvert \text{benes}_{t-1} \rvert + 10.0}$ | `NaN` for 2013 or gaps | **Predictor** |
| `tot_srvcs_yoy_diff` | Dynamic | Calculated | `float32` | $\Delta_t = \text{srvcs}_t - \text{srvcs}_{t-1}$ ($year_t - year_{t-1} = 1$) | `NaN` for 2013 or gaps | **Predictor** |
| `tot_srvcs_yoy_growth` | Dynamic | Calculated | `float32` | $\frac{\text{srvcs}_t - \text{srvcs}_{t-1}}{\lvert \text{srvcs}_{t-1} \rvert + 10.0}$ | `NaN` for 2013 or gaps | **Predictor** |
| `submitted_charges_yoy_diff`| Dynamic | Calculated | `float32` | $\Delta_t = \text{charges}_t - \text{charges}_{t-1}$ ($year_t - year_{t-1} = 1$) | `NaN` for 2013 or gaps | **Predictor** |
| `submitted_charges_yoy_growth`| Dynamic | Calculated | `float32` | $\frac{\text{charges}_t - \text{charges}_{t-1}}{\lvert \text{charges}_{t-1} \rvert + 100.0}$ | `NaN` for 2013 or gaps | **Predictor** |
| `pymt_per_bene_yoy_diff`| Dynamic | Calculated | `float32` | $\Delta_t = \text{ppb}_t - \text{ppb}_{t-1}$ ($year_t - year_{t-1} = 1$) | `NaN` for 2013 or gaps | **Predictor** |
| `pymt_per_bene_yoy_growth`| Dynamic | Calculated | `float32` | $\frac{\text{ppb}_t - \text{ppb}_{t-1}}{\lvert \text{ppb}_{t-1} \rvert + 10.0}$ | `NaN` for 2013 or gaps | **Predictor** |
| `is_new_billing_provider`| Operational | Calculated | `int8` | $1$ if provider has no prior Medicare billing record, else $0$. | 0% nulls | **Predictor** |

---

## **3. Usage Guidelines for P3 (Modeling) & P4 (Evaluation)**

1. **Handling Missing Values (`NaN`):**
   * **Tree Models (LightGBM / XGBoost):** Can ingest `NaN` values directly without imputation. Tree splitters naturally route missing historical lags to optimal default branches.
   * **Linear Models (Logistic Regression):** Impute missing peer Z-scores and YoY diffs using the **training set median** ($\text{median} = 0.0$ for Z-scores). Fit imputation scalers strictly on the training set.
2. **Metadata Column Dropping:**
   * Before model training, P3 must drop `['npi', 'year', 'state', 'specialty']` from the feature matrix $X$ (or one-hot encode top 30 specialties if desired, but exclude `state` to prevent regional bias).
3. **Split Plan Integrity:**
   * Never filter years manually. Merge against `split_plan.csv` to ensure training adheres to label maturity ($y + h \le T$).
