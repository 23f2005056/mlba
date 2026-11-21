# **FraudLens — 4-Person Work Division**

**Project:** Machine Learning-Based Prioritisation of Medicare Providers for Audit

**Objective:** Identify the top K healthcare providers whose historical billing behaviour and exclusion-related risk warrant further investigation.

The division below is designed so all four members can work simultaneously, with minimal dependencies.

## **1. Division of responsibilities**

**Person 1 — You**

Data and labels

**Responsibility:** CMS–LEIE matching and label validation.

Tasks:

1. Clean the LEIE dataset.
2. Verify exclusion dates and NPI quality.
3. Finalise Scenario A (primary) and Scenario B (robustness check).
4. Investigate reinstatements and historical completeness.
5. Remove providers already excluded before their decision year.
6. Calculate distinct positive providers in validation and test.
7. Implement the three-year label-maturity logic.

**Deliverables:**

```
leie_clean.parquet
labels.parquet
candidate_pool.parquet
label_audit.csv
```

**Person 2**

CMS and feature engineering

**Responsibility:** Prepare provider-level financial and operational features.

Tasks:

1. Validate all seven CMS annual files.
2. Check duplicate NPI-year records.
3. Clean provider identifiers and missing values.
4. Calculate payment per beneficiary.
5. Calculate services per beneficiary.
6. Calculate charge-to-allowed ratio.
7. Calculate peer-group deviations by specialty.
8. Calculate year-on-year changes.

**Deliverables:**

```
provider_features.parquet
cms_quality_report.csv
feature_dictionary.csv
```

**Person 3**

Machine learning

**Responsibility:** Develop the two ranking approaches.

**Approach A — Supervised ML**

- Logistic Regression.
- Gradient Boosting, if time permits.
- Predict future exclusion probability using Scenario A labels.

**Approach B — Unsupervised anomaly detection**

- Isolation Forest.
- Identify unusual provider billing patterns without using LEIE labels.

Additional tasks:

1. Implement chronological, maturity-aware training.
2. Tune models using permitted historical data.
3. Generate provider risk scores.
4. Explain important model features.

**Deliverables:**

```
supervised_model.pkl
anomaly_model.pkl
predictions.parquet
```

**Person 4**

Evaluation and presentation

**Responsibility:** Evaluate the models and demonstrate their business usefulness.

Tasks:

1. Implement highest total payment baseline.
2. Implement highest payment-per-beneficiary baseline.
3. Implement random-selection baseline.
4. Calculate Precision\@K, Recall\@K and Lift\@K.
5. Calculate bootstrap confidence intervals.
6. Compare supervised and anomaly rankings.
7. Compare top-K shortlist overlap and specialty composition.
8. Prepare charts, tables and dashboard.

**Deliverables:**

```
results.csv
top_k_shortlists.csv
model_comparison.csv
presentation_figures/
```

## **2. How the four members will work independently**

The important rule is that nobody should wait for another member's real dataset.

## **Workflow**

**P1**

LEIE labels

Independent

**P2**

CMS features

Independent

**P3**

Models using stub data

Independent

**P4**

Evaluation using stub data

Independent

**Integration**

P1 labels + P2 features

P3 predictions

P4 final evaluation

### **Freeze these file contracts**

| File                        | Primary key                    |
| --------------------------- | ------------------------------ |
| `labels.parquet`            | NPI, year                      |
| `provider_features.parquet` | NPI, year                      |
| `predictions.parquet`       | NPI, year, approach            |
| `results.csv`               | approach, K, evaluation period |

Use NPI as a string and year as an integer.

Everyone must create a small five-to-ten-provider stub dataset first. P3 and P4 must be able to run their code without waiting for P1 and P2.

## **3. Execution schedule**

November 21–25: Five-day execution plan

| Date       | Person 1                                                | Person 2                                   | Person 3                                     | Person 4                                           |
| ---------- | ------------------------------------------------------- | ------------------------------------------ | -------------------------------------------- | -------------------------------------------------- |
| **Nov 21** | Inspect LEIE and existing matching code                 | Audit CMS files                            | Build model pipelines on stubs               | Build evaluation on stubs                          |
| **Nov 22** | Finalise label feasibility and distinct positive counts | Finish CMS validation and initial features | Complete both model pipelines on stubs       | Complete baselines and metrics on stubs            |
| **Nov 23** | Publish real labels and candidate pool                  | Publish real provider features             | Integrate real data and generate predictions | Integrate predictions and test evaluation          |
| **Nov 24** | Validate labels and maturity                            | Validate features and peer groups          | Finalise models and scores                   | Complete model comparison and confidence intervals |
| **Nov 25** | Data-quality slides                                     | Feature-engineering slides                 | ML methodology slides                        | Results and visualisation slides                   |

**November 26–27:** Presentations.

**November 28–30:** Reproducibility checks, report and submission packaging.

**December 1:** Final submission.

## **4. Four rules the team must follow**

1. **No waiting:** Every member begins with stub data and builds their own pipeline independently.
2. **No changing contracts individually:** Any schema change must be agreed upon by everyone.
3. **No leakage:** For a prediction made at the end of year \\(T\\), use only training labels whose entire three-year outcome window matured by \\(T\\).
4. **No unsupported fraud claims:** The project identifies elevated audit risk and future exclusion associations, not proven fraudulent claims.

## **5. Daily coordination**

Hold one short team meeting each evening.

Each person reports only three things:

- What was completed?
- What is the expected deliverable?
- Is anything blocking the integration?

**The November 22 go/no-go checkpoint is particularly important.** Confirm that Scenario A has sufficient distinct positive providers, LEIE NPI coverage is adequate, and historical outcomes are observable.

If supervised labels are too weak, the team should still be able to complete the unsupervised anomaly-ranking approach.

## **My recommendation**

Keep this division fixed:

| Member       | Ownership                              |
| ------------ | -------------------------------------- |
| **You (P1)** | LEIE, labels and candidate eligibility |
| **P2**       | CMS cleaning and financial features    |
| **P3**       | Supervised and unsupervised ML         |
| **P4**       | Baselines, evaluation and presentation |

This gives each person a clear technical responsibility and an independent deliverable.

**Confidence: 99/100** on the work split and dependency structure. The modelling and evaluation scope will still depend on the November 22 label-feasibility results.