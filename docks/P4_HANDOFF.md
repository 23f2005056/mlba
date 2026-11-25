# P4 evaluation handoff

P4 reports Hits@K, Precision@K, Recall@K, Lift@K, Payment-Exposure Recall@K, historical payment exposure per shortlist place, and value-weighted lift for 2022 and 2023. The original workloads are K = 100, 500 and 1,000. The exploratory workloads are K = 2,500 and 5,000. The exposure metrics use P2's payments from the same NPI and decision year as the score. The results CSV includes 95% provider-level bootstrap intervals for the exposure measures, by method, year, and K. Actual dollars recovered per completed review, net savings and ROI belong in the audit pilot.

The slides compare K = 1,000, 2,500 and 5,000 and show the detailed metrics at 5,000. It is the only workload shown where all three models had hits in both years. LightGBM led in both years at that workload and is the lead pilot candidate if capacity allows for 5,000 reviews.

- [Evaluation protocol](P4_EVALUATION_PROTOCOL.md)
- [Report](../reports/p4_evaluation_summary.md)
- [Dashboard](../reports/p4_dashboard.html)
- [Consolidated metrics](../reports/p4_results.csv), one row per year, method, and K
- [Provider shortlists at the original workloads](../reports/p4_top_k_shortlists.csv)
- [Three-slide P4 PowerPoint](../presentation_figures/FraudLens_P4_Evaluation_ThreeSlides.pptx)

## Rebuild

Place the real P1 labels and candidate pool, P2 provider features, and P3 predictions in the repository's stubs folder. If P1 supplies eval_pool.parquet, place it in stubs or out for direct key verification. The large inputs are shared separately. Use Python 3.11 with pandas, NumPy, pyarrow and matplotlib.

```powershell
py -3.11 src/p4_prepare_cohort.py
py -3.11 src/p4_build_final_outputs.py
```

The full provider-level shortlist export, including K = 2,500 and 5,000, is generated locally at `out/p4/top_k_shortlists.csv`.

P4 uses P3's frozen scores for the 2022 and 2023 cohorts. LightGBM and Logistic Regression scores rank providers by exclusion risk. They are not calibrated probabilities because training down-sampled negatives and heavily weighted positives. Isolation Forest ranks unusual billing. P4 applies descending score and ascending NPI as the tie rule.
