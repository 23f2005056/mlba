# FraudLens P4 evaluation

P4 evaluated the saved P3 rankings for 2022 and 2023. The original workloads were K = 100, 500 and 1,000. After examining those results, P4 explored K = 2,500 and 5,000. A hit is a shortlisted individual practitioner who met the two-year Scenario A exclusion outcome.

LightGBM and Logistic Regression scores rank exclusion risk. They are not calibrated probabilities because training down-sampled negatives and heavily weighted positives. Isolation Forest ranks unusual billing. P4 sorts scores descending and breaks ties by ascending NPI.

## Metrics

| Metric | Calculation |
|---|---|
| Hits@K | Later exclusions in the shortlist |
| Precision@K | Hits divided by K |
| Recall@K | Hits divided by later exclusions in the year's full pool |
| Lift@K | Precision divided by the year's pool exclusion rate |
| Payment-Exposure Recall@K | Captured payments divided by all decision-year Medicare payments to later-excluded providers in the year's pool |
| Historical payment exposure per shortlist place | Captured payments divided by K |
| Value-weighted lift@K | Captured payments divided by the exposure expected from K random selections in the same year |
| Net savings and ROI | Pilot measures based on measured financial benefit, review costs, and measurable friction costs |

Captured payments are Medicare payments to shortlisted providers who were later excluded. P4 verified each payment against P2's `medicare_payment` for the same NPI and decision year as the score.

The 95% bootstrap intervals use 20,000 resamples of later-excluded NPIs within each year. Each draw is shared across methods, holds the shortlists fixed, and recalculates captured and pool payments. The intervals reflect sensitivity to which later-excluded providers appear in the resample. Zero-hit shortlists yield zero-width intervals under this design. The analytical random expectation has no interval.

## Results by year and workload

The table gives the three model rankings. The consolidated CSV also includes the four billing rules and the random expectation.

| Year | K | Pool reviewed | Model | Hits | Precision | Recall | Lift | Payment exposure | Payment-Exposure Recall@K [95% CI] | Exposure per place [95% CI] | Value-weighted lift [95% CI] |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 100 | 0.01% | Isolation Forest | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2022 | 100 | 0.01% | LightGBM | 1 | 1.00% | 1.22% | 142.15× | $70.92 | 0.00059% [0.00000%–0.00230%] | $0.71 [$0.00–$2.13] | 0.07× [0.00–0.27×] |
| 2022 | 100 | 0.01% | Logistic Regression | 1 | 1.00% | 1.22% | 142.15× | $70.92 | 0.00059% [0.00000%–0.00230%] | $0.71 [$0.00–$2.13] | 0.07× [0.00–0.27×] |
| 2022 | 500 | 0.04% | Isolation Forest | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2022 | 500 | 0.04% | LightGBM | 3 | 0.60% | 3.66% | 85.29× | $1,372.13 | 0.01146% [0.00000%–0.03585%] | $2.74 [$0.00–$7.09] | 0.27× [0.00–0.84×] |
| 2022 | 500 | 0.04% | Logistic Regression | 2 | 0.40% | 2.44% | 56.86× | $498.15 | 0.00416% [0.00000%–0.01466%] | $1.00 [$0.00–$2.85] | 0.10× [0.00–0.34×] |
| 2022 | 1,000 | 0.09% | Isolation Forest | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2022 | 1,000 | 0.09% | LightGBM | 3 | 0.30% | 3.66% | 42.65× | $1,372.13 | 0.01146% [0.00000%–0.03585%] | $1.37 [$0.00–$3.55] | 0.13× [0.00–0.42×] |
| 2022 | 1,000 | 0.09% | Logistic Regression | 3 | 0.30% | 3.66% | 42.65× | $873,951.40 | 7.29766% [0.00000%–23.42393%] | $873.95 [$0.00–$2,621.29] | 85.06× [0.00–273.04×] |
| 2022 | 2,500 | 0.21% | Isolation Forest | 1 | 0.04% | 1.22% | 5.69× | $70.92 | 0.00059% [0.00000%–0.00230%] | $0.03 [$0.00–$0.09] | 0.00× [0.00–0.01×] |
| 2022 | 2,500 | 0.21% | LightGBM | 5 | 0.20% | 6.10% | 28.43× | $80,685.02 | 0.67373% [0.00381%–2.30819%] | $32.27 [$0.20–$90.39] | 3.14× [0.02–10.76×] |
| 2022 | 2,500 | 0.21% | Logistic Regression | 4 | 0.16% | 4.88% | 22.74× | $2,022,435.40 | 16.88771% [0.00066%–39.65671%] | $808.97 [$0.03–$2,077.20] | 78.74× [0.00–184.90×] |
| 2022 | 5,000 | 0.43% | Isolation Forest | 2 | 0.04% | 2.44% | 5.69× | $498.15 | 0.00416% [0.00000%–0.01466%] | $0.10 [$0.00–$0.28] | 0.01× [0.00–0.03×] |
| 2022 | 5,000 | 0.43% | LightGBM | 12 | 0.24% | 14.63% | 34.12× | $1,709,281.66 | 14.27282% [2.44199%–32.30023%] | $341.86 [$55.93–$775.76] | 33.27× [5.69–75.30×] |
| 2022 | 5,000 | 0.43% | Logistic Regression | 7 | 0.14% | 8.54% | 19.90× | $2,182,554.21 | 18.22473% [0.70332%–40.84359%] | $436.51 [$14.73–$1,080.39] | 42.49× [1.64–95.22×] |
| 2023 | 100 | 0.01% | Isolation Forest | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 100 | 0.01% | LightGBM | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 100 | 0.01% | Logistic Regression | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 500 | 0.04% | Isolation Forest | 1 | 0.20% | 1.72% | 41.26× | $2,305,318.50 | 29.39331% [0.00000%–61.73731%] | $4,610.64 [$0.00–$13,831.91] | 703.40× [0.00–1,477.40×] |
| 2023 | 500 | 0.04% | LightGBM | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 500 | 0.04% | Logistic Regression | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 1,000 | 0.08% | Isolation Forest | 1 | 0.10% | 1.72% | 20.63× | $2,305,318.50 | 29.39331% [0.00000%–61.73731%] | $2,305.32 [$0.00–$6,915.96] | 351.70× [0.00–738.70×] |
| 2023 | 1,000 | 0.08% | LightGBM | 1 | 0.10% | 1.72% | 20.63× | $8,260.47 | 0.10532% [0.00000%–0.45866%] | $8.26 [$0.00–$24.78] | 1.26× [0.00–5.49×] |
| 2023 | 1,000 | 0.08% | Logistic Regression | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 2,500 | 0.21% | Isolation Forest | 2 | 0.08% | 3.45% | 16.50× | $2,305,482.78 | 29.39541% [0.00000%–61.74021%] | $922.19 [$0.00–$2,766.51] | 140.69× [0.00–295.49×] |
| 2023 | 2,500 | 0.21% | LightGBM | 2 | 0.08% | 3.45% | 16.50× | $14,243.98 | 0.18161% [0.00000%–0.63523%] | $5.70 [$0.00–$14.70] | 0.87× [0.00–3.04×] |
| 2023 | 2,500 | 0.21% | Logistic Regression | 0 | 0.00% | 0.00% | 0.00× | $0.00 | 0.00000% [0.00000%–0.00000%] | $0.00 [$0.00–$0.00] | 0.00× [0.00–0.00×] |
| 2023 | 5,000 | 0.42% | Isolation Forest | 3 | 0.06% | 5.17% | 12.38× | $2,925,921.28 | 37.30613% [0.00000%–67.00852%] | $585.18 [$0.00–$1,631.40] | 89.28× [0.00–160.35×] |
| 2023 | 5,000 | 0.42% | LightGBM | 5 | 0.10% | 8.62% | 20.63× | $85,126.04 | 1.08538% [0.12482%–3.34920%] | $17.03 [$2.39–$37.85] | 2.60× [0.30–8.01×] |
| 2023 | 5,000 | 0.42% | Logistic Regression | 3 | 0.06% | 5.17% | 12.38× | $57,582.57 | 0.73419% [0.00000%–2.95298%] | $11.52 [$0.00–$33.29] | 1.76× [0.00–7.07×] |

## Pilot decision

The presentation compares K = 1,000, 2,500 and 5,000. P4 chose 5,000 for the detailed exploratory comparison after seeing the test results. Of these three workloads, 5,000 was the only one where all three models had at least one hit in both years. At K = 2,500, Logistic Regression still had zero hits in 2023. LightGBM had the most observed hits at K = 5,000 in both years. Increasing K from 2,500 to 5,000 added 7 LightGBM hits in 2022 (5 to 12) and 3 in 2023 (2 to 5).

Pilot LightGBM and Logistic Regression at the same K, set by the audit team's available review capacity. Compare both against the current billing rule and use the pilot results to decide which model to deploy. Payment exposure can be concentrated in a few providers. Track actual recoveries and measured review costs.

Net benefit is measured financial benefit minus review and measurable friction costs. ROI is net benefit divided by those costs. The historical test data supply billing exposure and later exclusion outcomes.

## Files

- [Consolidated metrics](p4_results.csv)
- [Dashboard](p4_dashboard.html)
- [Top-K provider lists for the original workloads](p4_top_k_shortlists.csv). The rebuild writes the full provider-level export, including both exploratory workloads, to the local `out/p4/` folder.
- [Three-slide P4 presentation](../presentation_figures/FraudLens_P4_Evaluation_ThreeSlides.pptx)
