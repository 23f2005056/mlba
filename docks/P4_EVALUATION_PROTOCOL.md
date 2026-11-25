# FraudLens P4 evaluation protocol

## Test design

- Compare LightGBM, Logistic Regression, Isolation Forest, two raw payment rules, two peer-adjusted billing rules, and random selection.
- Report 2022 and 2023 separately. The original review workloads are K = 100, 500 and 1,000. Exploratory workloads K = 2,500 and 5,000 were examined after the original results. 2020 was used for model validation.
- Use P1's eligible individual evaluation pool, reconstructed by joining eligible provider-year labels with P2's feature keys. Require an observed two-year outcome and all three P3 scores for every provider in the pool.
- A hit means a shortlisted provider met the two-year Scenario A LEIE exclusion outcome, recorded as y_A_h2. P4 uses that historical outcome to evaluate audit prioritisation.
- Use P3's existing scores. LightGBM and Logistic Regression scores rank exclusion risk. They are not calibrated probabilities because training down-sampled negatives and heavily weighted positives. Isolation Forest ranks unusual billing.
- Rank scores descending and break ties by ascending ten-digit NPI. Rank the billing rules by total Medicare payment, payment per beneficiary, and P1's peer-adjusted payment and services per beneficiary. Missing peer scores receive the observed minimum.

## Approved measures

| Measure | Calculation |
|---|---|
| Share of pool reviewed | K divided by the number of eligible individual practitioners in that year's evaluation pool |
| Hits@K | Number of later-excluded providers in the shortlist |
| Precision@K | Hits divided by K |
| Recall@K | Hits divided by the number of later-excluded providers in the year's pool |
| Lift@K | Precision divided by the year's pool exclusion rate |
| Payment-Exposure Recall@K | Decision-year Medicare payments to captured later-excluded providers divided by payments to all later-excluded providers in the same year's pool |
| Historical payment exposure per shortlist place | Captured historical payments divided by K |
| Value-weighted lift@K | Captured historical payments divided by the expected captured payments under random selection of K providers from the same year |

P4 verifies `medicare_payment` against P2's features on NPI and decision year, the same year as the score. For each year, 95% exposure intervals use 20,000 provider-level resamples of the later-excluded NPIs, shared across ranking methods. Each draw recalculates captured and pool payments while holding shortlist membership fixed. A few providers can drive exposure estimates at small K. If total payment exposure is zero, payment-exposure recall and value-weighted lift are undefined.

For random selection, each provider has probability K / N of being selected, where N is the pool size. Expected hits and captured payments are K / N times their respective pool totals. The analytical random expectation has no bootstrap interval.

## Presentation and pilot workload

The three-slide deck compares K = 1,000, 2,500 and 5,000. K = 5,000 is the focus because every model had a hit in both years at that workload, the only one of the three shown where this occurred. At that workload LightGBM led on hits in both test years. The pilot K follows the audit team's available review capacity. Validate the exploratory finding prospectively at the chosen K.

## Audit pilot measures

Measure dollars recovered per completed review, net savings and ROI using audit outcomes. Estimate prevented losses against a credible comparison with the current audit process. Net benefit subtracts review costs and measurable friction costs from measured financial benefit. ROI divides net benefit by those costs. The historical test uses recorded Medicare payments as exposure.
