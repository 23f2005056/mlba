Yes. I think we can now give FraudLens a much stronger identity: **a healthcare financial-risk and audit-prioritisation system**, rather than simply a machine-learning model predicting exclusions.

I found some particularly useful 2026 CMS and GAO figures, including real cases where billions of dollars in Medicare billing were questioned or stopped.

The strongest approach is to build your story around three things:

1. The money at risk.
2. How questionable billing actually happens.
3. How FraudLens could help an audit team decide where to investigate first.

# **1. Start with the actual business problem: Medicare's financial exposure**

Medicare is an enormous financial system.

A KFF analysis published in March 2025 reported that Medicare covered approximately 68 million people, involved more than 1.4 million providers and represented nearly $1 trillion in spending in 2024.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.kff.org\&sz=32)

KFF



More importantly, CMS's latest FY2026 figures, published November 16, 2026, show:

**The financial exposure**

# **$33.96B**

Original Medicare

Estimated improper payments, FY2026

# **7.32%**

Overall error rate

Original Medicare, FY2026

# **$12.45B**

Part B providers

Estimated improper payments

# **9.94%**

Part B error rate

FY2026

Sources: CMS FY2026 Improper Payments and CMS Comprehensive Error Rate Testing (CERT). The reporting sample covers claims submitted July 1, 2024 through June 30, 2025.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

CMS+2



The Part B figures are particularly relevant to your project.

However, there is an important distinction.

**Improper payments are not the same as fraud.**

CMS defines improper payments to include:

- Overpayments.
- Underpayments.
- Payments with insufficient supporting information.
- Administrative and documentation errors.

CMS explicitly states that its improper-payment rate does not measure fraud.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

CMS



So, when somebody asks:

**"What percentage of Medicare providers are committing fraud?"**

The scientifically honest answer is that these national figures do not establish such a percentage.

The 9.94% is an improper-payment estimate for Part B provider payments, not the percentage of doctors committing fraud.

### **Additional provider-type evidence**

CMS's FY2026 CERT report also provides estimates by provider type.

| Part B provider type                        | Improper payment rate | Projected improper payments |
| ------------------------------------------- | --------------------- | --------------------------- |
| Independently billing clinical laboratories | 28.8%                 | $1.92B                      |
| Internal medicine                           | 16.8%                 | $1.56B                      |
| Family practice                             | 12.6%                 | $675M                       |
| Nurse practitioners                         | 7.4%                  | $742M                       |
| Physical therapists in private practice     | 16.7%                 | $657M                       |

These are estimates based on sampled claims, with provider-type-specific confidence intervals. They are not fraud prevalence rates among practitioners.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

Centers for Medicare & Medicaid Services



**How to use this in your presentation:** Show the overall Medicare figure, then zoom into Part B. It establishes why provider-level financial risk deserves attention.

# **2. Understand how Medicare actually pays healthcare providers**

This is probably the most important business concept to explain.

You specifically asked:

- What is the allowed amount for a procedure?
- How much does a provider actually charge?
- How much does Medicare actually pay?
- When does a difference become suspicious?

Let's understand the reimbursement process.

## **2.1 Three different monetary amounts**

Imagine a doctor performs a procedure for a Medicare beneficiary.

There are three different amounts.

**1. Submitted charge**

The amount the provider bills for the service.

# **$300**

**2. Medicare allowed amount**

The amount recognised under the applicable Medicare reimbursement rules.

# **$120**

**3. Actual Medicare payment**

The amount Medicare pays after applicable deductible and coinsurance adjustments.

# **$96**

Illustrative example only. Assumes a participating provider, an applicable 80/20 Part B cost-sharing arrangement, and that the deductible has been satisfied.

In this hypothetical example:

- Provider submits: $300.
- Medicare recognises: $120.
- Medicare pays: $96.
- Beneficiary or supplementary coverage is responsible for the remaining $24.

The difference between $300 and $120 is not automatically Medicare overpayment.

The $300 is the provider's submitted charge; $120 is the allowed amount.

CMS defines the Medicare allowed amount as including Medicare's payment, the beneficiary's deductible and coinsurance, and any third-party payment responsibility. Actual Medicare payment is reported separately.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://data.cms.gov\&sz=32)

by Provider and Service Data Dictionary+1



## **2.2 Is there a fixed allowed amount for every procedure?**

No. There is no single universal limit for all procedures.

Medicare generally uses the **Physician Fee Schedule (PFS)** for physician services.

It considers:

- The procedure's HCPCS/CPT code.
- Physician work.
- Practice expenses.
- Professional liability costs.
- Geographic location.
- Facility versus non-facility setting.
- Applicable conversion factor and payment policies.

CMS uses Relative Value Units (RVUs), Geographic Practice Cost Indices (GPCIs) and a conversion factor to calculate payment rates.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

CMS+1



The 2026 PFS has two conversion factors: $33.57 for qualifying Alternative Payment Model participants and $33.40 for non-qualifying participants. These are inputs to the formula, not the allowed price of a procedure.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

CMS



For a specific procedure, you need the correct code, year, locality, place of service and other applicable adjustments.

There are also other Medicare payment systems for certain supplies, drugs and services, so not every Part B service can be priced using the Physician Fee Schedule.

CMS provides an official lookup for more than 10,000 services:

[CMS Physician Fee Schedule Lookup Tool](https://www.cms.gov/medicare/physician-fee-schedule/search/overview?utm_source=chatgpt.com)

An additional nuance: non-participating providers may be subject to a statutory limiting-charge rule. It is calculated using the non-participating payment basis and is not simply an unrestricted right to charge any amount.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://pfs.data.cms.gov\&sz=32)

pfs.data.cms.gov



## **2.3 When does billing become suspicious?**

Consider five examples:

| Billing behaviour                                       | Why it could matter                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------- |
| A sudden increase in services                           | Could indicate unusual utilisation or a change in business activity |
| Very high payments per beneficiary                      | Could justify comparing the provider against similar practitioners  |
| Excessive procedure frequency                           | Could indicate unnecessary services                                 |
| Billing for procedures not actually performed           | Potentially fraudulent claims                                       |
| Billing a higher-level service than the records support | Possible upcoding                                                   |

The important point is that these behaviours require investigation.

A high charge, high payment or unusually large number of procedures does not by itself establish fraud.

For your business story, this lets you move away from abstract model variables and talk about real healthcare financial activity.

# **3. Real-world cases: This problem is already happening**

I found three particularly useful examples.

## **Case 1 — Medicare urinary catheter billing: More than $4 billion**

Potential fraud / payments suspended

[LOT SpeediCath urinary catheter, model 28414, approx. (25) boxes @ 30/box of coated catheter,(loc](https://images.openai.com/static-rsc-4/Vsjk9jrS9ZkzAgqfIXBDe1yBsfthZvdAuQ_RHeCEJg6F1AGwG-scp-BOdXOwS1chxQqo0h4hix2Tay7u6n_QNC9vdtmZy4I4dq2qplTliNfV2TKYreu8LiTc0Mnhasqrpab2_4ncI9VaR3W44racbQ1Pjh0F5QzT-IIvhreJpQg?purpose=inline)

[bidspotter.com](https://www.bidspotter.com/en-us/auction-catalogues/rabin/catalogue-id-rabin10219/lot-2511065f-a393-4430-ab5e-b145004970fe?utm_source=chatgpt.com)

According to a 2026 GAO report, 15 healthcare providers allegedly attempted to bill Medicare for more than $4 billion worth of urinary catheters that were never supplied to beneficiaries.

CMS used billing analytics to identify the suspicious increase.

The agency suspended payments and reportedly prevented more than 99% of the payments from being issued to those suppliers.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.gao.gov\&sz=32)

U.S. GAO+1



**Why this matters to FraudLens:**

This example demonstrates the importance of:

- Billing spikes.
- Unusual service volumes.
- Unusual payment growth.
- Identifying providers that behave differently from their peers.

These are directly related to features you already use.

One qualification: this was a group of DME suppliers, not exactly the individual-practitioner population in your final model. Use it as evidence of the broader Medicare billing-risk problem.

## **Case 2 — Arizona wound-graft fraud: $1.2 billion in false claims**

Guilty pleas

[AmnioBand® Membrane | WoundSource](https://images.openai.com/static-rsc-4/PmC9x4VafMnhSMHS5VFRU22OdWuE4Dp8suAr_8au5NFNbkBMCRfi5kWKEQzqWVyDYhm0FOOIRnEgMh3-uMx2l9HfS2ICNnqRWMRqyAl5yzAepB6NiWeZjrJxQc7orTAPR-yuJRqTEei63f-Qsv0WJGuledEI3pUM6Zstx-OsER0?purpose=inline)

[woundsource.com](https://www.woundsource.com/product/amnioband-membrane?utm_source=chatgpt.com)

In January 2025, an Arizona couple pleaded guilty in connection with a scheme involving medically unnecessary wound grafts.

According to DOJ:

- More than $1.212 billion in false and fraudulent claims were submitted to healthcare insurers.
- More than $960 million related to federal healthcare programmes, including Medicare.
- Insurers paid approximately $614.99 million.
- Sales representatives were financially incentivised to order grafts.
- Grafts were applied even where the treatment was medically unnecessary.
- The scheme involved medically untrained representatives influencing treatment decisions.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://www.justice.gov\&sz=32)

  United States Department of Justice



This is particularly useful because it demonstrates a financial incentive behind unnecessary medical procedures.

**Why this matters to FraudLens:**

It shows why investigating the following features can be valuable:

- Payment per beneficiary.
- Submitted charges per beneficiary.
- Services per beneficiary.
- Year-on-year growth.
- Unusual billing relative to comparable practitioners.

The case also shows why high medical expenditure does not automatically mean the treatment was appropriate.

A claim requires medical necessity and appropriate documentation.

## **Case 3 — CMS's 2025 Fraud Defense Operations Center**

Official CMS programme results

CMS launched the Fraud Defense Operations Center (FDOC) in March 2025.

Its published 2025 results included:

| Measure                           | Result                 |
| --------------------------------- | ---------------------- |
| Providers investigated            | 347                    |
| Providers with payments suspended | 249                    |
| Payments suspended                | More than $1.8 billion |

CMS reported suspicious billing involving durable medical equipment, skin substitutes and laboratories.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

Centers for Medicare & Medicaid Services



The distinction matters: these were suspended payments associated with suspected fraud, not necessarily confirmed losses or recovered money.

**Why this matters to your business story:**

This is real evidence that Medicare already uses dedicated teams combining:

- Data analysts.
- Investigators.
- Healthcare specialists.
- Legal professionals.
- Law enforcement.

Your project is addressing a real operational requirement.

# **4. The most important discovery: Medicare already uses analytics to prioritise investigations**

This is where your project becomes particularly relevant.

GAO published a report in March 2026 titled:

**Medicare: CMS's Use of Data Analytics to Identify and Prevent Fraud.**

[Read the GAO report](https://www.gao.gov/products/gao-26-107799?utm_source=chatgpt.com)

It explains that CMS and its contractors use analytics to identify:

- Sudden billing spikes.
- Unusual provider billing patterns.
- Differences from comparable providers.
- Providers associated with potential fraud.
- Emerging fraud schemes.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://files.gao.gov\&sz=32)

  GAO Files+1



More importantly, GAO reported that manual prepayment claim reviews generally covered less than 1% of claims. Such reviews require medical records, trained clinicians and coding specialists.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://files.gao.gov\&sz=32)

GAO Files



This is the operational problem.

**The audit-capacity problem**

# **Millions of claims**

Too many claims and providers for intensive manual examination of every case.

Manual prepayment reviews

**Generally under 1% of claims**

**Business requirement**

Prioritise the providers and billing patterns most deserving of closer review.

The GAO report also found that CMS estimated $11.9 billion in potentially fraudulent Medicare payments prevented through administrative actions during FY2022–FY2024.

This figure includes projected cost avoidance and other estimates, not just cash recovered.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://files.gao.gov\&sz=32)

GAO Files



**This is the strongest external validation of your business problem.**

However, CMS already has its own Fraud Prevention System and analytics. You should position FraudLens as a prototype exploring the use of public provider-level data for complementary audit prioritisation—not as a replacement for CMS's existing infrastructure.

# **5. Connect your actual model features to the business problem**

This is perhaps where your presentation can become much more impressive.

Do not just list the model's features. Explain what business behaviour they represent.

| Your feature                | Business meaning                                           | Potential audit question                                     |
| --------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| `medicare_payment_yoy_diff` | Change in Medicare payments from the previous year         | Why did Medicare payments increase sharply?                  |
| `pymt_per_bene`             | Medicare payment per beneficiary                           | Is expenditure per beneficiary unusually high?               |
| `srvcs_per_bene`            | Services per beneficiary                                   | Is the provider delivering unusually high service volumes?   |
| `charge_to_allowed`         | Relationship between submitted charges and allowed amounts | Does the provider have an unusual charge pattern?            |
| `pymt_per_bene_peer_z`      | Payment intensity relative to similar practitioners        | Is this provider an outlier within its specialty?            |
| `srvcs_per_bene_peer_z`     | Service intensity relative to peers                        | Is utilisation unusually high compared with similar doctors? |
| `pymt_per_bene_yoy_growth`  | Growth in payment per beneficiary                          | Is spending per patient rising unusually quickly?            |

## **Your LightGBM feature importance makes this story stronger**

**What did the model learn?**

# **40.37%**

**Medicare payment year-on-year difference**

Share of LightGBM total gain attributed to this feature.

It was the strongest feature in the model.

**Other important signals**

| Feature                              | Share of gain |
| ------------------------------------ | ------------- |
| Charge-to-allowed ratio              | 11.52%        |
| Payment per beneficiary peer Z-score | 11.50%        |
| Payment per beneficiary              | 5.49%         |

This is especially interesting because GAO independently describes billing-spike detection and peer-to-peer analysis as methods already used in Medicare program-integrity analytics.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://files.gao.gov\&sz=32)

GAO Files



You can make the connection:

> The most important feature in our LightGBM model was the year-on-year change in Medicare payment. This suggests that changes in payment behaviour are useful signals for identifying providers who may warrant further investigation.

Don't say the model proves a payment spike is fraud.

# **6. What is your project's actual business value?**

Now we translate the model's metrics into the number of providers an audit team might have to review.

Your primary 2022 evaluation population contained 1,165,640 provider-year records and 82 future-positive Scenario A cases.

## **6.1 What happens when an audit team reviews the top 1%?**

| Measure                          | LightGBM | Logistic Regression |
| -------------------------------- | -------- | ------------------- |
| Providers selected               | 11,657   | 11,657              |
| Future-positive cases identified | 19       | 12                  |
| Recall\@1%                       | 23.2%    | 14.6%               |

## **6.2 What happens when an audit team reviews the top 5%?**

| Measure                          | LightGBM | Logistic Regression |
| -------------------------------- | -------- | ------------------- |
| Providers selected               | 58,282   | 58,282              |
| Future-positive cases identified | 33       | 35                  |
| Recall\@5%                       | 40.2%    | 42.7%               |

These are approximate provider counts derived from your rounded, reconciled metrics. Positive cases mean providers later recorded with the selected Scenario A OIG exclusion outcome. They are not confirmed fraud cases.

This gives you a practical way to explain the model.

For example:

> If an audit team selected the highest-ranked 5% of providers in the 2022 evaluation population, Logistic Regression identified approximately 35 of the 82 providers who subsequently received a Scenario A exclusion. LightGBM identified approximately 33.

But there is an important limitation.

The positive base rate is only 0.00703%.

Consequently, even the 35 positive cases within approximately 58,282 selected providers represent a very small share of the review list.

This is why FraudLens should be positioned as an initial screening and prioritisation tool—not a final fraud detection or guilt determination system.

Also, the top-1% and top-5% thresholds are evaluation scenarios, not a confirmed CMS audit quota.

## **6.3 Where is the actual financial benefit?**

Your current experiment does not directly estimate dollars saved.

That is because it uses:

- Provider-level annual billing aggregates.
- Future OIG exclusion outcomes.
- No confirmed claim-by-claim overpayment amounts.
- No actual audit cost or recovery data.

A credible future business-value calculation would be:

\\[ \text{Net Audit Value} = \text{Verified Recoveries} + \text{Avoided Improper Payments} - \text{Audit and Operating Costs} \\]

To calculate that, you would need to establish:

1. How many flagged providers are confirmed to have improper payments after investigation.
2. Average recoverable or preventable payment per confirmed case.
3. Average cost of conducting an investigation.
4. What proportion of identified cases would otherwise have gone undetected.

Do not multiply the model's recall by the $12.45 billion Part B improper-payment estimate and call the result potential savings. That would be unsupported.

# **7. One very important improvement: Add procedure-level financial analysis**

I want to highlight an opportunity that could make a future version of FraudLens much more business-oriented.

Your current feature dataset comes from the CMS **by Provider** data.

It contains aggregated provider-level amounts:

- `Tot_Sbmtd_Chrg`
- `Tot_Mdcr_Alowd_Amt`
- `Tot_Mdcr_Pymt_Amt`
- `Tot_Srvcs`
- `Tot_Benes`

CMS defines these as provider-level totals across services.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://data.cms.gov\&sz=32)

data.cms.gov



This means you currently cannot say:

"Dr. X billed $500 for procedure Y when Medicare's allowed rate was $150."

Your present data is not sufficiently granular to support that claim.

### **How can we get to procedure-level billing?**

CMS also publishes a dataset called:

**Medicare Physician & Other Practitioners — by Provider and Service.**

[CMS Physician and Other Practitioners Data](https://data.cms.gov/provider-data/topics/medicare/medicare-physician-other-practitioners?utm_source=chatgpt.com)

Unlike the provider-level file, it provides information organised by:

- NPI.
- HCPCS/CPT code.
- Place of service.

It includes average submitted charges, Medicare allowed amounts, payments and service counts.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://catalog.data.gov\&sz=32)

by Provider and Service+3



You could connect this dataset to the official Medicare Physician Fee Schedule.

**Potential Phase 2: Procedure-level billing surveillance**

**Provider and Service Dataset**

NPI + HCPCS code + place of service

**Medicare Physician Fee Schedule**

Code-specific payment rates, locality and facility adjustments

**Procedure-level financial indicators**

Charge-to-schedule ratio, services per beneficiary, payment deviations and peer comparisons

**Potential audit signals**

Providers requiring examination of claims, medical records and medical necessity

This would enable you to ask much more specific business questions:

- Which procedures account for unusually high expenditure?
- Which practitioners charge substantially above the typical peer distribution?
- Are providers delivering an unusually high number of specific services per beneficiary?
- Are payments increasing because of procedure volume or procedure mix?
- Are unusual charge patterns associated with subsequent enforcement outcomes?

The important caveat is that the comparison must use the correct code, year, locality, place of service and payment rules. A high submitted charge compared with a fee schedule amount is not itself proof of a prohibited charge or fraud.

**My recommendation:** Keep this as a proposed extension, not something to claim your current model already does.

# **8. The actual business story I recommend for your presentation**

I would arrange the narrative into six slides, with a stronger business introduction.

| Slide | Title                                                | What it establishes                                      |
| ----- | ---------------------------------------------------- | -------------------------------------------------------- |
| 1     | The Financial Risk in Medicare                       | Medicare scale, improper payments and Part B exposure    |
| 2     | How Provider Billing Works                           | Submitted charges, allowed amounts and Medicare payments |
| 3     | Real-World Billing Abuse                             | Urinary catheter and wound-graft cases                   |
| 4     | The Audit Prioritisation Challenge                   | Limited manual review and the role of risk analytics     |
| 5     | FraudLens: From Billing Signals to Audit Priorities  | Your data, features, labels and model results            |
| 6     | Business Value, Limitations and Pilot Recommendation | What the system could support operationally              |

## **Suggested opening for your presentation**

You can use this almost word for word:

> Medicare processes an enormous volume of healthcare payments every year. But the challenge is not simply how much money Medicare spends. It is how to ensure that the money is being paid for appropriate, necessary and properly documented healthcare services.
>
> According to the latest CMS estimates, Original Medicare had approximately $33.96 billion in improper payments in FY2026. Within Part B providers, the estimated improper payment amount was $12.45 billion.
>
> Actual healthcare billing investigations demonstrate that these risks are not theoretical. There have been cases involving medically unnecessary wound grafts, claims for supplies never delivered and unusual billing spikes.
>
> However, investigating every provider and every claim manually is not practical. GAO reports that less than 1% of claims are generally subject to manual prepayment review.
>
> This creates a business need for better prioritisation: identifying which providers deserve closer examination first.
>
> Our project, FraudLens, explores whether historical Medicare billing behaviour can help rank individual practitioners according to their likelihood of subsequently receiving selected OIG exclusions.

## **Suggested closing**

> Our objective is not to replace healthcare investigators or automatically identify guilty providers.
>
> We want to help an audit team use historical billing information more effectively, identify unusual financial patterns and direct its limited investigative resources towards providers who warrant closer review.
>
> Our evaluation shows that supervised models can substantially improve the identification of future OIG-excluded providers compared with simple billing-based rankings.
>
> The next step is to pilot the models in a human-reviewed audit workflow and, with more granular claims data, extend the system towards procedure-level financial surveillance.

# **9. The five external facts I would definitely put in your presentation**

# **$33.96B**

**Original Medicare improper payments, FY2026**

CMS official estimate. Not all fraud.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

CMS



# **$12.45B**

**Part B provider improper payments, FY2026**

9.94% estimated improper-payment rate.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.cms.gov\&sz=32)

CMS



# **$4B+**

**Alleged urinary catheter billing scheme**

15 providers; CMS reported stopping over 99% of the payments.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.gao.gov\&sz=32)

U.S. GAO



# **$1.2B**

**Wound-graft false-claims case**

Arizona defendants pleaded guilty; approximately $615M was paid by insurers.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.justice.gov\&sz=32)

United States Department of Justice



# **<1%**

**Claims generally subject to manual prepayment review**

GAO's description of CMS's manual review process.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://files.gao.gov\&sz=32)

GAO Files



# **10. Sources worth retaining for your research**

These are the most useful sources I found. I would retain the first six as your primary references.

| Source                                                                                                                                                                                        | What to use it for                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [CMS — FY2026 Improper Payments](https://www.cms.gov/newsroom/fact-sheets/fiscal-year-2026-improper-payments?utm_source=chatgpt.com)                                                          | Latest national and programme-level improper-payment figures  |
| [CMS — CERT Improper Payment Rates](https://www.cms.gov/data-research/monitoring-programs/improper-payment-measurement-programs/comprehensive-error-rate-testing-cert?utm_source=chatgpt.com) | Part B provider payment-error estimates                       |
| [CMS — Physician Fee Schedule Lookup](https://www.cms.gov/medicare/physician-fee-schedule/search/overview?utm_source=chatgpt.com)                                                             | Procedure reimbursement amounts and pricing rules             |
| [GAO — CMS's Use of Data Analytics to Identify and Prevent Fraud](https://www.gao.gov/products/gao-26-107799?utm_source=chatgpt.com)                                                          | Actual analytics, audits, billing spikes and financial impact |
| [DOJ — Arizona $1.2B Wound-Graft Fraud Case](https://www.justice.gov/opa/pr/arizona-couple-pleads-guilty-12b-health-care-fraud?utm_source=chatgpt.com)                                        | Real-world unnecessary-service and reimbursement scheme       |
| [DOJ — 2026 National Health Care Fraud Cases](https://www.justice.gov/fraud/2026-national-hcf-case-summaries?utm_source=chatgpt.com)                                                          | More recent enforcement examples                              |
| [KFF — Medicare Program Integrity](https://www.kff.org/medicare/medicare-program-integrity-and-efforts-to-root-out-improper-payments-fraud-waste-and-abuse/?utm_source=chatgpt.com)           | Healthcare expenditure and policy context                     |
| [Becker's — Arizona $1.2B Healthcare Fraud Case](https://www.beckersasc.com/asc-news/arizona-couple-pleads-guilty-to-1-2b-healthcare-fraud/?utm_source=chatgpt.com)                           | Industry news coverage of the wound-graft case                |

## **My final recommendation**

The strongest positioning for your project is:

Recommended business positioning

# **FraudLens: Medicare Provider Financial Risk and Audit Prioritisation**

A provider-level analytical system that examines historical Medicare Part B billing patterns, identifies unusual financial and utilisation behaviour, and ranks individual practitioners for further audit investigation using subsequent OIG exclusion outcomes as an imperfect risk proxy.

There are three things I particularly want you to remember when presenting:

1. **Do not say 9.94% of doctors commit fraud.** It is a Part B improper-payment rate, not provider-level fraud prevalence.
2. **Do not say your current model detects procedure-level overbilling.** It operates on annual provider-level aggregates. Procedure-level analysis would require additional CMS data and fee-schedule matching.
3. **Do not claim that your 40% recall translates directly into 40% of fraud or billions saved.** It means the model identifies a proportion of providers who subsequently received your selected OIG exclusion outcome.

The strongest business argument is that Medicare has a real financial-integrity problem, limited manual-review capacity, and an established need for billing-pattern analytics. FraudLens demonstrates how public provider-level information can contribute to that prioritisation process.

That is a much more convincing story than simply saying you trained three machine-learning models.