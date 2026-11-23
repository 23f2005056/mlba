"""Supplementary two-year model comparison clustered by provider NPI.

The primary 2022 and 2023 results remain separate. This report counts each
later-excluded provider once across the two years and resamples distinct NPIs,
preserving repeated provider outcomes in one cluster.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
MODELS = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
YEARS = (2022, 2023)
KS = (100, 500, 1000)
REPLICATES = 20_000
SEED = 20261123


def main() -> None:
    positive_by_year = {}
    rows = []
    for year in YEARS:
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet", columns=["npi", "year", "y_A_h2"])
        cohort["npi"] = cohort["npi"].astype("string")
        if not cohort["year"].eq(year).all() or cohort["npi"].duplicated().any():
            raise ValueError(f"Invalid cohort keys for {year}")
        positive_by_year[year] = set(cohort.loc[cohort["y_A_h2"].eq(1), "npi"])
        rows.append({"period": str(year), "positive_provider_year_records": len(positive_by_year[year]),
                     "distinct_positive_npis": len(positive_by_year[year])})
    positive_npis = sorted(set.union(*positive_by_year.values()))
    if sum(len(s) for s in positive_by_year.values()) != 140 or len(positive_npis) != 113:
        raise ValueError("Positive-provider counts differ from P1's confirmed 140 records and 113 NPIs")
    rows.append({"period": "2022+2023", "positive_provider_year_records": 140,
                 "distinct_positive_npis": 113})
    pd.DataFrame(rows).to_csv(REPORTS / "p4_distinct_positive_summary.csv", index=False)

    shortlists = {
        model: pd.read_csv(DATA / f"topk_{model}_provisional.csv", dtype={"npi": "string"})
        for model in MODELS
    }
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(positive_npis), size=(REPLICATES, len(positive_npis)), dtype=np.int16)
    comparisons = []
    for k in KS:
        captured = {}
        for model in MODELS:
            hits = set()
            for year in YEARS:
                selected = shortlists[model].loc[
                    shortlists[model]["year"].eq(year) & shortlists[model]["K"].eq(k), "npi"
                ]
                if len(selected) != k or selected.duplicated().any():
                    raise ValueError(f"Invalid {model} shortlist at {year} K={k}")
                hits |= set(selected) & positive_by_year[year]
            captured[model] = hits
        for a, b in combinations(MODELS, 2):
            indicator_a = np.isin(positive_npis, list(captured[a])).astype(np.int8)
            indicator_b = np.isin(positive_npis, list(captured[b])).astype(np.int8)
            only_a = int(((indicator_a == 1) & (indicator_b == 0)).sum())
            only_b = int(((indicator_a == 0) & (indicator_b == 1)).sum())
            differences = (indicator_a - indicator_b)[indices].sum(axis=1)
            low, high = np.quantile(differences, [0.025, 0.975])
            discordant = only_a + only_b
            comparisons.append({
                "period": "2022+2023", "K_each_year": k,
                "model_a": a, "model_b": b,
                "positive_provider_year_records": 140,
                "distinct_positive_npis": 113,
                "distinct_hits_a": len(captured[a]),
                "distinct_hits_b": len(captured[b]),
                "shared_distinct_hits": len(captured[a] & captured[b]),
                "distinct_hits_only_a": only_a,
                "distinct_hits_only_b": only_b,
                "difference_distinct_hits_a_minus_b": only_a - only_b,
                "provider_cluster_bootstrap_ci95_low": low,
                "provider_cluster_bootstrap_ci95_high": high,
                "exact_paired_p_value": float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0,
                "bootstrap_replicates": REPLICATES,
                "random_seed": SEED,
            })
    report = pd.DataFrame(comparisons).sort_values(["K_each_year", "model_a", "model_b"])
    report.to_csv(REPORTS / "p4_pooled_npi_uncertainty.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(report[["K_each_year", "model_a", "model_b", "distinct_hits_a", "distinct_hits_b",
                  "difference_distinct_hits_a_minus_b", "provider_cluster_bootstrap_ci95_low",
                  "provider_cluster_bootstrap_ci95_high"]].to_string(index=False))
    print(f"Saved {REPORTS / 'p4_pooled_npi_uncertainty.csv'}")


if __name__ == "__main__":
    main()
