"""Paired uncertainty for model hit differences on held-out years.

Resample the same outcome-positive provider IDs for both fixed shortlists,
keeping the observed positive count fixed. This assesses uncertainty in
relative capture, not model refitting, future-year drift, or audit value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "p4_model_paired_uncertainty.csv"
YEARS = (2022, 2023)
KS = (100, 500, 1000)
REPLICATES = 20_000
SEED = 20261123


def main() -> None:
    pairs = pd.read_csv(REPORTS / "p4_model_pairwise_overlap.csv")
    if len(pairs) != 18 or pairs.duplicated(["year", "K", "model_a", "model_b"]).any():
        raise ValueError("Expected 18 unique year/K/model-pair results")
    models = sorted(set(pairs["model_a"]) | set(pairs["model_b"]))
    if set(models) != {"supervised_lgbm", "supervised_logreg", "anomaly_iforest"}:
        raise ValueError("Unexpected model")
    lists = {
        model: pd.read_csv(DATA / f"topk_{model}_provisional.csv", dtype={"npi": "string"})
        for model in models
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for year in YEARS:
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet", columns=["npi", "year", "y_A_h2"])
        cohort["npi"] = cohort["npi"].astype("string")
        if not cohort["year"].eq(year).all() or cohort["npi"].duplicated().any():
            raise ValueError(f"Invalid cohort for {year}")
        if not cohort["y_A_h2"].isin((0, 1)).all():
            raise ValueError(f"Invalid outcomes for {year}")
        positives = cohort.loc[cohort["y_A_h2"].eq(1), "npi"].to_numpy()
        p = len(positives)
        if p == 0:
            raise ValueError(f"No positives in {year}")
        # One index sample per year is reused for all K and model pairs, so
        # every comparison is paired on the same resampled providers.
        indices = rng.integers(0, p, size=(REPLICATES, p), dtype=np.int16)
        for k in KS:
            memberships = {}
            for model in models:
                selection = lists[model].loc[
                    lists[model]["year"].eq(year) & lists[model]["K"].eq(k), "npi"
                ]
                if len(selection) != k or selection.duplicated().any():
                    raise ValueError(f"Invalid {model} shortlist for {year} K={k}")
                memberships[model] = np.isin(positives, selection.to_numpy()).astype(np.int8)
            for pair in pairs.loc[pairs["year"].eq(year) & pairs["K"].eq(k)].itertuples(index=False):
                a, b = memberships[pair.model_a], memberships[pair.model_b]
                hits_a, hits_b = int(a.sum()), int(b.sum())
                only_a = int(((a == 1) & (b == 0)).sum())
                only_b = int(((a == 0) & (b == 1)).sum())
                if (hits_a, hits_b, only_a, only_b) != (
                    int(pair.hits_a), int(pair.hits_b), int(pair.hits_only_a), int(pair.hits_only_b)
                ):
                    raise ValueError(f"Pairwise report disagrees for {year} K={k}")
                differences = (a - b)[indices].sum(axis=1)
                low, high = np.quantile(differences, [0.025, 0.975])
                discordant = only_a + only_b
                exact_p = float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
                rows.append({
                    "year": year, "K": k, "model_a": pair.model_a, "model_b": pair.model_b,
                    "positive_count": p,
                    "hits_a": hits_a, "hits_b": hits_b,
                    "hits_only_a": only_a, "hits_only_b": only_b,
                    "hit_difference_a_minus_b": hits_a - hits_b,
                    "hit_difference_bootstrap_ci95_low": low,
                    "hit_difference_bootstrap_ci95_high": high,
                    "precision_difference_a_minus_b": (hits_a - hits_b) / k,
                    "precision_difference_bootstrap_ci95_low": low / k,
                    "precision_difference_bootstrap_ci95_high": high / k,
                    "recall_difference_a_minus_b": (hits_a - hits_b) / p,
                    "recall_difference_bootstrap_ci95_low": low / p,
                    "recall_difference_bootstrap_ci95_high": high / p,
                    "exact_paired_p_value": exact_p,
                    "bootstrap_replicates": REPLICATES, "random_seed": SEED,
                })
    report = pd.DataFrame(rows).sort_values(["year", "K", "model_a", "model_b"])
    report.to_csv(OUTPUT, index=False)
    print(report[[
        "year", "K", "model_a", "model_b", "hit_difference_a_minus_b",
        "hit_difference_bootstrap_ci95_low", "hit_difference_bootstrap_ci95_high", "exact_paired_p_value"
    ]].to_string(index=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
