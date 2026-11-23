"""Simulate random lists on P1's individual evaluation pool and report upstream coverage.

Random lists use the same individual evaluation pool and nested K values as
the P3 models. Coverage separately describes P1's broader upstream pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
KS = (100, 500, 1000)
YEARS = (2022, 2023)
REPLICATES = 20_000
SEED = 20261123


def main() -> None:
    coverage = pd.read_csv(REPORTS / "p4_input_coverage.csv")
    baseline = pd.read_csv(REPORTS / "p4_baseline_topk_metrics.csv")
    if set(coverage["year"]) != set(YEARS) or coverage["year"].duplicated().any():
        raise ValueError("Expected one coverage row per test year")
    rng = np.random.default_rng(SEED)
    random_rows = []
    example_lists = []
    coverage_rows = []
    for year in YEARS:
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet", columns=["npi", "year", "y_A_h2"])
        cohort["npi"] = cohort["npi"].astype("string")
        if not cohort["year"].eq(year).all() or cohort["npi"].duplicated().any():
            raise ValueError(f"{year}: invalid common cohort")
        if not cohort["y_A_h2"].isin((0, 1)).all():
            raise ValueError(f"{year}: invalid outcome")
        n = len(cohort)
        p = int(cohort["y_A_h2"].sum())
        cover = coverage.loc[coverage["year"].eq(year)].iloc[0]
        full_n = int(cover["eligible_candidates"])
        full_p = int(cover["eligible_positives"])
        if n != int(cover["with_all_three_scores"]) or p != int(cover["positives_with_all_three_scores"]):
            raise ValueError(f"{year}: coverage does not match common cohort")
        if n != int(cover["eval_pool_rows"]) or p != int(cover["eval_pool_positives"]):
            raise ValueError(f"{year}: scored cohort differs from P1's individual eval_pool")
        if full_n < n or full_p < p or max(KS) > n:
            raise ValueError(f"{year}: invalid coverage counts")
        coverage_rows.append({
            "year": year,
            "upstream_eligible_candidates": full_n,
            "upstream_later_exclusion_records": full_p,
            "individual_eval_pool": n,
            "eval_pool_later_exclusion_records": p,
            "eval_pool_share_of_upstream_candidates": n / full_n,
            "score_coverage_within_eval_pool": n / int(cover["eval_pool_rows"]),
            "upstream_candidates_outside_eval_pool": full_n - n,
            "upstream_positive_records_outside_eval_pool": full_p - p,
            "eval_pool_positive_share_of_upstream": p / full_p,
            "censored_candidates": int(cover["censored_candidates"]),
            "eval_pool_source": cover["eval_pool_source"],
        })

        labels = cohort["y_A_h2"].to_numpy(dtype=np.int8)
        names = cohort["npi"].to_numpy()
        draws = {k: np.empty(REPLICATES, dtype=np.int16) for k in KS}
        for replicate in range(REPLICATES):
            selected = rng.choice(n, max(KS), replace=False)
            if replicate == 0:
                for k in KS:
                    example_lists.append(pd.DataFrame({
                        "year": year, "K": k,
                        "rank": np.arange(1, k + 1, dtype=np.int32),
                        "npi": names[selected[:k]],
                        "y_A_h2": labels[selected[:k]],
                    }))
            selected_labels = labels[selected]
            for k in KS:
                draws[k][replicate] = int(selected_labels[:k].sum())

        for k in KS:
            sample = draws[k]
            expected = k * p / n
            variance = k * (p / n) * (1 - p / n) * (n - k) / (n - 1)
            standard_error = np.sqrt(variance / REPLICATES)
            if abs(float(sample.mean()) - expected) > 5 * standard_error + 1e-12:
                raise ValueError(f"{year} K={k}: random mean disagrees with exact expectation")
            reference = baseline.loc[
                baseline["year"].eq(year) & baseline["K"].eq(k)
                & baseline["approach"].eq("random_expected")
            ]
            if len(reference) != 1 or not np.isclose(float(reference.iloc[0]["hits_at_k"]), expected):
                raise ValueError(f"{year} K={k}: previous random expectation disagrees")
            random_rows.append({
                "year": year, "K": k,
                "pool_size": n, "positive_count": p,
                "replicates": REPLICATES, "random_seed": SEED,
                "mean_hits_at_k": float(sample.mean()),
                "sd_hits_at_k": float(sample.std(ddof=1)),
                "hits_at_k_p025": float(np.quantile(sample, 0.025)),
                "hits_at_k_p975": float(np.quantile(sample, 0.975)),
                "fraction_with_zero_hits": float(np.mean(sample == 0)),
                "fraction_with_at_least_one_hit": float(np.mean(sample >= 1)),
                "exact_expected_hits_at_k": expected,
                "exact_expected_precision_at_k": p / n,
            })

    random = pd.DataFrame(random_rows).sort_values(["year", "K"])
    cover_summary = pd.DataFrame(coverage_rows).sort_values("year")
    random.to_csv(REPORTS / "p4_seeded_random_baseline.csv", index=False)
    cover_summary.to_csv(REPORTS / "p4_scoring_coverage_summary.csv", index=False)
    pd.concat(example_lists, ignore_index=True).to_csv(DATA / "topk_seeded_random_example.csv", index=False)
    print(random[["year", "K", "mean_hits_at_k", "sd_hits_at_k", "fraction_with_zero_hits", "exact_expected_hits_at_k"]].to_string(index=False))
    print(cover_summary.to_string(index=False))
    print("Saved random baseline and evaluation-pool coverage reports")


if __name__ == "__main__":
    main()
