"""Evaluate simple audit rankings on the same held-out P4 cohorts as LightGBM.

The random baseline is reported as its exact expectation rather than a single
noisy draw. Confidence intervals and financial exposure are later stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
LIGHTGBM_REPORT = REPORTS / "p4_topk_metrics_supervised_lgbm.csv"
BASELINE_REPORT = REPORTS / "p4_baseline_topk_metrics.csv"
COMPARISON_REPORT = REPORTS / "p4_lightgbm_vs_baselines.csv"
SHORTLISTS = DATA / "topk_payment_baselines.csv"
TEST_YEARS = (2022, 2023)
BASELINES = {
    "highest_total_payment": "medicare_payment",
    "highest_payment_per_beneficiary": "pymt_per_bene",
}


def evaluate_year(year: int, ks: list[int]) -> tuple[list[dict], list[pd.DataFrame]]:
    cohort = pd.read_parquet(
        DATA / f"cohort_{year}.parquet",
        columns=["npi", "year", "y_A_h2", "medicare_payment", "pymt_per_bene"],
    )
    if cohort.empty or not cohort["year"].eq(year).all():
        raise ValueError(f"{year}: missing or unexpected cohort year")
    if cohort["npi"].duplicated().any() or not cohort["y_A_h2"].isin([0, 1]).all():
        raise ValueError(f"{year}: duplicate provider or invalid outcome")
    if not np.isfinite(cohort[["medicare_payment", "pymt_per_bene"]].to_numpy()).all():
        raise ValueError(f"{year}: non-finite baseline value")

    n = len(cohort)
    positives = int(cohort["y_A_h2"].sum())
    if positives == 0 or max(ks) > n:
        raise ValueError(f"{year}: no positives or K exceeds cohort")
    prevalence = positives / n
    rows: list[dict] = []
    shortlists: list[pd.DataFrame] = []

    for approach, score_column in BASELINES.items():
        ranked = cohort.sort_values(
            [score_column, "npi"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        for k in ks:
            top = ranked.iloc[:k].copy()
            hits = int(top["y_A_h2"].sum())
            cutoff = ranked[score_column].iloc[k - 1]
            above = ranked[score_column].gt(cutoff)
            same = ranked[score_column].eq(cutoff)
            tie_size = int(same.sum())
            selected_from_tie = int(same.iloc[:k].sum())
            hits_above = int(ranked.loc[above, "y_A_h2"].sum())
            positives_in_tie = int(ranked.loc[same, "y_A_h2"].sum())
            rows.append(
                {
                    "year": year,
                    "approach": approach,
                    "K": k,
                    "pool_size": n,
                    "positive_count": positives,
                    "hits_at_k": hits,
                    "precision_at_k": hits / k,
                    "recall_at_k": hits / positives,
                    "lift_at_k": (hits / k) / prevalence,
                    "audits_per_hit": k / hits if hits else float("nan"),
                    "cutoff_tie_size": tie_size,
                    "selected_from_cutoff_tie": selected_from_tie,
                    "min_hits_over_tie_orders": hits_above
                    + max(0, selected_from_tie - (tie_size - positives_in_tie)),
                    "max_hits_over_tie_orders": hits_above
                    + min(selected_from_tie, positives_in_tie),
                }
            )
            top.insert(0, "K", k)
            top.insert(1, "approach", approach)
            top.insert(2, "rank", np.arange(1, k + 1, dtype=np.int32))
            shortlists.append(
                top[["year", "approach", "K", "rank", "npi", score_column, "y_A_h2"]].rename(
                    columns={score_column: "ranking_value"}
                )
            )

    for k in ks:
        rows.append(
            {
                "year": year,
                "approach": "random_expected",
                "K": k,
                "pool_size": n,
                "positive_count": positives,
                "hits_at_k": k * prevalence,
                "precision_at_k": prevalence,
                "recall_at_k": k / n,
                "lift_at_k": 1.0,
                "audits_per_hit": float("nan"),
                "cutoff_tie_size": 0,
                "selected_from_cutoff_tie": 0,
                "min_hits_over_tie_orders": float("nan"),
                "max_hits_over_tie_orders": float("nan"),
            }
        )
    return rows, shortlists


def main() -> None:
    lgbm = pd.read_csv(LIGHTGBM_REPORT)
    if not lgbm["approach"].eq("supervised_lgbm").all():
        raise ValueError("Expected only LightGBM rows")
    ks = sorted(int(k) for k in lgbm["K"].unique())
    rows: list[dict] = []
    shortlists: list[pd.DataFrame] = []
    for year in TEST_YEARS:
        year_rows, year_lists = evaluate_year(year, ks)
        rows.extend(year_rows)
        shortlists.extend(year_lists)

    baselines = pd.DataFrame(rows).sort_values(["year", "K", "approach"])
    baselines.to_csv(BASELINE_REPORT, index=False)
    pd.concat(shortlists, ignore_index=True).to_csv(SHORTLISTS, index=False)

    comparison = lgbm[["year", "K", "hits_at_k", "precision_at_k"]].merge(
        baselines[["year", "K", "approach", "hits_at_k", "precision_at_k"]],
        on=["year", "K"], suffixes=("_lgbm", "_baseline"), validate="one_to_many"
    )
    comparison = comparison.rename(columns={"approach": "baseline"})
    comparison["incremental_hits_lgbm"] = (
        comparison["hits_at_k_lgbm"] - comparison["hits_at_k_baseline"]
    )
    comparison["precision_difference_lgbm"] = (
        comparison["precision_at_k_lgbm"] - comparison["precision_at_k_baseline"]
    )
    comparison.sort_values(["year", "K", "baseline"]).to_csv(COMPARISON_REPORT, index=False)
    print(baselines[["year", "K", "approach", "hits_at_k", "precision_at_k"]].to_string(index=False))
    print(f"Saved {BASELINE_REPORT}")
    print(f"Saved {COMPARISON_REPORT}")
    print(f"Saved {SHORTLISTS}")


if __name__ == "__main__":
    main()
