"""Complete a P3 model's baseline, interval, and shortlist reports.

Uses the same held-out cohort, K values, and existing payment baselines as
LightGBM. Billing is historical exposure, not recoverable savings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p4_confidence_intervals import exact_interval


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
APPROACHES = {"supervised_logreg": "logreg", "anomaly_iforest": "iforest"}
PAYMENT_BASELINES = ("highest_total_payment", "highest_payment_per_beneficiary")


def check_keys(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"Duplicate {description} keys")


def main(approach: str) -> None:
    alias = APPROACHES[approach]
    metrics = pd.read_csv(REPORTS / f"p4_topk_metrics_{approach}.csv")
    baselines = pd.read_csv(REPORTS / "p4_baseline_topk_metrics.csv")
    model_lists = pd.read_csv(DATA / f"topk_{approach}_provisional.csv", dtype={"npi": "string"})
    payment_lists = pd.read_csv(DATA / "topk_payment_baselines.csv", dtype={"npi": "string"})
    if not metrics["approach"].eq(approach).all() or set(metrics["year"]) != {2022, 2023}:
        raise ValueError("Unexpected model metrics")
    if set(metrics["K"]) != {100, 500, 1000} or len(metrics) != 6:
        raise ValueError("Expected three K values in both test years")
    check_keys(metrics, ["year", "K"], "model metric")
    check_keys(baselines, ["year", "K", "approach"], "baseline metric")
    for row in metrics.itertuples(index=False):
        matches = baselines.loc[baselines["year"].eq(row.year) & baselines["K"].eq(row.K)]
        if set(matches["approach"]) != {*PAYMENT_BASELINES, "random_expected"}:
            raise ValueError("Missing baseline")
        if not matches["pool_size"].eq(row.common_pool_size).all() or not matches["positive_count"].eq(row.common_pool_positives).all():
            raise ValueError("Baseline and model populations differ")

    comparison = metrics[["year", "K", "hits_at_k", "precision_at_k"]].merge(
        baselines[["year", "K", "approach", "hits_at_k", "precision_at_k"]],
        on=["year", "K"], suffixes=(f"_{alias}", "_baseline"), validate="one_to_many"
    ).rename(columns={"approach": "baseline"})
    comparison[f"incremental_hits_{alias}"] = comparison[f"hits_at_k_{alias}"] - comparison["hits_at_k_baseline"]
    comparison[f"precision_difference_{alias}"] = comparison[f"precision_at_k_{alias}"] - comparison["precision_at_k_baseline"]
    comparison.sort_values(["year", "K", "baseline"]).to_csv(REPORTS / f"p4_{alias}_vs_baselines.csv", index=False)

    interval_rows = []
    interval_input = pd.concat([
        metrics.rename(columns={"common_pool_size": "pool_size", "common_pool_positives": "positive_count"})
        [["year", "approach", "K", "pool_size", "positive_count", "hits_at_k"]],
        baselines.loc[baselines["approach"].isin(PAYMENT_BASELINES),
                      ["year", "approach", "K", "pool_size", "positive_count", "hits_at_k"]],
    ], ignore_index=True)
    for row in interval_input.itertuples(index=False):
        n, p, k, hits = int(row.pool_size), int(row.positive_count), int(row.K), int(row.hits_at_k)
        if not (0 <= hits <= min(k, p) and 0 < p <= n and 0 < k <= n):
            raise ValueError("Invalid interval input counts")
        precision_low, precision_high = exact_interval(hits, k)
        recall_low, recall_high = exact_interval(hits, p)
        prevalence = p / n
        interval_rows.append({
            "year": row.year, "approach": row.approach, "K": k,
            "pool_size": n, "positive_count": p, "hits_at_k": hits,
            "precision_at_k": hits / k,
            "precision_ci95_low": precision_low, "precision_ci95_high": precision_high,
            "recall_at_k": hits / p,
            "recall_ci95_low": recall_low, "recall_ci95_high": recall_high,
            "lift_at_k": (hits / k) / prevalence,
            "lift_ci95_low_conditional_on_prevalence": precision_low / prevalence,
            "lift_ci95_high_conditional_on_prevalence": precision_high / prevalence,
        })
    pd.DataFrame(interval_rows).sort_values(["year", "K", "approach"]).to_csv(
        REPORTS / f"p4_topk_confidence_intervals_{approach}.csv", index=False
    )

    summaries = []
    specialties = []
    check_keys(model_lists, ["year", "K", "npi"], "model shortlist")
    check_keys(payment_lists, ["year", "K", "approach", "npi"], "payment shortlist")
    for year in (2022, 2023):
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet", columns=[
            "npi", "year", "y_A_h2", "medicare_payment", "pymt_per_bene", "specialty"
        ])
        cohort["npi"] = cohort["npi"].astype("string")
        if not cohort["year"].eq(year).all() or cohort["npi"].duplicated().any() or not cohort["y_A_h2"].isin((0, 1)).all():
            raise ValueError(f"Invalid cohort for {year}")
        by_npi = cohort.set_index("npi")
        for k in (100, 500, 1000):
            selected = model_lists.loc[model_lists["year"].eq(year) & model_lists["K"].eq(k)].sort_values("rank")
            if len(selected) != k or not np.array_equal(selected["rank"].to_numpy(), np.arange(1, k + 1)):
                raise ValueError(f"Invalid shortlist for {year} K={k}")
            if not selected["npi"].isin(by_npi.index).all():
                raise ValueError("Shortlist outside cohort")
            top = by_npi.loc[selected["npi"]].copy()
            expected_hits = int(metrics.loc[metrics["year"].eq(year) & metrics["K"].eq(k), "hits_at_k"].iloc[0])
            if int(top["y_A_h2"].sum()) != expected_hits:
                raise ValueError(f"Hit count mismatch for {year} K={k}")
            top["medicare_payment"] = top["medicare_payment"].astype("float64")
            summary = {
                "year": year, "approach": approach, "K": k, "hits_at_k": expected_hits,
                "observed_medicare_payment_sum": float(top["medicare_payment"].sum()),
                "observed_medicare_payment_median": float(top["medicare_payment"].median()),
                "payment_per_beneficiary_median": float(top["pymt_per_bene"].median()),
                "distinct_specialties": int(top["specialty"].nunique(dropna=True)),
                "missing_specialty": int(top["specialty"].isna().sum()),
            }
            for baseline in PAYMENT_BASELINES:
                other = payment_lists.loc[
                    payment_lists["year"].eq(year) & payment_lists["K"].eq(k)
                    & payment_lists["approach"].eq(baseline)
                ]
                if len(other) != k or other["npi"].duplicated().any():
                    raise ValueError(f"Invalid {baseline} list")
                overlap = len(set(selected["npi"]) & set(other["npi"]))
                summary[f"overlap_with_{baseline}"] = overlap
                summary[f"overlap_pct_with_{baseline}"] = overlap / k
            summaries.append(summary)
            breakdown = top.groupby("specialty", dropna=False).agg(
                providers=("y_A_h2", "size"),
                later_exclusions=("y_A_h2", "sum"),
                observed_medicare_payment_sum=("medicare_payment", "sum"),
            ).reset_index()
            breakdown["year"] = year
            breakdown["K"] = k
            breakdown["specialty"] = breakdown["specialty"].fillna("[missing]")
            specialties.append(breakdown)
    pd.DataFrame(summaries).sort_values(["year", "K"]).to_csv(
        REPORTS / f"p4_{alias}_shortlist_summary.csv", index=False
    )
    pd.concat(specialties, ignore_index=True).sort_values(
        ["year", "K", "providers", "specialty"], ascending=[True, True, False, True]
    ).to_csv(REPORTS / f"p4_{alias}_shortlist_specialties.csv", index=False)
    print(metrics[["year", "K", "hits_at_k", "precision_at_k", "recall_at_k", "lift_at_k"]].to_string(index=False))
    print(f"Saved {approach} comparison, intervals, and shortlist reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=APPROACHES, required=True)
    main(parser.parse_args().approach)
