"""Evaluate one P3 model's existing top-K lists on held-out test years.

Uses the prepared common cohort. Comparison against other models and payment
baselines, bootstrap intervals, and shortlist analysis are later stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
APPROACHES = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
TEST_YEARS = (2022, 2023)


def evaluate_year(
    year: int,
    approach: str,
    shortlists: pd.DataFrame,
    coverage: pd.DataFrame,
) -> list[dict]:
    cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet", columns=["npi", "year", "y_A_h2"])
    ranked = pd.read_parquet(
        DATA / f"ranking_{approach}_{year}.parquet",
        columns=["npi", "year", "approach", "score", "rank"],
    )
    if cohort["npi"].duplicated().any() or ranked["npi"].duplicated().any():
        raise ValueError(f"{year}: duplicate provider")
    if not cohort["year"].eq(year).all() or not ranked["year"].eq(year).all():
        raise ValueError(f"{year}: unexpected year")
    if not ranked["approach"].eq(approach).all() or len(cohort) != len(ranked):
        raise ValueError(f"{year}: ranking does not match common cohort")
    if not ranked["rank"].eq(np.arange(1, len(ranked) + 1)).all():
        raise ValueError(f"{year}: ranking order is invalid")
    if not cohort["y_A_h2"].isin([0, 1]).all():
        raise ValueError(f"{year}: invalid outcome")

    ranked = ranked.merge(cohort, on=["npi", "year"], how="left", validate="one_to_one")
    if ranked["y_A_h2"].isna().any():
        raise ValueError(f"{year}: ranked provider without outcome")
    n = len(ranked)
    positives = int(ranked["y_A_h2"].sum())
    if positives == 0:
        raise ValueError(f"{year}: no positive providers")

    cover = coverage.loc[coverage["year"].eq(year)]
    if len(cover) != 1:
        raise ValueError(f"{year}: missing coverage row")
    cover = cover.iloc[0]
    if int(cover["with_all_three_scores"]) != n or int(cover["positives_with_all_three_scores"]) != positives:
        raise ValueError(f"{year}: cohort disagrees with coverage report")

    rows: list[dict] = []
    for k in sorted(shortlists.loc[shortlists["year"].eq(year), "K"].unique()):
        k = int(k)
        if k > n:
            raise ValueError(f"{year}: K exceeds common pool")
        selected = shortlists.loc[
            shortlists["year"].eq(year) & shortlists["K"].eq(k)
        ].sort_values("rank")
        top = ranked.iloc[:k]
        if len(selected) != k or not selected["rank"].eq(np.arange(1, k + 1)).all():
            raise ValueError(f"{year} K={k}: invalid shortlist size or rank")
        if not selected["npi"].reset_index(drop=True).eq(top["npi"].reset_index(drop=True)).all():
            raise ValueError(f"{year} K={k}: shortlist does not match ranking")
        if not np.allclose(selected["score"].to_numpy(), top["score"].to_numpy(), rtol=0, atol=1e-12):
            raise ValueError(f"{year} K={k}: shortlist scores do not match ranking")

        hits = int(top["y_A_h2"].sum())
        cutoff_score = top["score"].iloc[-1]
        above = ranked["score"].gt(cutoff_score)
        same = ranked["score"].eq(cutoff_score)
        tie_size = int(same.sum())
        selected_from_tie = int(same.iloc[:k].sum())
        hits_above = int(ranked.loc[above, "y_A_h2"].sum())
        positives_in_tie = int(ranked.loc[same, "y_A_h2"].sum())
        min_hits = hits_above + max(0, selected_from_tie - (tie_size - positives_in_tie))
        max_hits = hits_above + min(selected_from_tie, positives_in_tie)

        upstream_positives = int(cover["eligible_positives"])
        rows.append(
            {
                "year": year,
                "approach": approach,
                "K": k,
                "common_pool_size": n,
                "common_pool_positives": positives,
                "upstream_candidate_size": int(cover["eligible_candidates"]),
                "upstream_candidate_positives": upstream_positives,
                "outside_eval_pool_upstream_candidates": int(cover["eligible_candidates"]) - n,
                "outside_eval_pool_upstream_positives": upstream_positives - positives,
                "hits_at_k": hits,
                "precision_at_k": hits / k,
                "recall_at_k": hits / positives,
                "lift_at_k": (hits / k) / (positives / n),
                "audits_per_hit": k / hits if hits else float("nan"),
                "cutoff_tie_size": tie_size,
                "selected_from_cutoff_tie": selected_from_tie,
                "positives_in_cutoff_tie": positives_in_tie,
                "min_hits_over_tie_orders": min_hits,
                "max_hits_over_tie_orders": max_hits,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=APPROACHES, required=True)
    args = parser.parse_args()
    shortlists = pd.read_csv(
        DATA / f"topk_{args.approach}_provisional.csv", dtype={"npi": "string"}
    )
    coverage = pd.read_csv(REPORTS / "p4_input_coverage.csv")
    if not shortlists["approach"].eq(args.approach).all():
        raise ValueError("Shortlist contains another approach")

    rows: list[dict] = []
    for year in TEST_YEARS:
        rows.extend(evaluate_year(year, args.approach, shortlists, coverage))
    report = pd.DataFrame(rows).sort_values(["year", "K"])
    destination = REPORTS / f"p4_topk_metrics_{args.approach}.csv"
    report.to_csv(destination, index=False)
    print(report[["year", "K", "hits_at_k", "precision_at_k", "recall_at_k", "lift_at_k", "audits_per_hit", "min_hits_over_tie_orders", "max_hits_over_tie_orders"]].to_string(index=False))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
