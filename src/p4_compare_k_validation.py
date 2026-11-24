"""Compare audit capacities on 2020 validation data only.

Do not use 2022/2023 test outcomes to choose K. This report measures later
Scenario A exclusions found per audit, not government money saved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "out" / "p4" / "cohort_2020.parquet"
REPORT = ROOT / "reports" / "p4_k_validation.csv"
KS = (100, 500, 1000)
METHODS = {
    "supervised_lgbm": "score_supervised_lgbm",
    "supervised_logreg": "score_supervised_logreg",
    "anomaly_iforest": "score_anomaly_iforest",
    "highest_total_payment": "medicare_payment",
    "highest_payment_per_beneficiary": "pymt_per_bene",
}


def main() -> None:
    columns = ["npi", "year", "y_A_h2"] + list(METHODS.values())
    cohort = pd.read_parquet(SOURCE, columns=columns)
    if cohort.empty or not cohort["year"].eq(2020).all():
        raise ValueError("Validation input must contain only 2020")
    if cohort["npi"].duplicated().any():
        raise ValueError("Duplicate provider in validation cohort")
    if not cohort["y_A_h2"].isin([0, 1]).all():
        raise ValueError("Invalid validation outcome")

    n = len(cohort)
    positives = int(cohort["y_A_h2"].sum())
    if positives == 0 or n < max(KS):
        raise ValueError("No positives or insufficient candidates")
    base_rate = positives / n

    rows: list[dict] = []
    for method, score_col in METHODS.items():
        if cohort[score_col].isna().any():
            raise ValueError(f"{method}: missing score")
        ordered = cohort.sort_values(
            [score_col, "npi"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        for k in KS:
            selected = ordered.iloc[:k]
            hits = int(selected["y_A_h2"].sum())
            cutoff_score = ordered[score_col].iloc[k - 1]
            above_score = ordered[score_col].gt(cutoff_score)
            same_score = ordered[score_col].eq(cutoff_score)
            tie_size = int(same_score.sum())
            selected_from_tie = int(same_score.iloc[:k].sum())
            hits_above_tie = int(ordered.loc[above_score, "y_A_h2"].sum())
            positives_in_tie = int(ordered.loc[same_score, "y_A_h2"].sum())
            min_hits = hits_above_tie + max(
                0, selected_from_tie - (tie_size - positives_in_tie)
            )
            max_hits = hits_above_tie + min(selected_from_tie, positives_in_tie)
            rows.append(
                {
                    "year": 2020,
                    "role": "validation",
                    "approach": method,
                    "K": k,
                    "pool_size": n,
                    "positive_count": positives,
                    "hits_at_k": hits,
                    "precision_at_k": hits / k,
                    "recall_at_k": hits / positives,
                    "lift_at_k": (hits / k) / base_rate,
                    "audits_per_hit": k / hits if hits else float("nan"),
                    "tie_crosses_cutoff": selected_from_tie < tie_size,
                    "cutoff_tie_size": tie_size,
                    "selected_from_cutoff_tie": selected_from_tie,
                    "positives_in_cutoff_tie": positives_in_tie,
                    "min_hits_over_tie_orders": min_hits,
                    "max_hits_over_tie_orders": max_hits,
                    "expected_hits_random_tie_order": hits_above_tie
                    + selected_from_tie * positives_in_tie / tie_size,
                }
            )

    # Random selection has an exact expectation; one random draw would be
    # particularly misleading with this rare outcome.
    for k in KS:
        rows.append(
            {
                "year": 2020,
                "role": "validation",
                "approach": "random_expected",
                "K": k,
                "pool_size": n,
                "positive_count": positives,
                "hits_at_k": k * base_rate,
                "precision_at_k": base_rate,
                "recall_at_k": k / n,
                "lift_at_k": 1.0,
                "audits_per_hit": float("nan"),
                "tie_crosses_cutoff": False,
                "cutoff_tie_size": 0,
                "selected_from_cutoff_tie": 0,
                "positives_in_cutoff_tie": 0,
                "min_hits_over_tie_orders": float("nan"),
                "max_hits_over_tie_orders": float("nan"),
                "expected_hits_random_tie_order": float("nan"),
            }
        )

    result = pd.DataFrame(rows)
    payment_hits = result.loc[
        result["approach"].eq("highest_total_payment"), ["K", "hits_at_k"]
    ].set_index("K")["hits_at_k"]
    result["incremental_hits_vs_total_payment"] = result["hits_at_k"] - result["K"].map(payment_hits)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(["K", "approach"]).to_csv(REPORT, index=False)
    print(result.sort_values(["K", "approach"])[
        ["approach", "K", "hits_at_k", "precision_at_k", "incremental_hits_vs_total_payment", "audits_per_hit", "tie_crosses_cutoff"]
    ].to_string(index=False))
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
