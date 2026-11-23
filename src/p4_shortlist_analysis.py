"""Describe LightGBM shortlists and their overlap with payment baselines.

Medicare payment is observed billing in the decision year, not fraud or
recoverable savings. No personally identifying provider list is published.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
YEARS = (2022, 2023)
KS = (100, 500, 1000)
BASELINES = ("highest_total_payment", "highest_payment_per_beneficiary")


def main() -> None:
    model = pd.read_csv(DATA / "topk_supervised_lgbm_provisional.csv", dtype={"npi": "string"})
    baselines = pd.read_csv(DATA / "topk_payment_baselines.csv", dtype={"npi": "string"})
    metrics = pd.read_csv(REPORTS / "p4_topk_metrics_supervised_lgbm.csv")
    summaries = []
    specialties = []
    for year in YEARS:
        cohort = pd.read_parquet(
            DATA / f"cohort_{year}.parquet",
            columns=["npi", "year", "y_A_h2", "medicare_payment", "pymt_per_bene", "specialty"],
        )
        cohort["npi"] = cohort["npi"].astype("string")
        if cohort["npi"].duplicated().any() or not cohort["year"].eq(year).all():
            raise ValueError(f"{year}: invalid cohort keys")
        if not cohort["y_A_h2"].isin((0, 1)).all():
            raise ValueError(f"{year}: invalid outcome")
        by_npi = cohort.set_index("npi")

        for k in KS:
            selected = model.loc[model["year"].eq(year) & model["K"].eq(k)].sort_values("rank")
            if len(selected) != k or selected["npi"].duplicated().any():
                raise ValueError(f"{year} K={k}: invalid LightGBM shortlist")
            if not np.array_equal(selected["rank"].to_numpy(), np.arange(1, k + 1)):
                raise ValueError(f"{year} K={k}: invalid rank order")
            if not selected["npi"].isin(by_npi.index).all():
                raise ValueError(f"{year} K={k}: shortlist outside cohort")
            top = by_npi.loc[selected["npi"]].copy()
            expected = metrics.loc[metrics["year"].eq(year) & metrics["K"].eq(k)]
            if len(expected) != 1 or int(top["y_A_h2"].sum()) != int(expected.iloc[0]["hits_at_k"]):
                raise ValueError(f"{year} K={k}: hit count disagrees with evaluation")
            # Source values may be float32; aggregate in float64 so specialty
            # subtotals reconcile with the overall shortlist total.
            top["medicare_payment"] = top["medicare_payment"].astype("float64")

            row = {
                "year": year,
                "approach": "supervised_lgbm",
                "K": k,
                "hits_at_k": int(top["y_A_h2"].sum()),
                "observed_medicare_payment_sum": float(top["medicare_payment"].sum()),
                "observed_medicare_payment_median": float(top["medicare_payment"].median()),
                "payment_per_beneficiary_median": float(top["pymt_per_bene"].median()),
                "distinct_specialties": int(top["specialty"].nunique(dropna=True)),
                "missing_specialty": int(top["specialty"].isna().sum()),
            }
            for baseline in BASELINES:
                other = baselines.loc[
                    baselines["year"].eq(year)
                    & baselines["K"].eq(k)
                    & baselines["approach"].eq(baseline)
                ]
                if len(other) != k or other["npi"].duplicated().any():
                    raise ValueError(f"{year} K={k}: invalid {baseline} shortlist")
                overlap = len(set(selected["npi"]) & set(other["npi"]))
                row[f"overlap_with_{baseline}"] = overlap
                row[f"overlap_pct_with_{baseline}"] = overlap / k
            summaries.append(row)

            counts = top.groupby("specialty", dropna=False).agg(
                providers=("y_A_h2", "size"),
                later_exclusions=("y_A_h2", "sum"),
                observed_medicare_payment_sum=("medicare_payment", "sum"),
            ).reset_index()
            counts["year"] = year
            counts["K"] = k
            counts["specialty"] = counts["specialty"].fillna("[missing]")
            specialties.append(counts)

    summary = pd.DataFrame(summaries).sort_values(["year", "K"])
    specialty = pd.concat(specialties, ignore_index=True).sort_values(
        ["year", "K", "providers", "specialty"], ascending=[True, True, False, True]
    )
    summary.to_csv(REPORTS / "p4_lightgbm_shortlist_summary.csv", index=False)
    specialty.to_csv(REPORTS / "p4_lightgbm_shortlist_specialties.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved {REPORTS / 'p4_lightgbm_shortlist_summary.csv'}")
    print(f"Saved {REPORTS / 'p4_lightgbm_shortlist_specialties.csv'}")


if __name__ == "__main__":
    main()
