"""Prepare the common held-out P4 cohort; do not calculate ranking metrics.

Reads the four real Parquet inputs currently supplied under ``stubs/``.
Writes one compact cohort per test year under ignored ``out/p4/`` and a small
coverage report under ``reports/``. The old synthetic stub generator must not
be run against these input paths.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "stubs"
OUTPUT = ROOT / "out" / "p4"
REPORT = ROOT / "reports" / "p4_input_coverage.csv"
TEST_YEARS = (2022, 2023)
APPROACHES = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
KEY = ["npi", "year"]


def read_year(filename: str, columns: list[str], year: int) -> pd.DataFrame:
    path = INPUT / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=columns, filters=[("year", "==", year)])
    if frame.empty or not frame["year"].eq(year).all():
        raise ValueError(f"{filename}: missing or unexpected rows for {year}")
    return frame


def check_keys(frame: pd.DataFrame, name: str, keys: list[str]) -> None:
    if frame[keys].isna().any().any():
        raise ValueError(f"{name}: null key")
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate {keys}")
    if not frame["npi"].astype("string").str.fullmatch(r"\d{10}").all():
        raise ValueError(f"{name}: NPI must be a ten-digit string")


def prepare_year(year: int) -> dict[str, int]:
    candidates = read_year("candidate_pool.parquet", KEY, year)
    labels = read_year(
        "labels.parquet", KEY + ["eligible", "y_A_h2", "censored_h2"], year
    )
    features = read_year(
        "provider_features.parquet",
        KEY + ["medicare_payment", "pymt_per_bene", "specialty"],
        year,
    )
    predictions = read_year(
        "predictions.parquet", ["NPI", "year", "approach", "score"], year
    ).rename(columns={"NPI": "npi"})

    for name, frame in (
        ("candidate_pool", candidates),
        ("labels", labels),
        ("provider_features", features),
    ):
        check_keys(frame, name, KEY)
    check_keys(predictions, "predictions", KEY + ["approach"])
    if predictions["approach"].isna().any():
        raise ValueError("predictions: null approach")
    if set(predictions["approach"].unique()) != set(APPROACHES):
        raise ValueError("predictions: unexpected or missing approach")
    if not np.isfinite(predictions["score"].to_numpy()).all():
        raise ValueError("predictions: non-finite score")

    pool = candidates.merge(labels, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not pool["_merge"].eq("both").all():
        raise ValueError(f"{year}: candidate without label")
    pool = pool.drop(columns="_merge")
    if not pool["eligible"].eq(True).all():
        raise ValueError(f"{year}: candidate pool includes an ineligible provider")
    if not pool["y_A_h2"].isin([0, 1]).all():
        raise ValueError(f"{year}: non-binary or missing primary outcome")

    full_candidates = len(pool)
    full_positives = int(pool["y_A_h2"].sum())
    censored = int(pool["censored_h2"].sum())
    pool = pool.loc[~pool["censored_h2"]].drop(columns=["eligible", "censored_h2"])
    if pool.empty:
        raise ValueError(f"{year}: no uncensored candidates")

    pool = pool.merge(features, on=KEY, how="left", validate="one_to_one", indicator=True)
    has_features = pool["_merge"].eq("both")
    with_features = int(has_features.sum())
    positives_with_features = int(pool.loc[has_features, "y_A_h2"].sum())
    if not np.isfinite(pool.loc[has_features, ["medicare_payment", "pymt_per_bene"]].to_numpy()).all():
        raise ValueError(f"{year}: missing or non-finite payment baseline value")
    pool = pool.drop(columns="_merge")
    del candidates, labels, features
    gc.collect()

    for approach in APPROACHES:
        one = predictions.loc[
            predictions["approach"].eq(approach), KEY + ["score"]
        ].rename(columns={"score": f"score_{approach}"})
        pool = pool.merge(one, on=KEY, how="left", validate="one_to_one")
        del one
    del predictions
    gc.collect()

    score_columns = [f"score_{approach}" for approach in APPROACHES]
    all_scored = pool[score_columns].notna().all(axis=1)
    common = pool.loc[has_features & all_scored].copy()
    if common.empty:
        raise ValueError(f"{year}: no common scored candidates")
    if not np.isfinite(common[score_columns].to_numpy()).all():
        raise ValueError(f"{year}: non-finite common-pool score")
    common = common.sort_values("npi").reset_index(drop=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"cohort_{year}.parquet"
    common.to_parquet(destination, index=False, compression="zstd")

    result = {
        "year": year,
        "eligible_candidates": full_candidates,
        "eligible_positives": full_positives,
        "censored_candidates": censored,
        "uncensored_candidates": len(pool),
        "uncensored_positives": int(pool["y_A_h2"].sum()),
        "with_features": with_features,
        "positives_with_features": positives_with_features,
        "with_all_three_scores": len(common),
        "positives_with_all_three_scores": int(common["y_A_h2"].sum()),
    }
    print(f"{year}: {result}")
    print(f"Saved {destination}")
    return result


def main() -> None:
    rows = [prepare_year(year) for year in TEST_YEARS]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
