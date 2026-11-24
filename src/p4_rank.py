"""Create a deterministic provider ranking from a prepared P4 cohort.

This stage deliberately reads no outcome label and selects no top-K shortlist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
COHORT_DIR = ROOT / "out" / "p4"
APPROACHES = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
TEST_YEARS = (2022, 2023)


def rank_year(approach: str, year: int) -> None:
    source = COHORT_DIR / f"cohort_{year}.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)

    score_column = f"score_{approach}"
    ranked = pd.read_parquet(source, columns=["npi", "year", score_column])
    if ranked.empty or not ranked["year"].eq(year).all():
        raise ValueError(f"{source}: missing or unexpected year")
    if ranked["npi"].isna().any() or ranked["npi"].duplicated().any():
        raise ValueError(f"{source}: null or duplicate NPI")
    if not np.isfinite(ranked[score_column].to_numpy()).all():
        raise ValueError(f"{source}: non-finite {score_column}")

    ranked = ranked.rename(columns={score_column: "score"})
    ranked = ranked.sort_values(
        ["score", "npi"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked.insert(2, "approach", approach)
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=np.int32)

    destination = COHORT_DIR / f"ranking_{approach}_{year}.parquet"
    ranked.to_parquet(destination, index=False, compression="zstd")
    print(
        f"{year} {approach}: ranked {len(ranked):,} providers; "
        f"score range {ranked['score'].min():.6g} to {ranked['score'].max():.6g}; "
        f"saved {destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=APPROACHES, required=True)
    args = parser.parse_args()
    for year in TEST_YEARS:
        rank_year(args.approach, year)


if __name__ == "__main__":
    main()
