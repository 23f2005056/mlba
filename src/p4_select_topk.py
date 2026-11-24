"""Select provisional top-K shortlists from a ranked P4 test cohort.

This stage does not read outcomes or calculate evaluation metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RANKING_DIR = ROOT / "out" / "p4"
APPROACHES = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
TEST_YEARS = (2022, 2023)


def select_year(approach: str, year: int, ks: list[int]) -> tuple[list[pd.DataFrame], list[dict]]:
    source = RANKING_DIR / f"ranking_{approach}_{year}.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    ranked = pd.read_parquet(source, columns=["npi", "year", "approach", "score", "rank"])
    if ranked.empty or not ranked["year"].eq(year).all():
        raise ValueError(f"{source}: unexpected year")
    if not ranked["approach"].eq(approach).all():
        raise ValueError(f"{source}: unexpected approach")
    if ranked["rank"].iloc[0] != 1 or ranked["rank"].iloc[-1] != len(ranked):
        raise ValueError(f"{source}: invalid rank range")
    if len(ranked) < max(ks):
        raise ValueError(f"{source}: K exceeds number of ranked providers")

    shortlists: list[pd.DataFrame] = []
    tie_checks: list[dict] = []
    for k in ks:
        top = ranked.iloc[:k].copy()
        top.insert(2, "K", k)
        shortlists.append(top[["year", "approach", "K", "rank", "npi", "score"]])

        cutoff_score = ranked["score"].iloc[k - 1]
        same_score = ranked["score"].eq(cutoff_score)
        tie_size = int(same_score.sum())
        selected_from_tie = int(same_score.iloc[:k].sum())
        tie_checks.append(
            {
                "year": year,
                "approach": approach,
                "K": k,
                "cutoff_score": cutoff_score,
                "cutoff_tie_size": tie_size,
                "selected_from_cutoff_tie": selected_from_tie,
                "tie_crosses_cutoff": selected_from_tie < tie_size,
            }
        )
        print(
            f"{year} K={k}: selected {len(top)} providers; "
            f"cutoff tie {selected_from_tie}/{tie_size} selected"
        )
    return shortlists, tie_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=APPROACHES, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[100, 500, 1000])
    args = parser.parse_args()
    ks = sorted(set(args.k))
    if not ks or ks[0] < 1:
        raise ValueError("K values must be positive")

    shortlists: list[pd.DataFrame] = []
    tie_checks: list[dict] = []
    for year in TEST_YEARS:
        lists, checks = select_year(args.approach, year, ks)
        shortlists.extend(lists)
        tie_checks.extend(checks)

    shortlist_path = RANKING_DIR / f"topk_{args.approach}_provisional.csv"
    pd.concat(shortlists, ignore_index=True).to_csv(shortlist_path, index=False)
    tie_path = ROOT / "reports" / f"p4_topk_ties_{args.approach}.csv"
    pd.DataFrame(tie_checks).to_csv(tie_path, index=False)
    print(f"Saved {shortlist_path}")
    print(f"Saved {tie_path}")


if __name__ == "__main__":
    main()
