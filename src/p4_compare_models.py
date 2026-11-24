"""Compare P3 model shortlists at equal year and audit capacity.

Reports aggregate provider overlap and later-exclusion hits. The comparison
is descriptive; uncertainty in differences is a separate evaluation step.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
MODELS = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
YEARS = (2022, 2023)
KS = (100, 500, 1000)


def main() -> None:
    shortlists = {}
    metrics = {}
    for model in MODELS:
        selected = pd.read_csv(DATA / f"topk_{model}_provisional.csv", dtype={"npi": "string"})
        results = pd.read_csv(REPORTS / f"p4_topk_metrics_{model}.csv")
        if not selected["approach"].eq(model).all() or not results["approach"].eq(model).all():
            raise ValueError(f"Unexpected approach in {model} inputs")
        if selected.duplicated(["year", "K", "npi"]).any() or results.duplicated(["year", "K"]).any():
            raise ValueError(f"Duplicate provider or metric key for {model}")
        if set(zip(results["year"], results["K"])) != {(y, k) for y in YEARS for k in KS}:
            raise ValueError(f"Missing metric year/K for {model}")
        shortlists[model] = selected
        metrics[model] = results.set_index(["year", "K"])

    pair_rows = []
    all_rows = []
    for year in YEARS:
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet", columns=["npi", "year", "y_A_h2"])
        cohort["npi"] = cohort["npi"].astype("string")
        if not cohort["year"].eq(year).all() or cohort["npi"].duplicated().any():
            raise ValueError(f"Invalid cohort keys in {year}")
        if not cohort["y_A_h2"].isin((0, 1)).all():
            raise ValueError(f"Invalid outcomes in {year}")
        positive = set(cohort.loc[cohort["y_A_h2"].eq(1), "npi"])
        pool = set(cohort["npi"])
        if len(pool) != len(cohort):
            raise ValueError(f"Invalid cohort size in {year}")
        previous = {model: set() for model in MODELS}
        for k in KS:
            lists = {}
            for model in MODELS:
                frame = shortlists[model].loc[
                    shortlists[model]["year"].eq(year) & shortlists[model]["K"].eq(k)
                ].sort_values("rank")
                names = set(frame["npi"])
                if len(frame) != k or len(names) != k or not names <= pool:
                    raise ValueError(f"Invalid {year} K={k} shortlist for {model}")
                if not frame["rank"].reset_index(drop=True).eq(pd.Series(range(1, k + 1))).all():
                    raise ValueError(f"Invalid rank order for {model}")
                if not previous[model] <= names:
                    raise ValueError(f"Top-K lists are not nested for {model}")
                previous[model] = names
                result = metrics[model].loc[(year, k)]
                if int(result["common_pool_size"]) != len(pool) or int(result["common_pool_positives"]) != len(positive):
                    raise ValueError(f"Unequal model populations in {year}")
                if len(names & positive) != int(result["hits_at_k"]):
                    raise ValueError(f"Hit count mismatch for {model}, {year}, K={k}")
                lists[model] = names

            for a, b in combinations(MODELS, 2):
                common = lists[a] & lists[b]
                hits_a = lists[a] & positive
                hits_b = lists[b] & positive
                pair_rows.append({
                    "year": year, "K": k, "model_a": a, "model_b": b,
                    "hits_a": len(hits_a), "hits_b": len(hits_b),
                    "provider_overlap": len(common),
                    "provider_overlap_pct_of_k": len(common) / k,
                    "shared_hits": len(hits_a & hits_b),
                    "hits_only_a": len(hits_a - hits_b),
                    "hits_only_b": len(hits_b - hits_a),
                    "union_hits": len(hits_a | hits_b),
                })

            union = set.union(*(lists[model] for model in MODELS))
            triple = set.intersection(*(lists[model] for model in MODELS))
            hits_by_model = {model: lists[model] & positive for model in MODELS}
            hit_union = set.union(*hits_by_model.values())
            all_rows.append({
                "year": year, "K": k,
                "common_pool_size": len(pool), "common_pool_positives": len(positive),
                "hits_lgbm": len(hits_by_model["supervised_lgbm"]),
                "hits_logreg": len(hits_by_model["supervised_logreg"]),
                "hits_iforest": len(hits_by_model["anomaly_iforest"]),
                "providers_in_any_shortlist": len(union),
                "providers_in_all_three_shortlists": len(triple),
                "distinct_hits_in_any_shortlist": len(hit_union),
                "hits_found_only_by_lgbm": len(hits_by_model["supervised_lgbm"] - hits_by_model["supervised_logreg"] - hits_by_model["anomaly_iforest"]),
                "hits_found_only_by_logreg": len(hits_by_model["supervised_logreg"] - hits_by_model["supervised_lgbm"] - hits_by_model["anomaly_iforest"]),
                "hits_found_only_by_iforest": len(hits_by_model["anomaly_iforest"] - hits_by_model["supervised_lgbm"] - hits_by_model["supervised_logreg"]),
            })

    pair = pd.DataFrame(pair_rows).sort_values(["year", "K", "model_a", "model_b"])
    all_models = pd.DataFrame(all_rows).sort_values(["year", "K"])
    pair.to_csv(REPORTS / "p4_model_pairwise_overlap.csv", index=False)
    all_models.to_csv(REPORTS / "p4_model_shortlist_comparison.csv", index=False)
    print(all_models.to_string(index=False))
    print(f"Saved {REPORTS / 'p4_model_pairwise_overlap.csv'}")
    print(f"Saved {REPORTS / 'p4_model_shortlist_comparison.csv'}")


if __name__ == "__main__":
    main()
