"""Exact 95% intervals for fixed-shortlist top-K evaluation results.

These intervals describe uncertainty in the observed hit fraction under a
binomial model. They do not include model-training uncertainty, uncertainty
about who will be excluded, or a paired comparison between shortlists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
INPUTS = (
    REPORTS / "p4_topk_metrics_supervised_lgbm.csv",
    REPORTS / "p4_baseline_topk_metrics.csv",
)
OUTPUT = REPORTS / "p4_topk_confidence_intervals.csv"
ALPHA = 0.05


def exact_interval(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval with finite boundary values."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Invalid success or trial count")
    lower = 0.0 if successes == 0 else float(beta.ppf(ALPHA / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - ALPHA / 2, successes + 1, trials - successes))
    return lower, upper


def main() -> None:
    model, baselines = (pd.read_csv(path) for path in INPUTS)
    model = model.rename(columns={"common_pool_size": "pool_size", "common_pool_positives": "positive_count"})
    baselines = baselines.loc[baselines["approach"].ne("random_expected")]
    columns = ["year", "approach", "K", "pool_size", "positive_count", "hits_at_k"]
    data = pd.concat([model[columns], baselines[columns]], ignore_index=True)
    if data.duplicated(["year", "approach", "K"]).any():
        raise ValueError("Duplicate year/approach/K result")
    if set(data["approach"]) != {"supervised_lgbm", "highest_total_payment", "highest_payment_per_beneficiary"}:
        raise ValueError("Unexpected approach")
    if len(data) != 18:
        raise ValueError("Expected three approaches, three K values, and two years")

    rows = []
    for row in data.itertuples(index=False):
        n, positives, k, hits = (int(row.pool_size), int(row.positive_count), int(row.K), int(row.hits_at_k))
        if not (0 <= hits <= min(k, positives) and 0 < positives <= n and 0 < k <= n):
            raise ValueError(f"Invalid counts: {row}")
        precision_low, precision_high = exact_interval(hits, k)
        recall_low, recall_high = exact_interval(hits, positives)
        prevalence = positives / n
        rows.append({
            "year": int(row.year),
            "approach": row.approach,
            "K": k,
            "pool_size": n,
            "positive_count": positives,
            "hits_at_k": hits,
            "precision_at_k": hits / k,
            "precision_ci95_low": precision_low,
            "precision_ci95_high": precision_high,
            "recall_at_k": hits / positives,
            "recall_ci95_low": recall_low,
            "recall_ci95_high": recall_high,
            "lift_at_k": (hits / k) / prevalence,
            "lift_ci95_low_conditional_on_prevalence": precision_low / prevalence,
            "lift_ci95_high_conditional_on_prevalence": precision_high / prevalence,
        })
    result = pd.DataFrame(rows).sort_values(["year", "K", "approach"])
    result.to_csv(OUTPUT, index=False)
    print(result[["year", "approach", "K", "hits_at_k", "precision_at_k", "precision_ci95_low", "precision_ci95_high"]].to_string(index=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
