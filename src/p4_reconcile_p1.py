"""Reconcile P1's percentage-cutoff results and add peer baselines.

The provider-cluster bootstrap resamples outcome-positive NPIs jointly across
2022 and 2023, retaining every year for a sampled NPI. It conditions on the
observed negative-provider scores and fixed rankings; it does not refit models.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "p4"
REPORTS = ROOT / "reports"
FEATURES = ROOT / "stubs" / "provider_features.parquet"
YEARS = (2022, 2023)
QS = (0.01, 0.05)
KS = (100, 500, 1000)
REPLICATES = 20_000
SEED = 20261123
MODELS = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
PEERS = {
    "p1_baseline_pymt_per_bene_peer_z": "pymt_per_bene_peer_z",
    "p1_baseline_srvcs_per_bene_peer_z": "srvcs_per_bene_peer_z",
}
SCORES = {
    "supervised_lgbm": "score_supervised_lgbm",
    "supervised_logreg": "score_supervised_logreg",
    "anomaly_iforest": "score_anomaly_iforest",
    **PEERS,
}
# P1's independent message, rounded to one decimal percentage point and
# three AUC decimals. The only tie-dependent result is 2022 LightGBM at 1%.
P1 = {
    (2022, "anomaly_iforest"): (0.678, 0.073, 0.159),
    (2023, "anomaly_iforest"): (0.687, 0.069, 0.241),
    (2022, "supervised_logreg"): (0.819, 0.146, 0.427),
    (2023, "supervised_logreg"): (0.756, 0.103, 0.379),
    (2022, "supervised_lgbm"): (0.790, 0.220, 0.402),
    (2023, "supervised_lgbm"): (0.745, 0.103, 0.345),
    (2022, "p1_baseline_pymt_per_bene_peer_z"): (0.541, 0.085, 0.195),
    (2023, "p1_baseline_pymt_per_bene_peer_z"): (0.588, 0.052, 0.190),
    (2022, "p1_baseline_srvcs_per_bene_peer_z"): (0.570, 0.073, 0.146),
    (2023, "p1_baseline_srvcs_per_bene_peer_z"): (0.612, 0.052, 0.190),
}


def cutoff_counts(scores: np.ndarray, labels: np.ndarray, order: np.ndarray, k: int) -> dict:
    cutoff = scores[order[k - 1]]
    above = scores > cutoff
    tied = scores == cutoff
    selected_tied = k - int(above.sum())
    hits_above = int(labels[above].sum())
    positives_tied = int(labels[tied].sum())
    tie_size = int(tied.sum())
    return {
        "hits": int(labels[order[:k]].sum()),
        "cutoff_tie_size": tie_size,
        "selected_from_cutoff_tie": selected_tied,
        "positives_in_cutoff_tie": positives_tied,
        "min_hits_over_tie_orders": hits_above + max(0, selected_tied - (tie_size - positives_tied)),
        "max_hits_over_tie_orders": hits_above + min(selected_tied, positives_tied),
    }


def main() -> None:
    frames = {}
    positive_npis = set()
    for year in YEARS:
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet")
        peer = pd.read_parquet(
            FEATURES,
            columns=["npi", "year", *PEERS.values()],
            filters=[("year", "==", year)],
        )
        if peer.duplicated(["npi", "year"]).any():
            raise ValueError(f"Duplicate P2 feature key in {year}")
        cohort = cohort.merge(peer, on=["npi", "year"], how="left", validate="one_to_one")
        if cohort["npi"].duplicated().any() or not cohort["y_A_h2"].isin((0, 1)).all():
            raise ValueError(f"Invalid evaluation cohort for {year}")
        frames[year] = cohort
        positive_npis.update(cohort.loc[cohort["y_A_h2"].eq(1), "npi"].astype(str))
    provider_ids = np.array(sorted(positive_npis))
    if len(provider_ids) != 113:
        raise ValueError(f"Expected 113 distinct positive NPIs, got {len(provider_ids)}")
    provider_index = {npi: i for i, npi in enumerate(provider_ids)}
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(provider_ids), size=(REPLICATES, len(provider_ids)), dtype=np.int16)

    percent_rows = []
    baseline_rows = []
    baseline_shortlists = []
    boot_rows = []
    paired_rows = []
    vectors = {}
    positive_presence = {}
    for year in YEARS:
        cohort = frames[year]
        y = cohort["y_A_h2"].to_numpy(dtype=np.int8)
        n = len(y)
        p = int(y.sum())
        positive_mask = y == 1
        positive_indices = np.array([provider_index[str(npi)] for npi in cohort.loc[positive_mask, "npi"]])
        presence = np.bincount(positive_indices, minlength=len(provider_ids)).astype(np.int8)
        positive_presence[year] = presence
        if presence.sum() != p or presence.max() > 1:
            raise ValueError(f"Invalid positive NPI mapping for {year}")
        denom = presence[draw].sum(axis=1)
        if (denom == 0).any():
            raise ValueError(f"Bootstrap draw has no positives in {year}")
        names = cohort["npi"].astype(str).to_numpy()
        for approach, column in SCORES.items():
            s = cohort[column].to_numpy(dtype=np.float64)
            missing = int(np.isnan(s).sum())
            if missing:
                if approach not in PEERS or np.isnan(s).all():
                    raise ValueError(f"Unexpected missing scores for {approach}, {year}")
                s = np.where(np.isnan(s), np.nanmin(s), s)
            if not np.isfinite(s).all():
                raise ValueError(f"Non-finite scores for {approach}, {year}")
            order = np.lexsort((names, -s))
            auc = float(roc_auc_score(y, s))
            negative_sorted = np.sort(s[~positive_mask])
            positive_scores = s[positive_mask]
            left = np.searchsorted(negative_sorted, positive_scores, side="left")
            right = np.searchsorted(negative_sorted, positive_scores, side="right")
            auc_contribution = (left + 0.5 * (right - left)) / len(negative_sorted)
            if not np.isclose(auc_contribution.mean(), auc, atol=1e-12):
                raise ValueError(f"AUC concordance mismatch for {approach}, {year}")
            vec_auc = np.zeros(len(provider_ids), dtype=np.float64)
            vec_auc[positive_indices] = auc_contribution
            vectors[(year, approach, "auc")] = vec_auc

            record = {"year": year, "approach": approach, "pool_size": n,
                      "positive_count": p, "roc_auc": auc, "missing_scores_filled_at_minimum": missing}
            for q in QS:
                k = int(np.ceil(q * n))
                info = cutoff_counts(s, y, order, k)
                tag = f"{int(q * 100)}pct"
                record[f"K_{tag}"] = k
                record[f"hits_{tag}"] = info["hits"]
                record[f"recall_{tag}"] = info["hits"] / p
                for field in ("cutoff_tie_size", "selected_from_cutoff_tie", "positives_in_cutoff_tie",
                              "min_hits_over_tie_orders", "max_hits_over_tie_orders"):
                    record[f"{field}_{tag}"] = info[field]
                captured = np.zeros(n, dtype=np.int8)
                captured[order[:k]] = 1
                vec = np.zeros(len(provider_ids), dtype=np.float64)
                vec[positive_indices] = captured[positive_mask]
                vectors[(year, approach, f"recall_{tag}")] = vec
            p1_auc, p1_r1, p1_r5 = P1[(year, approach)]
            record.update(p1_reported_auc=p1_auc, p1_reported_recall_1pct=p1_r1,
                          p1_reported_recall_5pct=p1_r5)
            if abs(auc - p1_auc) > 0.00051:
                raise ValueError(f"P1 AUC mismatch for {approach}, {year}")
            for tag, reference in (("1pct", p1_r1), ("5pct", p1_r5)):
                if not (record[f"min_hits_over_tie_orders_{tag}"] / p - 0.00051 <= reference
                        <= record[f"max_hits_over_tie_orders_{tag}"] / p + 0.00051):
                    raise ValueError(f"P1 recall outside cutoff tie bounds for {approach}, {year}, {tag}")
            percent_rows.append(record)

            if approach in PEERS:
                for k in KS:
                    info = cutoff_counts(s, y, order, k)
                    baseline_shortlists.append(pd.DataFrame({
                        "year": year, "approach": approach, "K": k,
                        "rank": np.arange(1, k + 1, dtype=np.int32),
                        "npi": names[order[:k]], "ranking_value": s[order[:k]],
                    }))
                    baseline_rows.append({
                        "year": year, "approach": approach, "K": k,
                        "pool_size": n, "positive_count": p,
                        "hits_at_k": info["hits"],
                        "precision_at_k": info["hits"] / k,
                        "recall_at_k": info["hits"] / p,
                        "lift_at_k": (info["hits"] / k) / (p / n),
                        "audits_per_hit": k / info["hits"] if info["hits"] else np.nan,
                        **{field: info[field] for field in info if field != "hits"},
                    })
            del order, negative_sorted

        for metric in ("auc", "recall_1pct", "recall_5pct"):
            for approach in SCORES:
                vec = vectors[(year, approach, metric)]
                samples = vec[draw].sum(axis=1) / denom
                low, high = np.quantile(samples, (0.025, 0.975))
                boot_rows.append({"period": str(year), "approach": approach, "metric": metric,
                                  "estimate": float(vec.sum() / p), "ci95_low": low, "ci95_high": high,
                                  "bootstrap_replicates": REPLICATES, "random_seed": SEED,
                                  "bootstrap_scope": "positive_NPI_clusters; observed negatives fixed"})
            for a, b in combinations(MODELS, 2):
                va, vb = vectors[(year, a, metric)], vectors[(year, b, metric)]
                differences = (va - vb)[draw].sum(axis=1) / denom
                low, high = np.quantile(differences, (0.025, 0.975))
                paired_rows.append({"period": str(year), "metric": metric, "model_a": a, "model_b": b,
                                    "difference_a_minus_b": float((va - vb).sum() / p),
                                    "ci95_low": low, "ci95_high": high,
                                    "bootstrap_replicates": REPLICATES, "random_seed": SEED})

    pooled_presence = sum(positive_presence.values())
    pooled_denom = pooled_presence[draw].sum(axis=1)
    for metric in ("auc", "recall_1pct", "recall_5pct"):
        pooled_metric = "year_weighted_auc" if metric == "auc" else metric
        for approach in SCORES:
            vec = sum(vectors[(year, approach, metric)] for year in YEARS)
            samples = vec[draw].sum(axis=1) / pooled_denom
            low, high = np.quantile(samples, (0.025, 0.975))
            boot_rows.append({"period": "2022+2023", "approach": approach, "metric": pooled_metric,
                              "estimate": float(vec.sum() / pooled_presence.sum()),
                              "ci95_low": low, "ci95_high": high,
                              "bootstrap_replicates": REPLICATES, "random_seed": SEED,
                              "bootstrap_scope": "positive_NPI_clusters; observed negatives fixed"})
        for a, b in combinations(MODELS, 2):
            va = sum(vectors[(year, a, metric)] for year in YEARS)
            vb = sum(vectors[(year, b, metric)] for year in YEARS)
            differences = (va - vb)[draw].sum(axis=1) / pooled_denom
            low, high = np.quantile(differences, (0.025, 0.975))
            paired_rows.append({"period": "2022+2023", "metric": pooled_metric,
                                "model_a": a, "model_b": b,
                                "difference_a_minus_b": float((va - vb).sum() / pooled_presence.sum()),
                                "ci95_low": low, "ci95_high": high,
                                "bootstrap_replicates": REPLICATES, "random_seed": SEED})

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(percent_rows).sort_values(["year", "approach"]).to_csv(
        REPORTS / "p4_p1_percentage_reconciliation.csv", index=False)
    pd.DataFrame(baseline_rows).sort_values(["year", "K", "approach"]).to_csv(
        REPORTS / "p4_p1_peer_baseline_topk.csv", index=False)
    pd.concat(baseline_shortlists, ignore_index=True).to_csv(
        DATA / "topk_p1_peer_baselines.csv", index=False)
    pd.DataFrame(boot_rows).sort_values(["period", "metric", "approach"]).to_csv(
        REPORTS / "p4_p1_percentage_bootstrap.csv", index=False)
    pd.DataFrame(paired_rows).sort_values(["period", "metric", "model_a", "model_b"]).to_csv(
        REPORTS / "p4_p1_percentage_paired_bootstrap.csv", index=False)
    print(pd.DataFrame(percent_rows)[["year", "approach", "roc_auc", "hits_1pct", "recall_1pct",
                                      "hits_5pct", "recall_5pct"]].to_string(index=False))
    print("Saved P1 reconciliation, peer baselines, provider-bootstrap and paired-bootstrap reports")


if __name__ == "__main__":
    main()
