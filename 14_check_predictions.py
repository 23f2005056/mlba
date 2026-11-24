"""P1 integrity check of the team's predictions + an independent recomputation of the headline metrics.
Usage: python 14_check_predictions.py out/predictions.parquet"""
import sys
import numpy as np, pandas as pd

PRED = sys.argv[1] if len(sys.argv) > 1 else "out/predictions.parquet"
H, COL = 2, "y_A_h2"

plan = pd.read_csv("out/split_plan.csv")
test_years = sorted(plan[(plan.horizon == H) & (plan.role == "test")]["prediction_year"].unique())
lab = pd.read_parquet("out/labels.parquet", columns=["npi", "year", "eligible", COL])

pred = pd.read_parquet(PRED)
pred.columns = [c.lower() for c in pred.columns]
score_c = next(c for c in pred.columns if c in ("score", "risk_score", "prediction", "pred", "anomaly_score"))
app_c = next(c for c in pred.columns if c in ("approach", "model"))
pred["npi"] = pred["npi"].astype(str).str.strip().str.zfill(10)
pred["year"] = pred["year"].astype(int)
print(f"columns: {list(pred.columns)} | score column: {score_c} | approach column: {app_c}")
print("approaches:", sorted(pred[app_c].unique()), "| years present:", sorted(pred["year"].unique()), "| expected test years:", test_years)

print("\n=== INTEGRITY CHECKS ===")
print("duplicate (npi, year, approach) rows:", int(pred.duplicated(["npi", "year", app_c]).sum()))
print("missing scores:", int(pred[score_c].isna().sum()))
pool_keys = pd.read_parquet(
    "out/eval_pool.parquet",
    columns=["npi", "year"]
)
pool_keys["npi"] = pool_keys["npi"].astype(str)
lab["npi"] = lab["npi"].astype(str)
elig = lab.merge(pool_keys, on=["npi", "year"], how="inner")      # evaluation pool = eligible individuals with features
outside = pred.merge(pool_keys.assign(_in=True), on=["npi", "year"], how="left")["_in"].isna().sum()
print("predictions for providers OUTSIDE the evaluation pool (organizations / ineligible):", int(outside))
m = pred.merge(lab[["npi", "year", "eligible", COL]], on=["npi", "year"], how="left")
print("predictions for providers NOT in the labels file:", int(m["eligible"].isna().sum()))
print("predictions for INELIGIBLE providers (should be 0):", int((m["eligible"] == False).sum()))

rows = []
for a in sorted(pred[app_c].unique()):
    for y in test_years:
        pool = elig[elig["year"] == y]
        pa = pred[(pred[app_c] == a) & (pred["year"] == y)]
        cov = pa["npi"].isin(pool["npi"]).sum() / len(pool) if len(pool) else np.nan
        d = pool.merge(pa[["npi", score_c]], on="npi", how="inner")
        P, N = int(d[COL].sum()), len(d)
        if P == 0 or N == 0:
            continue
        s = d[score_c].values; yv = d[COL].values
        r = pd.Series(s).rank(method="average").values
        auc = (r[yv == 1].sum() - P * (P + 1) / 2) / (P * (N - P))
        # Deterministic ranking:
        # 1. Score descending
        # 2. Ten-digit NPI ascending for equal scores

        npi = d["npi"].astype(str).str.strip().str.zfill(10).to_numpy()

        o = np.lexsort((npi, -s))

        # Select exactly the ceiling of the specified pool percentages
        k1 = int(np.ceil(len(pool) * 0.01))
        k5 = int(np.ceil(len(pool) * 0.05))

        rec1 = yv[o[:k1]].sum() / P
        rec5 = yv[o[:k5]].sum() / P
        rows.append(dict(approach=a, year=y, pool=len(pool), coverage=round(cov, 4), positives=P, auc=round(auc, 3),
                         recall_1pct=round(rec1, 3), lift_1pct=round(rec1 / 0.01, 1),
                         recall_5pct=round(rec5, 3), lift_5pct=round(rec5 / 0.05, 1)))
res = pd.DataFrame(rows)
print("\n=== INDEPENDENT METRICS (compare with P4's results.csv) ===")
print(res.to_string(index=False))
low = res[res["coverage"] < 0.99]
if len(low):
    print("\n!! COVERAGE BELOW 99% - the model did not score the full eligible pool for:")
    print(low[["approach", "year", "coverage"]].to_string(index=False))
    print("   Metrics on a partial pool are biased. Ask P3 to score every eligible provider.")