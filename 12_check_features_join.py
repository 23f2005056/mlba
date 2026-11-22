"""Join and leakage check of P2's feature file against the frozen labels.
Usage: python 12_check_features_join.py out/provider_features.parquet"""
import re, sys
import numpy as np, pandas as pd

FEAT = sys.argv[1] if len(sys.argv) > 1 else "out/provider_features.parquet"
COL, SCAN_YEAR = "y_A_h2", 2020          # year used for the single-feature leakage scan

lab = pd.read_parquet("out/labels.parquet", columns=["npi", "year", "eligible", COL])
f = pd.read_parquet(FEAT)
f.columns = [c.lower() if c.lower() in ("npi", "year") else c for c in f.columns]
f["npi"] = f["npi"].astype(str).str.strip().str.zfill(10)
f["year"] = f["year"].astype(int)

print(f"feature rows: {len(f):,} | label rows: {len(lab):,} | feature columns: {f.shape[1] - 2}")
print("duplicate (npi, year) in features:", int(f.duplicated(['npi', 'year']).sum()))

m = lab.merge(f, on=["npi", "year"], how="left", indicator=True)
has = m["_merge"] == "both"
print("\nCOVERAGE (share of label rows that have a feature row), by year:")
print(m.groupby("year").apply(lambda g: round((g["_merge"] == "both").mean(), 4), include_groups=False).to_string())
miss_pos = m[(m[COL] == 1) & ~has].groupby("year").size()
print("\nPOSITIVES WITHOUT FEATURES, by year (should be ~0):")
print(miss_pos.to_string() if len(miss_pos) else "none")

bad = [c for c in f.columns if re.search(r"(^y_|excl|eligible|censor|leie|label|fraud|target)", c, re.I)]
print("\nSUSPICIOUS COLUMN NAMES (must not be features):", bad if bad else "none")

d = m[(m["year"] == SCAN_YEAR) & m["eligible"] & has]
P, N = int(d[COL].sum()), len(d)
print(f"\nSINGLE-FEATURE SCAN, year {SCAN_YEAR}: {N:,} eligible providers, {P} positives")
num = [c for c in f.columns if c not in ("npi", "year") and pd.api.types.is_numeric_dtype(f[c])]
res = []
for c in num:
    s = d[c].astype(float); s = s.fillna(s.min())
    r = s.rank(method="average").values
    auc = (r[d[COL].values == 1].sum() - P * (P + 1) / 2) / (P * (N - P))
    res.append((c, round(auc, 3)))
res = pd.DataFrame(res, columns=["feature", "auc"]).assign(dist=lambda x: (x.auc - 0.5).abs()).sort_values("dist", ascending=False)
print(res.head(12).drop(columns="dist").to_string(index=False))
print("\nAny feature with AUC >= 0.85 or <= 0.15 is almost certainly leaking label information. Ask P2 how it is built.")
