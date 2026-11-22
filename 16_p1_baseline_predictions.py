"""Quick end-to-end test with REAL data: simple peer-benchmarked rankings for the test years.
Writes out/p1_baseline_predictions.parquet (columns: npi, year, approach, score). NOT the team's model."""
import pandas as pd

TEST_YEARS = [2022, 2023]
SCORES = {"p1_baseline_pymt_per_bene_peer_z": "pymt_per_bene_peer_z",
          "p1_baseline_srvcs_per_bene_peer_z": "srvcs_per_bene_peer_z"}

feat = pd.read_parquet("out/provider_features.parquet", columns=["npi", "year"] + list(SCORES.values()),
                       filters=[("year", "in", TEST_YEARS)])
feat["npi"] = feat["npi"].astype(str).str.zfill(10); feat["year"] = feat["year"].astype(int)
pool = pd.read_parquet("out/eval_pool.parquet"); pool["npi"] = pool["npi"].astype(str)
pool = pool[pool["year"].isin(TEST_YEARS)]
d = pool.merge(feat, on=["npi", "year"], how="left")
print("pool rows (test years):", f"{len(d):,}")

parts = []
for name, col in SCORES.items():
    s = d[col].astype("float64")
    print(f"{name}: missing {int(s.isna().sum()):,} -> filled with the minimum (ranked last)")
    parts.append(pd.DataFrame({"npi": d["npi"], "year": d["year"], "approach": name, "score": s.fillna(s.min())}))
out = pd.concat(parts, ignore_index=True)
out.to_parquet("out/p1_baseline_predictions.parquet", index=False)
print("saved out/p1_baseline_predictions.parquet", f"{len(out):,} rows")