"""Build the evaluation pool = eligible providers that also have a feature row (individual practitioners),
and recount the positives on THAT pool. Does not modify labels.parquet."""
import re
import numpy as np, pandas as pd

PLAN = {2: {"val": [2020], "test": [2022, 2023]}, 3: {"val": [2020], "test": [2022]}}
THR = dict(green=50, amber=20)

lab = pd.read_parquet("out/labels.parquet")
lab["npi"] = lab["npi"].astype(str)
keys = pd.read_parquet("out/provider_features.parquet", columns=["npi", "year"])
keys["npi"] = keys["npi"].astype(str).str.zfill(10); keys["year"] = keys["year"].astype("int16")
lab = lab.merge(keys.assign(in_features=True), on=["npi", "year"], how="left")
lab["in_features"] = lab["in_features"].fillna(False).astype(bool)
lab["in_pool"] = lab["eligible"] & lab["in_features"]

pool = lab.loc[lab["in_pool"], ["npi", "year"]]
pool.to_parquet("out/eval_pool.parquet", index=False)
print(f"label rows: {len(lab):,} | eligible: {int(lab.eligible.sum()):,} | eligible AND has features (eval pool): {len(pool):,}")

ycols = sorted(c for c in lab.columns if re.fullmatch(r"y_[AB]_h\d", c))
print("\n=== POSITIVES: all eligible providers  ->  evaluation pool (individuals with features) ===")
rows = []
for c in ycols:
    for y, g in lab.groupby("year"):
        rows.append(dict(label=c, year=int(y), pool=int(g.in_pool.sum()), positives_pool=int(g.loc[g.in_pool, c].sum()),
                         positives_all_eligible=int(g.loc[g.eligible, c].sum()),
                         base_rate_pool=g.loc[g.in_pool, c].sum() / max(g.in_pool.sum(), 1)))
aud = pd.DataFrame(rows); aud.to_csv("out/label_audit_pool.csv", index=False)
for c in ("y_A_h2", "y_B_h2", "y_A_h3"):
    if c in ycols:
        a = aud[aud.label == c]
        print(f"{c}: " + " ".join(f"{r.year}:{r.positives_pool}(of {r.positives_all_eligible})" for r in a.itertuples()))
print("(format: positives in pool (of positives among all eligible rows))")
r2022 = aud[(aud.label == "y_A_h2") & (aud.year == 2022)]
if len(r2022):
    r = r2022.iloc[0]
    print(f"\n2022, y_A_h2: pool {r.pool:,}, positives {r.positives_pool}, base rate {r.base_rate_pool:.5%}; "
          f"random top-10 expects {10 * r.base_rate_pool:.4f} hits, random top-1% expects {0.01 * r.positives_pool:.2f} hits")

def allowed_train(T, h): return [y for y in sorted(lab.year.unique()) if y + h <= T]
print("\n=== DISTINCT POSITIVE PROVIDERS ON THE EVAL POOL ===")
for h, p in PLAN.items():
    for S in "AB":
        c = f"y_{S}_h{h}"
        if c not in ycols: continue
        for role in ("val", "test"):
            n = lab.loc[lab.year.isin(p[role]) & lab.in_pool & lab[c].eq(1), "npi"].nunique()
            flag = "GREEN" if n >= THR["green"] else "AMBER" if n >= THR["amber"] else "RED"
            print(f"h={h} scenario {S} {role:4s} {p[role]}: {n:4d} [{flag}]")
        trn = sorted({y for T in p["val"] + p["test"] for y in allowed_train(T, h)})
        n = lab.loc[lab.year.isin(trn) & lab.in_pool & lab[c].eq(1), "npi"].nunique()
        print(f"h={h} scenario {S} train {trn[0]}-{trn[-1]}: {n}")
print("\nsaved out/eval_pool.parquet and out/label_audit_pool.csv")