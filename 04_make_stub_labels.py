"""Gives P3/P4 a tiny labels.parquet with the REAL schema so nobody waits for you."""
import numpy as np, pandas as pd
from config import *
rng = np.random.default_rng(0)
npis = [f"{1000000000 + i}" for i in range(10)]
lab = pd.DataFrame([(n, y) for n in npis for y in YEARS], columns=["npi", "year"])
lab["excluded_before_year"] = False
lab["excluded_in_year"] = False
lab["eligible"] = True
for S in ("A", "B"):
    for h in HORIZONS:
        lab[f"y_{S}_h{h}"] = (rng.random(len(lab)) < 0.1).astype("int8")
for h in HORIZONS:
    lab[f"censored_h{h}"] = lab["year"] + h > 2026
lab["npi"] = lab["npi"].astype("string"); lab["year"] = lab["year"].astype("int16")
(OUT / "stub").mkdir(exist_ok=True)
lab.to_parquet(OUT / "stub" / "labels.parquet", index=False)
print(lab.head(3).to_string()); print("columns:", list(lab.columns))