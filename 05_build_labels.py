"""Self-contained label builder (does not need your other scripts).
Reads: raw/leie/UPDATED.csv, optional raw/leie/rein/*.csv, and out/cms_npi_year.parquet (columns npi, year)
Writes: out/labels.parquet, out/candidate_pool.parquet, out/label_audit.csv, out/split_plan.csv
"""
import re, glob
import numpy as np, pandas as pd
from pathlib import Path

# ------------------------- SETTINGS (edit here) -------------------------
LEIE_MAIN    = "raw/leie/UPDATED.csv"
REIN_GLOB    = "raw/leie/rein/*.csv"
CMS_NPI_YEAR = "out/cms_npi_year.parquet"
OUT = Path("out")
MIN_YEAR, MAX_YEAR = 2013, 2023            # CMS years you will really use (use 2017 if P2 only trusts 2017-2023)
HORIZONS = [1, 2, 3]
SCENARIO_A = {"1128a1", "1128a3", "1128b1", "1128b6", "1128b7"}   # your team's fraud-related codes
PLAN = {2: {"val": [2020], "test": [2022, 2023]},                  # horizon -> prediction years
        3: {"val": [2020], "test": [2022]}}
THR = dict(green=50, amber=20)
# ------------------------------------------------------------------------

def npi_ok(s):
    if not re.fullmatch(r"\d{10}", s) or s == "0000000000":
        return False
    t = 0
    for i, ch in enumerate(reversed("80840" + s)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9: d -= 9
        t += d
    return t % 10 == 0

def read_leie(path, source):
    d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    d.columns = [c.strip().upper() for c in d.columns]
    for c in ["NPI", "EXCLTYPE", "EXCLDATE", "REINDATE"]:
        if c not in d.columns: d[c] = ""
    return pd.DataFrame({
        "npi_raw": d["NPI"].str.strip(),
        "excl_type": d["EXCLTYPE"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True),
        "excl_date": pd.to_datetime(d["EXCLDATE"].str.strip(), format="%Y%m%d", errors="coerce"),
        "rein_date": pd.to_datetime(d["REINDATE"].str.strip(), format="%Y%m%d", errors="coerce"),  # "0" -> NaT
        "source": source})

parts = [read_leie(LEIE_MAIN, "main")]
for f in sorted(glob.glob(REIN_GLOB)):
    parts.append(read_leie(f, "rein"))
leie = pd.concat(parts, ignore_index=True)
CUTOFF = leie["excl_date"].max()
print(f"LEIE rows: {len(leie)} (supplement rows: {(leie.source=='rein').sum()}) | latest exclusion date in file: {CUTOFF.date()}")
print("rows with NO usable exclusion date:", int(leie["excl_date"].isna().sum()))

leie["npi"] = leie["npi_raw"].where(leie["npi_raw"].map(npi_ok), pd.NA)
ev = leie[leie["npi"].notna() & leie["excl_date"].notna()].drop_duplicates(["npi", "excl_type", "excl_date"]).copy()
ev["ey"] = ev["excl_date"].dt.year.astype(int)
ev["ry"] = ev["rein_date"].dt.year
ev["scenario_a"] = ev["excl_type"].isin(SCENARIO_A)
print(f"valid-NPI exclusion events: {len(ev)} | unique NPIs: {ev.npi.nunique()} | scenario A events: {int(ev.scenario_a.sum())}")

cms = pd.read_parquet(CMS_NPI_YEAR)[["npi", "year"]].drop_duplicates()
cms["npi"] = cms["npi"].astype(str).str.strip().str.zfill(10)
cms["year"] = cms["year"].astype(int)
cms = cms[cms["year"].between(MIN_YEAR, MAX_YEAR)].reset_index(drop=True)
YEARS = list(range(MIN_YEAR, MAX_YEAR + 1))
print("CMS rows:", len(cms), "| NPIs:", cms.npi.nunique(), "| years:", MIN_YEAR, "-", MAX_YEAR)

# ---- coverage: how many excluded NPIs ever appear in CMS? ----
in_cms = set(cms["npi"])
for name, sub in (("any exclusion type", ev), ("scenario A", ev[ev.scenario_a])):
    u = set(sub["npi"])
    print(f"coverage - {name}: {len(u & in_cms)} of {len(u)} excluded NPIs appear in CMS {MIN_YEAR}-{MAX_YEAR}")

# ---- work only on the small set of CMS rows whose NPI is in LEIE ----
small = cms[cms["npi"].isin(set(ev["npi"]))][["npi", "year"]].copy()

rows = []
for npi, ey, ry in zip(ev["npi"], ev["ey"], ev["ry"]):
    last = MAX_YEAR if pd.isna(ry) else min(MAX_YEAR, int(ry) - 1)
    if MIN_YEAR <= ey <= last and (pd.isna(ry) or ry > ey):
        rows.append((npi, ey, 1))
    for T in range(max(ey + 1, MIN_YEAR), last + 1):
        rows.append((npi, T, 2))
st = pd.DataFrame(rows, columns=["npi", "year", "code"]).groupby(["npi", "year"], as_index=False)["code"].max()
small = small.merge(st, on=["npi", "year"], how="left")
small["excluded_before_year"] = small["code"].eq(2)
small["excluded_in_year"] = small["code"].eq(1)
small["eligible"] = ~(small["excluded_before_year"] | small["excluded_in_year"])
small = small.drop(columns="code")

def pos_pairs(sub, h):
    r = [(n, T) for n, ey in zip(sub["npi"], sub["ey"]) for T in range(ey - h, ey) if MIN_YEAR <= T <= MAX_YEAR]
    return pd.DataFrame(r, columns=["npi", "year"]).drop_duplicates()

ycols = []
for S, sub in (("A", ev[ev["scenario_a"]]), ("B", ev)):
    for h in HORIZONS:
        col = f"y_{S}_h{h}"; ycols.append(col)
        p = pos_pairs(sub, h); p[col] = 1
        small = small.merge(p, on=["npi", "year"], how="left")
        small[col] = small[col].fillna(0).astype("int8")
        small.loc[~small["eligible"], col] = 0

lab = cms.merge(small, on=["npi", "year"], how="left")
for c in ["excluded_before_year", "excluded_in_year"]:
    lab[c] = lab[c].fillna(False).astype(bool)
lab["eligible"] = lab["eligible"].fillna(True).astype(bool)
for c in ycols:
    lab[c] = lab[c].fillna(0).astype("int8")
for h in HORIZONS:
    cens = {T: pd.Timestamp(f"{T + h}-12-31") > CUTOFF for T in YEARS}
    lab[f"censored_h{h}"] = lab["year"].map(cens).astype(bool)
lab["npi"] = lab["npi"].astype("string"); lab["year"] = lab["year"].astype("int16")

OUT.mkdir(exist_ok=True)
lab.to_parquet(OUT / "labels.parquet", index=False)
lab.loc[lab["eligible"], ["npi", "year"]].to_parquet(OUT / "candidate_pool.parquet", index=False)

# ---- audit ----
rows = []
for S in ("A", "B"):
    for h in HORIZONS:
        col = f"y_{S}_h{h}"
        g = lab.groupby("year").agg(eligible=("eligible", "sum"), positives=(col, "sum")).reset_index()
        for r in g.itertuples():
            br = r.positives / r.eligible if r.eligible else np.nan
            rows.append(dict(scenario=S, horizon=h, year=r.year, eligible=int(r.eligible), positives=int(r.positives),
                             base_rate=br, censored=bool(lab.loc[lab.year == r.year, f"censored_h{h}"].iloc[0]),
                             exp_hits_random_K10=10 * br, exp_hits_random_K100=100 * br, exp_hits_random_K1000=1000 * br))
aud = pd.DataFrame(rows); aud.to_csv(OUT / "label_audit.csv", index=False)
print("\n=== POSITIVES PER YEAR (eligible providers excluded within h years; * = window not fully observed) ===")
for S in ("A", "B"):
    for h in HORIZONS:
        a = aud[(aud.scenario == S) & (aud.horizon == h)]
        print(f"scenario {S}, h={h}: " + " ".join(f"{r.year}:{r.positives}{'*' if r.censored else ''}" for r in a.itertuples()))
print("\n=== EXCLUDED PROVIDERS STILL IN CMS (QA) ===")
print(lab.groupby("year")[["excluded_before_year", "excluded_in_year"]].sum().T.to_string())

def allowed_train(T, h): return [y for y in YEARS if y + h <= T]
plan_rows = []
print("\n=== SPLIT PLAN AND DISTINCT POSITIVE PROVIDERS ===")
for h, p in PLAN.items():
    for role in ("val", "test"):
        for T in p[role]:
            plan_rows.append(dict(horizon=h, role=role, prediction_year=T, allowed_train_years=" ".join(map(str, allowed_train(T, h)))))
    for S in ("A", "B"):
        col = f"y_{S}_h{h}"
        for role in ("val", "test"):
            n = lab.loc[lab["year"].isin(p[role]) & lab[col].eq(1), "npi"].nunique()
            flag = "GREEN" if n >= THR["green"] else "AMBER" if n >= THR["amber"] else "RED"
            print(f"h={h} scenario {S} {role:4s} {p[role]}: {n:5d} distinct positives [{flag}]")
        trn = sorted({y for T in (p['val'] + p['test']) for y in allowed_train(T, h)})
        n = lab.loc[lab["year"].isin(trn) & lab[col].eq(1), "npi"].nunique()
        print(f"h={h} scenario {S} train years {trn[0]}-{trn[-1]}: {n} distinct positives")
pd.DataFrame(plan_rows).to_csv(OUT / "split_plan.csv", index=False)
print("\nsaved to", OUT)