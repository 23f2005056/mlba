"""
FraudLens - Phase 4 feature engineering

Reads  data/processed/provider_year.parquet      (build_dataset.py)
Writes data/processed/features.parquet           (one row per provider-year)
       reports/feature_dictionary.csv            (every feature documented)

Design rules (write these in your model card)
  * Peer group  = cleaned specialty x billing year. Groups with < MIN_PEER providers
    get NULL peer features (not made-up ones).
  * Robust z-score = (x - median) / (1.4826 * MAD) on log-scaled metrics, clipped to +/-10.
  * NO LEAKAGE IN TIME: a provider-year uses only that year and EARLIER years
    (LAG look-backs only). No feature looks at later years, exclusion info, or
    whether the provider keeps billing.
  * Banned as model inputs: labels, state, zip, year, NPI, specialty ID.
  * Finance columns (expected_pay, excess_payment) are kept for Phase 7 ranking.

Run:  python features.py        (no pandas needed)
"""
from pathlib import Path

import duckdb

PARQUET = Path("data/processed/provider_year.parquet")
OUT_PARQUET = Path("data/processed/features.parquet")
DICT_CSV = Path("reports/feature_dictionary.csv")
MIN_PEER = 30
CLIP_Z = 10

# 2016 files used different spellings. Same specialty, different label.
SPECIALTY_FIX = {
    "Cardiovascular Disease (Cardiology)": "Cardiology",
    "Clinical Cardiatric Electrophysiology": "Clinical Cardiac Electrophysiology",
    "Oral Surgery (Dentists only)": "Oral Surgery (Dentist only)",
}

# metric name -> SQL (log-scaled so heavy tails do not dominate)
METRICS = {
    "ppb":   "ln(1 + tot_mdcr_pymt_amt / tot_benes)",
    "spb":   "ln(1 + tot_srvcs / tot_benes)",
    "cr":    "ln(1 + tot_sbmtd_chrg / NULLIF(tot_mdcr_alowd_amt, 0))",
    "ra":    "ln(1 + (tot_mdcr_pymt_amt / tot_benes) / NULLIF(bene_avg_risk_scre, 0))",
    "codes": "ln(NULLIF(tot_hcpcs_cds, 0))",
    "pps":   "ln(1 + tot_mdcr_pymt_amt / NULLIF(tot_srvcs, 0))",
    "benes": "ln(tot_benes)",
}
RANKED = ["ppb", "spb", "ra", "pps"]          # also get a within-peer percentile rank
DZ = ["ppb", "spb"]                           # change in peer position vs prior year

DESCRIPTIONS = {
    "ppb": "Medicare payment per beneficiary", "spb": "services per beneficiary",
    "cr": "submitted charges / allowed amount (charge ratio)",
    "ra": "payment per beneficiary / avg beneficiary risk score",
    "codes": "number of distinct HCPCS codes (procedure breadth)",
    "pps": "Medicare payment per service", "benes": "number of beneficiaries (volume)",
}


def sq(p):
    return "'" + str(Path(p).resolve()).replace("\\", "/").replace("'", "''") + "'"


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


con = duckdb.connect(":memory:")

# ---------------------------------------------------------------- base
banner("1. BASE TABLE (specialty cleaned, log metrics)")
fix = "CASE specialty " + " ".join(
    f"WHEN '{k}' THEN '{v}'" for k, v in SPECIALTY_FIX.items()) + " ELSE specialty END"
m_cols = ", ".join(f"{sql} AS m_{k}" for k, sql in METRICS.items())
con.execute(f"""
    CREATE TABLE base AS
    SELECT cms_year, npi, {fix} AS spec, y_A, y_B, y_tier1,
           tot_benes, tot_srvcs, tot_mdcr_pymt_amt, tot_mdcr_alowd_amt,
           bene_avg_risk_scre, bene_avg_age,
           tot_mdcr_pymt_amt / tot_benes AS ppb_raw,
           tot_srvcs / tot_benes AS spb_raw,
           {m_cols}
    FROM read_parquet({sq(PARQUET)})""")
n = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
print(f"Rows: {n:,}")
print("Specialties after cleaning:",
      con.execute("SELECT COUNT(DISTINCT spec) FROM base").fetchone()[0])

# ---------------------------------------------------------------- peer stats
banner("2. PEER GROUP MEDIANS AND MAD (specialty x year)")
med_sel = ", ".join(f"median(m_{k}) AS med_{k}" for k in METRICS)
con.execute(f"""
    CREATE TABLE peer_med AS
    SELECT spec, cms_year, COUNT(*) AS peer_n, {med_sel},
           median(ppb_raw) AS med_ppb_raw, median(spb_raw) AS med_spb_raw
    FROM base GROUP BY 1, 2""")
mad_sel = ", ".join(f"median(abs(b.m_{k} - p.med_{k})) AS mad_{k}" for k in METRICS)
con.execute(f"""
    CREATE TABLE peer_mad AS
    SELECT b.spec, b.cms_year, {mad_sel}
    FROM base b JOIN peer_med p USING (spec, cms_year) GROUP BY 1, 2""")
small = con.execute(f"SELECT COUNT(*) FROM peer_med WHERE peer_n < {MIN_PEER}").fetchone()[0]
total = con.execute("SELECT COUNT(*) FROM peer_med").fetchone()[0]
print(f"Peer groups: {total:,}   with < {MIN_PEER} providers (peer features set to NULL): {small:,}")
zero_mad = {k: con.execute(f"SELECT COUNT(*) FROM peer_mad WHERE mad_{k} = 0").fetchone()[0]
            for k in METRICS}
print("Peer groups with MAD = 0 (z-score undefined -> NULL):", zero_mad)

# ---------------------------------------------------------------- percentile ranks
banner("3. WITHIN-PEER PERCENTILE RANKS")
pr_sel = ", ".join(
    f"percent_rank() OVER (PARTITION BY spec, cms_year ORDER BY m_{k}) AS pr_{k}" for k in RANKED)
con.execute(f"CREATE TABLE ranks AS SELECT npi, cms_year, {pr_sel} FROM base")

# ---------------------------------------------------------------- main feature join
banner("4. FEATURES (z-scores, ratios, trends)")
z_sel = ",\n       ".join(
    f"CASE WHEN p.peer_n >= {MIN_PEER} AND q.mad_{k} > 0 THEN "
    f"GREATEST(LEAST((b.m_{k} - p.med_{k}) / (1.4826 * q.mad_{k}), {CLIP_Z}), -{CLIP_Z}) END AS z_{k}"
    for k in METRICS)
pr_cols = ",\n       ".join(
    f"CASE WHEN p.peer_n >= {MIN_PEER} THEN r.pr_{k} END AS pr_{k}" for k in RANKED)
log_cols = ",\n       ".join(
    f"b.m_{k} AS f_log_{k}" for k in METRICS)

con.execute(f"""
    CREATE TABLE f1 AS
    SELECT b.cms_year, b.npi, b.spec AS specialty_clean, p.peer_n,
           b.y_A, b.y_B, b.y_tier1,
           {log_cols},
           b.tot_mdcr_pymt_amt / NULLIF(b.tot_mdcr_alowd_amt, 0) AS f_paid_to_allowed,
           b.bene_avg_risk_scre AS f_risk_score,
           b.bene_avg_age       AS f_avg_age,
           {z_sel},
           {pr_cols},
           CASE WHEN p.peer_n >= {MIN_PEER}
                THEN LEAST(b.ppb_raw / NULLIF(p.med_ppb_raw, 0), 50) END AS ratio_ppb_to_peer,
           CASE WHEN p.peer_n >= {MIN_PEER}
                THEN LEAST(b.spb_raw / NULLIF(p.med_spb_raw, 0), 50) END AS ratio_spb_to_peer,
           CASE WHEN p.peer_n >= {MIN_PEER}
                THEN p.med_ppb_raw * b.tot_benes END AS expected_pay,
           CASE WHEN p.peer_n >= {MIN_PEER}
                THEN GREATEST(0, b.tot_mdcr_pymt_amt - p.med_ppb_raw * b.tot_benes) END AS excess_payment,
           b.tot_mdcr_pymt_amt
    FROM base b
    JOIN peer_med p USING (spec, cms_year)
    JOIN peer_mad q USING (spec, cms_year)
    JOIN ranks r ON r.npi = b.npi AND r.cms_year = b.cms_year""")

# trends + tenure: LOOK-BACK ONLY (LAG / running count). No look-ahead.
def growth(col):
    return (f"CASE WHEN LAG(b.cms_year) OVER w = b.cms_year - 1 THEN "
            f"GREATEST(LEAST(ln((b.{col} + 1) / (LAG(b.{col}) OVER w + 1)), 5), -5) END")

con.execute(f"""
    CREATE TABLE trend AS
    SELECT b.npi, b.cms_year,
           {growth('tot_mdcr_pymt_amt')} AS g_pay,
           {growth('tot_srvcs')}         AS g_srvcs,
           {growth('tot_benes')}         AS g_benes,
           {growth('ppb_raw')}           AS g_ppb,
           COUNT(*) OVER (PARTITION BY b.npi ORDER BY b.cms_year
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                         AS years_observed_to_date
    FROM base b
    WINDOW w AS (PARTITION BY b.npi ORDER BY b.cms_year)""")

dz_cols = ",\n           ".join(
    f"CASE WHEN LAG(f1.cms_year) OVER w = f1.cms_year - 1 THEN f1.z_{k} - LAG(f1.z_{k}) OVER w END AS dz_{k}"
    for k in DZ)
con.execute(f"""
    CREATE TABLE features AS
    SELECT f1.*, t.g_pay, t.g_srvcs, t.g_benes, t.g_ppb, t.years_observed_to_date,
           {dz_cols}
    FROM f1 JOIN trend t ON t.npi = f1.npi AND t.cms_year = f1.cms_year
    WINDOW w AS (PARTITION BY f1.npi ORDER BY f1.cms_year)""")

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
con.execute(f"COPY features TO {sq(OUT_PARQUET)} (FORMAT PARQUET)")
print(f"Wrote {OUT_PARQUET.resolve()}")

# ---------------------------------------------------------------- dictionary
banner("5. FEATURE DICTIONARY")
rows = []
for k in METRICS:
    rows.append((f"f_log_{k}", f"ln(1+x) of {DESCRIPTIONS[k]}", "model input"))
    rows.append((f"z_{k}", f"robust peer z-score of log {DESCRIPTIONS[k]}; "
                 f"(x-median)/(1.4826*MAD) within specialty x year, clipped +/-{CLIP_Z}", "model input"))
for k in RANKED:
    rows.append((f"pr_{k}", f"percentile rank of {DESCRIPTIONS[k]} within specialty x year", "model input"))
rows += [
    ("f_paid_to_allowed", "Medicare paid / allowed amount", "model input"),
    ("f_risk_score", "average beneficiary risk score (patient complexity)", "model input"),
    ("f_avg_age", "average beneficiary age", "model input"),
    ("ratio_ppb_to_peer", "payment per beneficiary / peer median (cap 50)", "model input"),
    ("ratio_spb_to_peer", "services per beneficiary / peer median (cap 50)", "model input"),
    ("g_pay", "log growth of Medicare payment vs previous consecutive year (cap +/-5)", "model input"),
    ("g_srvcs", "log growth of services vs previous consecutive year", "model input"),
    ("g_benes", "log growth of beneficiaries vs previous consecutive year", "model input"),
    ("g_ppb", "log growth of payment per beneficiary vs previous year", "model input"),
    ("dz_ppb", "change in peer z-score of payment per beneficiary vs previous year", "model input"),
    ("dz_spb", "change in peer z-score of services per beneficiary vs previous year", "model input"),
    ("years_observed_to_date", "years the provider appears in 2016..Y (left-censored at 2016)", "model input"),
    ("expected_pay", "peer median payment per beneficiary x beneficiaries", "FINANCE ONLY (Phase 7)"),
    ("excess_payment", "max(0, payment - expected_pay): proxy for money at risk, NOT proven loss", "FINANCE ONLY (Phase 7)"),
    ("peer_n", "providers in the specialty-year peer group", "diagnostic"),
    ("specialty_clean", "cleaned specialty (peer group id)", "ID - not a model input by default"),
    ("y_A / y_B / y_tier1", "labels", "LABEL - never an input"),
]
DICT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(DICT_CSV, "w", encoding="utf-8") as f:
    f.write("feature,definition,role\n")
    for r in rows:
        f.write(",".join('"' + x.replace('"', "'") + '"' for x in r) + "\n")
model_inputs = [r[0] for r in rows if r[2] == "model input"]
print(f"{len(model_inputs)} model-input features documented -> {DICT_CSV}")

# ---------------------------------------------------------------- sanity report
banner("6. SANITY REPORT")
print("Share of rows with NULL (peer group too small, no prior year, MAD=0):")
for c in model_inputs:
    nr = con.execute(f"SELECT AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END) FROM features").fetchone()[0]
    if nr > 0.001:
        print(f"  {c:26s} {nr:7.2%}")

print("\nMean of key features: positives (y_A=1) vs everyone else:")
cmp_cols = ["z_ppb", "z_spb", "z_ra", "z_pps", "z_cr", "z_codes", "z_benes",
            "pr_ppb", "ratio_ppb_to_peer", "g_pay", "g_ppb", "dz_ppb", "years_observed_to_date"]
print(f"  {'feature':26s} {'positive':>10s} {'others':>10s}")
for c in cmp_cols:
    a, b = con.execute(f"""SELECT AVG(CASE WHEN y_A=1 THEN {c} END),
                                  AVG(CASE WHEN y_A=0 THEN {c} END) FROM features""").fetchone()
    fmt = lambda v: "NULL" if v is None else f"{v:10.3f}"
    print(f"  {c:26s} {fmt(a):>10s} {fmt(b):>10s}")

print("\nLeakage guard - confirm no feature name refers to labels or the future:")
bad = [c for c in model_inputs if c.lower().startswith(("y_", "excl")) or "next" in c.lower()]
print("  suspicious names:", bad or "none")
print("\nReminder: do NOT feed state, zip, year, NPI, labels, or specialty ID to the model "
      "unless the evaluation protocol says so.")
banner("DONE")



con.close()