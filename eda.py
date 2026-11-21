"""
FraudLens - Phase 3 EDA (aimed at the positives)

Reads  data/processed/provider_year.parquet   (made by build_dataset.py)
Writes reports/eda/*.csv  and prints the key tables.

Answers:
  1. Is the data sane?  (distributions, impossible values, suppression floor)
  2. Where are positives?  (rate per 10,000 provider-years by specialty/state/year)
  3. Do positives look different from peers?  (percentile rank WITHIN specialty-year)
  4. What do naive baselines already achieve?  (the bar the model must beat)
  5. How big will the test set really be?  (split design numbers)

Run:  python eda.py        (no pandas needed)
"""
from pathlib import Path

import duckdb

PARQUET = Path("data/processed/provider_year.parquet")
OUT = Path("reports/eda")
OUT.mkdir(parents=True, exist_ok=True)
LABEL = "y_A"                       # primary label
TRAIN, VAL, TEST = (2016, 2019), (2020, 2020), (2021, 2022)


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def show(headers, rows, name=None, limit=None):
    rows = [["NULL" if v is None else (f"{v:,.4f}" if isinstance(v, float) else str(v))
             for v in r] for r in rows]
    if name:
        con.execute  # keep linter quiet
        with open(OUT / f"{name}.csv", "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join('"' + x.replace('"', "'") + '"' for x in r) + "\n")
    if limit:
        rows = rows[:limit]
    if not rows:
        print("(no rows)")
        return
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(r[i].ljust(w[i]) for i in range(len(headers))))


con = duckdb.connect(":memory:")
con.execute(f"""
    CREATE TABLE py AS
    SELECT cms_year, npi, specialty, state, {LABEL} AS y,
           tot_benes, tot_srvcs, tot_hcpcs_cds,
           tot_sbmtd_chrg, tot_mdcr_alowd_amt, tot_mdcr_pymt_amt, bene_avg_risk_scre,
           tot_mdcr_pymt_amt / NULLIF(tot_benes, 0)                  AS ppb,
           tot_srvcs         / NULLIF(tot_benes, 0)                  AS spb,
           tot_sbmtd_chrg    / NULLIF(tot_mdcr_alowd_amt, 0)         AS charge_ratio,
           (tot_mdcr_pymt_amt / NULLIF(tot_benes, 0))
               / NULLIF(bene_avg_risk_scre, 0)                       AS risk_adj_ppb
    FROM read_parquet('{PARQUET.as_posix()}')""")
n = con.execute("SELECT COUNT(*) FROM py").fetchone()[0]
pos = con.execute("SELECT SUM(y) FROM py").fetchone()[0]
print(f"Rows: {n:,}   positives ({LABEL}): {int(pos):,}")

# ------------------------------------------------------------------ 1. sanity
banner("1. DATA SANITY")
print("Per-year row counts and totals:")
show(["year", "providers", "positives", "total_medicare_paid_$B"],
     con.execute("""SELECT cms_year, COUNT(*), SUM(y), ROUND(SUM(tot_mdcr_pymt_amt)/1e9,2)
                    FROM py GROUP BY 1 ORDER BY 1""").fetchall(), "year_summary")

print("\nDistribution of the core metrics (all providers):")
rows = []
for m in ["tot_benes", "tot_mdcr_pymt_amt", "ppb", "spb", "charge_ratio",
          "risk_adj_ppb", "bene_avg_risk_scre", "tot_hcpcs_cds"]:
    r = con.execute(f"""SELECT MIN({m}), quantile_cont({m},0.01), quantile_cont({m},0.5),
                               quantile_cont({m},0.99), MAX({m}) FROM py""").fetchone()
    rows.append([m] + [round(x, 2) if x is not None else None for x in r])
show(["metric", "min", "p1", "median", "p99", "max"], rows, "metric_distribution")
print("\nIf min(tot_benes) is ~11, CMS suppresses providers with fewer beneficiaries "
      "(small practices are invisible to us) - note this in the data card.")

print("\nPossible data-quality flags:")
for label, cond in [
    ("allowed > submitted charges (charge_ratio < 1)", "charge_ratio < 1"),
    ("paid > allowed", "tot_mdcr_pymt_amt > tot_mdcr_alowd_amt * 1.0001"),
    ("services per beneficiary > 500", "spb > 500"),
    ("payment per beneficiary > $50,000", "ppb > 50000"),
    ("risk score <= 0", "bene_avg_risk_scre <= 0"),
]:
    c = con.execute(f"SELECT COUNT(*) FROM py WHERE {cond}").fetchone()[0]
    print(f"  {label:50s} {c:>9,}")

# ------------------------------------------------------------------ 2. where are positives
banner("2. WHERE ARE THE POSITIVES? (rate per 10,000 provider-years)")
print("By specialty (>= 20,000 provider-years and >= 3 positive NPIs), highest rate first:")
show(["specialty", "provider_years", "positives", "pos_npis", "rate_per_10k"],
     con.execute("""
        SELECT specialty, COUNT(*) AS n, SUM(y) AS pos,
               COUNT(DISTINCT CASE WHEN y=1 THEN npi END) AS pos_npis,
               ROUND(SUM(y)*10000.0/COUNT(*), 2) AS rate
        FROM py GROUP BY 1
        HAVING COUNT(*) >= 20000 AND COUNT(DISTINCT CASE WHEN y=1 THEN npi END) >= 3
        ORDER BY rate DESC""").fetchall(), "rate_by_specialty", limit=20)

print("\nBy state (>= 20,000 provider-years), highest rate first - top 12:")
show(["state", "provider_years", "positives", "rate_per_10k"],
     con.execute("""
        SELECT state, COUNT(*), SUM(y), ROUND(SUM(y)*10000.0/COUNT(*), 2) AS rate
        FROM py GROUP BY 1 HAVING COUNT(*) >= 20000 ORDER BY rate DESC""").fetchall(),
     "rate_by_state", limit=12)

print("\nBy billing year:")
show(["year", "positives", "rate_per_10k"],
     con.execute("""SELECT cms_year, SUM(y), ROUND(SUM(y)*10000.0/COUNT(*),2)
                    FROM py GROUP BY 1 ORDER BY 1""").fetchall(), "rate_by_year")
print("\nUse these rates (not raw counts) to decide peer groups and what to report "
      "in the model card's fairness-by-specialty section.")

# ------------------------------------------------------------------ 3. positives vs peers
banner("3. DO POSITIVES SIT HIGHER THAN THEIR PEERS?")
print("Percentile rank WITHIN specialty and year (0 = lowest, 1 = highest).")
print("Non-positives average ~0.50 by construction; a value well above 0.50 = signal.\n")
con.execute("""
    CREATE TABLE ranks AS
    SELECT cms_year, npi, y, tot_mdcr_pymt_amt,
      percent_rank() OVER (PARTITION BY specialty, cms_year ORDER BY ppb)          AS r_ppb,
      percent_rank() OVER (PARTITION BY specialty, cms_year ORDER BY spb)          AS r_spb,
      percent_rank() OVER (PARTITION BY specialty, cms_year ORDER BY charge_ratio) AS r_charge,
      percent_rank() OVER (PARTITION BY specialty, cms_year ORDER BY risk_adj_ppb) AS r_riskadj,
      percent_rank() OVER (PARTITION BY specialty, cms_year ORDER BY tot_hcpcs_cds) AS r_codes,
      percent_rank() OVER (PARTITION BY specialty, cms_year ORDER BY tot_mdcr_pymt_amt) AS r_total_in_spec,
      percent_rank() OVER (PARTITION BY cms_year ORDER BY tot_mdcr_pymt_amt)       AS r_total_global
    FROM py""")
metrics = ["r_ppb", "r_spb", "r_charge", "r_riskadj", "r_codes",
           "r_total_in_spec", "r_total_global"]
avg = ", ".join(f"ROUND(AVG({m}),3)" for m in metrics)
rows = con.execute(f"""
    SELECT CASE WHEN y=1 THEN 'positive' ELSE 'not observed excluded' END, COUNT(*), {avg}
    FROM ranks GROUP BY y ORDER BY y DESC""").fetchall()
show(["group", "n"] + metrics, rows, "positives_vs_peers")

# ------------------------------------------------------------------ 4. baselines
banner("4. NAIVE BASELINES - the bar the model has to beat")
print("Flag the top 1% (and 5%) by each rule; how many positives fall inside?")
print(f"Random ranking would capture ~1% (5%) of positives.\n")
rows = []
for m, label in [("r_total_global", "total Medicare payment (naive investigator)"),
                 ("r_ppb", "payment per beneficiary vs specialty peers"),
                 ("r_spb", "services per beneficiary vs peers"),
                 ("r_charge", "charge ratio vs peers"),
                 ("r_riskadj", "risk-adjusted payment vs peers"),
                 ("r_codes", "code breadth vs peers")]:
    r = con.execute(f"""
        SELECT
          ROUND(SUM(CASE WHEN y=1 AND {m} >= 0.99 THEN 1 ELSE 0 END)*1.0/SUM(y), 4),
          ROUND(SUM(CASE WHEN y=1 AND {m} >= 0.95 THEN 1 ELSE 0 END)*1.0/SUM(y), 4),
          ROUND(SUM(CASE WHEN y=1 AND {m} >= 0.99 THEN 1 ELSE 0 END)*1.0
                / NULLIF(SUM(CASE WHEN {m} >= 0.99 THEN 1 ELSE 0 END),0) * 10000, 2)
        FROM ranks""").fetchone()
    rows.append([label] + list(r))
show(["rule", "recall@top1%", "recall@top5%", "positives_per_10k_flagged@1%"],
     rows, "naive_baselines")
print("\nBase rate is ~1.2 positives per 10,000 provider-years. If a rule flags "
      "e.g. 4 per 10,000, that is a ~3x lift. These are the numbers your "
      "LightGBM model (and the PR-AUC lift) must beat.")

# ------------------------------------------------------------------ 5. split design
banner("5. SPLIT DESIGN - how large will validation/test really be?")
def pos_where(a, b):
    return con.execute(f"""SELECT COUNT(*), COUNT(DISTINCT npi) FROM py
                           WHERE y=1 AND cms_year BETWEEN {a} AND {b}""").fetchone()
for name, (a, b) in {"train 2016-19": TRAIN, "val 2020": VAL, "test 2021-22": TEST}.items():
    c, u = pos_where(a, b)
    print(f"  {name:14s} positive provider-years {c:>4,}   unique NPIs {u:>4,}")

c = con.execute(f"""
    SELECT COUNT(DISTINCT npi) FROM py WHERE y=1 AND cms_year BETWEEN {TEST[0]} AND {TEST[1]}
      AND npi NOT IN (SELECT npi FROM py WHERE y=1 AND cms_year BETWEEN {TRAIN[0]} AND {TRAIN[1]})
""").fetchone()[0]
print(f"\n  Test-year positive NPIs with NO positive row in training years: {c:,}")

print("\nNPI-disjoint split by hashing the NPI (30% of providers held out for test):")
r = con.execute(f"""
    SELECT SUM(CASE WHEN hash(npi) % 10 < 3 AND cms_year BETWEEN {TEST[0]} AND {TEST[1]} THEN y ELSE 0 END),
           COUNT(DISTINCT CASE WHEN hash(npi) % 10 < 3 AND cms_year BETWEEN {TEST[0]} AND {TEST[1]} AND y=1 THEN npi END),
           SUM(CASE WHEN hash(npi) % 10 >= 3 AND cms_year BETWEEN {TRAIN[0]} AND {TRAIN[1]} THEN y ELSE 0 END),
           COUNT(DISTINCT CASE WHEN hash(npi) % 10 >= 3 AND cms_year BETWEEN {TRAIN[0]} AND {TRAIN[1]} AND y=1 THEN npi END)
    FROM py""").fetchone()
print(f"  TEST  (held-out NPIs, 2021-22): {int(r[0]):,} positive provider-years, {r[1]:,} unique NPIs")
print(f"  TRAIN (other NPIs, 2016-19):    {int(r[2]):,} positive provider-years, {r[3]:,} unique NPIs")
print("\nIf the test set has only ~50-100 positive NPIs, report bootstrap confidence "
      "intervals (resampling by NPI) and treat model comparisons as indicative. "
      "A complementary option: NPI-grouped 5-fold cross-validation over all years.")

banner("EDA COMPLETE  -  CSVs in reports/eda/")
con.close()