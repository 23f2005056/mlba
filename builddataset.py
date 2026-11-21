"""
FraudLens - build the labelled provider-year table (end of Phase 2 / start of Phase 3)

Output:  data/processed/provider_year.parquet     (one row per individual provider per year)
Columns: identifiers + raw CMS measures + label columns
Labels : y_A       primary label  (5 fraud-related codes, exact NPI match, 3-yr window)
         y_B       sensitivity    (all exclusion types, exact NPI match)
         y_tier1   sensitivity    (A OR name+state recovery for LEIE rows with no NPI)
         excl_year / excl_type are NOT written - they would leak the label.

Run:  python build_dataset.py        (no pandas needed)
"""
from pathlib import Path

import duckdb

CMS_FOLDER = Path(r"C:\Users\DELL\Downloads\cms\227664-V1")
LEIE_CSV = Path(r"C:\Users\DELL\Downloads\cms\UPDATED (1).csv")
OUT_DIR = Path("data/processed")
YEARS = list(range(2016, 2023))
WINDOW = 3
FRAUD_TYPES = ["1128a1", "1128a3", "1128b1", "1128b6", "1128b7"]
inlist = ",".join(f"'{t}'" for t in FRAUD_TYPES)

NUMERIC = ["Tot_HCPCS_Cds", "Tot_Benes", "Tot_Srvcs", "Tot_Sbmtd_Chrg",
           "Tot_Mdcr_Alowd_Amt", "Tot_Mdcr_Pymt_Amt", "Bene_Avg_Risk_Scre",
           "Bene_Avg_Age"]
CATEG = {"Rndrng_Prvdr_Type": "specialty", "Rndrng_Prvdr_State_Abrvtn": "state",
         "Rndrng_Prvdr_Zip5": "zip5", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "medicare_participating"}


def sq(p):
    return "'" + str(Path(p).resolve()).replace("\\", "/").replace("'", "''") + "'"


def norm(col):
    return (f"regexp_replace(upper(trim(CAST({col} AS VARCHAR))),"
            f"'[^A-Z0-9]','','g')")


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


con = duckdb.connect(":memory:")

# ------------------------------------------------------------ CMS
banner("LOADING CMS (individual providers only)")
first = True
for y in YEARS:
    f = sorted((CMS_FOLDER / str(y)).glob("*.csv"))[0]
    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto({sq(f)}, header=true, all_varchar=true)"
    ).fetchall()}
    miss = [c for c in NUMERIC + list(CATEG) + ["Rndrng_NPI", "Rndrng_Prvdr_Ent_Cd",
            "Rndrng_Prvdr_Last_Org_Name", "Rndrng_Prvdr_First_Name"] if c not in cols]
    if miss:
        print(f"  {y}: columns not found (will be NULL): {miss}")
    num = ", ".join(
        (f"TRY_CAST(replace({c}, ',', '') AS DOUBLE) AS {c.lower()}" if c in cols
         else f"CAST(NULL AS DOUBLE) AS {c.lower()}") for c in NUMERIC)
    cat = ", ".join(
        (f"trim(CAST({c} AS VARCHAR)) AS {a}" if c in cols
         else f"CAST(NULL AS VARCHAR) AS {a}") for c, a in CATEG.items())
    q = f"""
        SELECT {y} AS cms_year,
               trim(CAST(Rndrng_NPI AS VARCHAR)) AS npi,
               {norm('Rndrng_Prvdr_Last_Org_Name')} AS ln,
               {norm('Rndrng_Prvdr_First_Name')}    AS fn,
               {cat}, {num}
        FROM read_csv_auto({sq(f)}, header=true, all_varchar=true)
        WHERE upper(trim(CAST(Rndrng_Prvdr_Ent_Cd AS VARCHAR))) = 'I'"""
    con.execute(("CREATE TABLE cms AS " if first else "INSERT INTO cms ") + q)
    first = False
    print(f"  loaded {y}")
print(f"Provider-years: {con.execute('SELECT COUNT(*) FROM cms').fetchone()[0]:,}")

# ------------------------------------------------------------ LEIE
con.execute(f"""
    CREATE TABLE leie AS
    SELECT CASE WHEN regexp_matches(trim(CAST(NPI AS VARCHAR)), '^[0-9]{{10}}$')
                 AND trim(CAST(NPI AS VARCHAR)) <> '0000000000'
                THEN trim(CAST(NPI AS VARCHAR)) END AS npi,
           lower(trim(CAST(EXCLTYPE AS VARCHAR))) AS excltype,
           try_cast(substr(trim(CAST(EXCLDATE AS VARCHAR)),1,4) AS INTEGER) AS excl_year,
           {norm('LASTNAME')}  AS ln,
           {norm('FIRSTNAME')} AS fn,
           upper(trim(CAST(STATE AS VARCHAR))) AS st
    FROM read_csv_auto({sq(LEIE_CSV)}, header=true, all_varchar=true)""")

# ------------------------------------------------------------ labels
banner("BUILDING LABELS")
win = "l.excl_year > c.cms_year AND l.excl_year <= c.cms_year + " + str(WINDOW)

con.execute(f"""CREATE TABLE y_a AS
    SELECT DISTINCT c.cms_year, c.npi FROM cms c JOIN leie l ON c.npi = l.npi
    WHERE l.excltype IN ({inlist}) AND {win}""")
con.execute(f"""CREATE TABLE y_b AS
    SELECT DISTINCT c.cms_year, c.npi FROM cms c JOIN leie l ON c.npi = l.npi
    WHERE {win}""")

# name+state recovery for LEIE rows WITHOUT an NPI (CMS key must identify ONE npi)
con.execute("""CREATE TABLE key_t1 AS
    SELECT ln, fn, st, MIN(npi) AS npi FROM
        (SELECT ln, fn, state AS st, npi FROM cms)
    WHERE ln <> '' AND fn <> '' AND st IS NOT NULL
    GROUP BY 1,2,3 HAVING COUNT(DISTINCT npi) = 1""")
con.execute(f"""CREATE TABLE y_t1 AS
    SELECT DISTINCT c.cms_year, c.npi
    FROM leie l JOIN key_t1 k ON l.ln=k.ln AND l.fn=k.fn AND l.st=k.st
    JOIN cms c ON c.npi = k.npi
    WHERE l.npi IS NULL AND l.excltype IN ({inlist})
      AND l.ln <> '' AND l.fn <> '' AND {win}""")

con.execute("""
    CREATE TABLE provider_year AS
    SELECT c.* EXCLUDE (ln, fn),
           CASE WHEN a.npi  IS NOT NULL THEN 1 ELSE 0 END AS y_A,
           CASE WHEN b.npi  IS NOT NULL THEN 1 ELSE 0 END AS y_B,
           CASE WHEN a.npi IS NOT NULL OR t.npi IS NOT NULL THEN 1 ELSE 0 END AS y_tier1
    FROM cms c
    LEFT JOIN y_a a ON a.npi = c.npi AND a.cms_year = c.cms_year
    LEFT JOIN y_b b ON b.npi = c.npi AND b.cms_year = c.cms_year
    LEFT JOIN y_t1 t ON t.npi = c.npi AND t.cms_year = c.cms_year""")

# ------------------------------------------------------------ write + report
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "provider_year.parquet"
con.execute(f"COPY provider_year TO {sq(out)} (FORMAT PARQUET)")
print(f"\nWrote {out.resolve()}")

banner("SUMMARY")
print("Rows:", f"{con.execute('SELECT COUNT(*) FROM provider_year').fetchone()[0]:,}")
for lab in ("y_A", "y_B", "y_tier1"):
    p, u = con.execute(
        f"SELECT SUM({lab}), COUNT(DISTINCT CASE WHEN {lab}=1 THEN npi END) FROM provider_year"
    ).fetchone()
    print(f"  {lab:8s} positive provider-years: {int(p):>6,}   unique NPIs: {u:,}")

print("\nDuplicate (npi, year) rows:",
      con.execute("""SELECT COUNT(*) FROM (SELECT 1 FROM provider_year
                     GROUP BY npi, cms_year HAVING COUNT(*) > 1)""").fetchone()[0])

print("\nMissing / zero values in key columns (share of rows):")
for c in ["tot_benes", "tot_srvcs", "tot_sbmtd_chrg", "tot_mdcr_alowd_amt",
          "tot_mdcr_pymt_amt", "bene_avg_risk_scre", "specialty", "state"]:
    n_null, n_zero = con.execute(f"""
        SELECT AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END),
               AVG(CASE WHEN TRY_CAST({c} AS DOUBLE) = 0 THEN 1.0 ELSE 0 END)
        FROM provider_year""").fetchone()
    print(f"  {c:22s} null {n_null:6.2%}   zero {n_zero:6.2%}")

print("\nSpecialties with the most providers (peer-group sizes) - top 10:")
for s, n in con.execute("""SELECT specialty, COUNT(DISTINCT npi) FROM provider_year
                           GROUP BY 1 ORDER BY 2 DESC LIMIT 10""").fetchall():
    print(f"  {str(s):45s} {n:,}")
small = con.execute("""SELECT COUNT(*) FROM (SELECT specialty, cms_year FROM provider_year
                       GROUP BY 1,2 HAVING COUNT(*) < 30)""").fetchone()[0]
print(f"\nSpecialty-year peer groups with fewer than 30 providers: {small:,} "
      "(peer z-scores will be unreliable there)")
print("\nNOTE: y_A / y_B / y_tier1 are LABELS. Never use them, or excl_year, as features.")
banner("DONE")
con.close()