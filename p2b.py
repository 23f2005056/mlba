"""
FraudLens - Phase 2b follow-up

  1. Corrected LEIE NPI coverage (the placeholder in this file is '0', not
     '0000000000', so earlier "with NPI" counts were inflated)
  2. What the Scenario-A label is actually made of (by exclusion code)
  3. Why only 68% of NPI matches agree on name+state - inspect mismatches
  4. How many MORE positives can be recovered from LEIE rows with no NPI
     (name+state, and name+state+ZIP)

Run:  python phase2b_followup.py      (no pandas needed)
"""
from pathlib import Path

import duckdb

CMS_FOLDER = Path(r"C:\Users\DELL\Downloads\cms\227664-V1")
LEIE_CSV = Path(r"C:\Users\DELL\Downloads\cms\UPDATED (1).csv")
YEARS = list(range(2016, 2023))
WINDOW = 3
FRAUD_TYPES = ["1128a1", "1128a3", "1128b1", "1128b6", "1128b7"]
TRAIN, VAL, TEST = (2016, 2019), (2020, 2020), (2021, 2022)
inlist = ",".join(f"'{t}'" for t in FRAUD_TYPES)


def sq(p):
    return "'" + str(Path(p).resolve()).replace("\\", "/").replace("'", "''") + "'"


def norm(col):
    return (f"regexp_replace(upper(trim(CAST({col} AS VARCHAR))),"
            f"'[^A-Z0-9]','','g')")


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def table(headers, rows):
    if not rows:
        print("(no rows)")
        return
    rows = [["NULL" if v is None else str(v) for v in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(r[i].ljust(w[i]) for i in range(len(headers))))


con = duckdb.connect(":memory:")

# ---------------------------------------------------------------- load
banner("LOADING")
first = True
has_zip = True
for y in YEARS:
    f = sorted((CMS_FOLDER / str(y)).glob("*.csv"))[0]
    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto({sq(f)}, header=true, all_varchar=true)"
    ).fetchall()}
    zip_expr = ("substr(trim(CAST(Rndrng_Prvdr_Zip5 AS VARCHAR)),1,5)"
                if "Rndrng_Prvdr_Zip5" in cols else "NULL")
    has_zip = has_zip and "Rndrng_Prvdr_Zip5" in cols
    q = f"""
        SELECT {y} AS cms_year,
               trim(CAST(Rndrng_NPI AS VARCHAR)) AS npi,
               upper(trim(CAST(Rndrng_Prvdr_Ent_Cd AS VARCHAR))) AS ent,
               {norm('Rndrng_Prvdr_Last_Org_Name')} AS ln,
               {norm('Rndrng_Prvdr_First_Name')} AS fn,
               upper(trim(CAST(Rndrng_Prvdr_State_Abrvtn AS VARCHAR))) AS st,
               {zip_expr} AS zip5,
               trim(CAST(Rndrng_Prvdr_Last_Org_Name AS VARCHAR)) AS ln_raw,
               trim(CAST(Rndrng_Prvdr_First_Name AS VARCHAR)) AS fn_raw
        FROM read_csv_auto({sq(f)}, header=true, all_varchar=true)
        WHERE upper(trim(CAST(Rndrng_Prvdr_Ent_Cd AS VARCHAR))) = 'I'"""
    con.execute(("CREATE TABLE cms AS " if first else "INSERT INTO cms ") + q)
    first = False
print(f"CMS individual provider-years: "
      f"{con.execute('SELECT COUNT(*) FROM cms').fetchone()[0]:,}")
print(f"ZIP column available in CMS files: {has_zip}")

leie_cols = {r[0] for r in con.execute(
    f"DESCRIBE SELECT * FROM read_csv_auto({sq(LEIE_CSV)}, header=true, all_varchar=true)"
).fetchall()}
con.execute(f"""
    CREATE TABLE leie AS
    SELECT trim(CAST(NPI AS VARCHAR)) AS npi_raw,
           CASE WHEN regexp_matches(trim(CAST(NPI AS VARCHAR)), '^[0-9]{{10}}$')
                 AND trim(CAST(NPI AS VARCHAR)) <> '0000000000'
                THEN trim(CAST(NPI AS VARCHAR)) END AS npi,   -- valid NPI or NULL
           lower(trim(CAST(EXCLTYPE AS VARCHAR))) AS excltype,
           try_cast(substr(trim(CAST(EXCLDATE AS VARCHAR)),1,4) AS INTEGER) AS excl_year,
           {norm('LASTNAME')}  AS ln,
           {norm('FIRSTNAME')} AS fn,
           upper(trim(CAST(STATE AS VARCHAR))) AS st,
           substr(trim(CAST(ZIP AS VARCHAR)),1,5) AS zip5,
           trim(CAST(LASTNAME AS VARCHAR))  AS ln_raw,
           trim(CAST(FIRSTNAME AS VARCHAR)) AS fn_raw,
           trim(CAST(CITY AS VARCHAR)) AS city
    FROM read_csv_auto({sq(LEIE_CSV)}, header=true, all_varchar=true)""")
tot = con.execute("SELECT COUNT(*) FROM leie").fetchone()[0]
valid = con.execute("SELECT COUNT(*) FROM leie WHERE npi IS NOT NULL").fetchone()[0]
print(f"LEIE rows: {tot:,}")

# ---------------------------------------------------------------- 1
banner("1. CORRECTED LEIE NPI COVERAGE")
print(f"Rows with a VALID 10-digit NPI: {valid:,} of {tot:,} ({valid / tot:.1%})")
print("\nPer exclusion year (valid NPI only) - redo of the survivorship check:")
table(["excl_year", "all_rows", "valid_npi", "selected_types_valid_npi"],
      con.execute(f"""
        SELECT excl_year, COUNT(*),
               SUM(CASE WHEN npi IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN npi IS NOT NULL AND excltype IN ({inlist}) THEN 1 ELSE 0 END)
        FROM leie WHERE excl_year BETWEEN 2010 AND 2025
        GROUP BY 1 ORDER BY 1""").fetchall())
print("\n=> Most LEIE rows have NO NPI. Excluded providers without an NPI cannot be "
      "labelled by NPI match, so some 'unlabelled' CMS providers are really positives "
      "(this is the positive-unlabelled problem).")

# ---------------------------------------------------------------- 2
banner("2. WHAT SCENARIO A IS MADE OF")
con.execute(f"""
    CREATE TABLE posA AS
    SELECT DISTINCT c.cms_year, c.npi, l.excltype, l.excl_year
    FROM cms c JOIN leie l ON c.npi = l.npi
    WHERE l.excltype IN ({inlist})
      AND l.excl_year > c.cms_year AND l.excl_year <= c.cms_year + {WINDOW}""")
table(["excltype", "provider_years", "unique_npis"],
      con.execute("""SELECT excltype, COUNT(*), COUNT(DISTINCT npi)
                     FROM posA GROUP BY 1 ORDER BY 2 DESC""").fetchall())
print("\nIf 1128a1 dominates, the target is mostly 'program-related crimes', "
      "not only billing fraud. Name it honestly in the data card.")

# ---------------------------------------------------------------- 3
banner("3. WHY ONLY ~68% OF NPI MATCHES AGREE ON NAME + STATE")
con.execute("""
    CREATE TABLE a_check AS
    SELECT DISTINCT p.npi, p.cms_year, l.ln AS l_ln, l.fn AS l_fn, l.st AS l_st,
           l.ln_raw AS l_ln_raw, l.fn_raw AS l_fn_raw,
           c.ln AS c_ln, c.fn AS c_fn, c.st AS c_st,
           c.ln_raw AS c_ln_raw, c.fn_raw AS c_fn_raw
    FROM posA p
    JOIN leie l ON l.npi = p.npi AND l.excltype IN ('1128a1','1128a3','1128b1','1128b6','1128b7')
    JOIN cms c ON c.npi = p.npi AND c.cms_year = p.cms_year""")
con.execute("""
    CREATE TABLE a_class AS
    SELECT *,
      CASE WHEN l_ln = c_ln AND l_fn = c_fn AND l_st = c_st THEN '1 full agree'
           WHEN l_ln = c_ln AND l_fn = c_fn                  THEN '2 same name, different state'
           WHEN l_ln = c_ln AND (l_fn LIKE c_fn || '%' OR c_fn LIKE l_fn || '%')
                                                              THEN '3 same last, first name prefix/initial'
           WHEN l_ln = c_ln                                   THEN '4 same last name, different first'
           ELSE                                                    '5 last name differs' END AS cls
    FROM a_check""")
table(["mismatch class", "rows"],
      con.execute("SELECT cls, COUNT(*) FROM a_class GROUP BY 1 ORDER BY 1").fetchall())
print("\nClasses 1-3 are almost certainly the same person (name format, moved state). "
      "Class 5 (last name differs) may be a name change OR a wrong NPI match.")
print("\n25 examples where last name differs (class 5) - eyeball these:")
table(["LEIE last", "LEIE first", "LEIE st", "CMS last", "CMS first", "CMS st"],
      con.execute("""SELECT DISTINCT l_ln_raw, l_fn_raw, l_st, c_ln_raw, c_fn_raw, c_st
                     FROM a_class WHERE cls LIKE '5%' LIMIT 25""").fetchall())
print("\n15 examples where last name matches but first differs (class 3/4):")
table(["LEIE last", "LEIE first", "LEIE st", "CMS last", "CMS first", "CMS st"],
      con.execute("""SELECT DISTINCT l_ln_raw, l_fn_raw, l_st, c_ln_raw, c_fn_raw, c_st
                     FROM a_class WHERE cls LIKE '3%' OR cls LIKE '4%' LIMIT 15""").fetchall())

# ---------------------------------------------------------------- 4
banner("4. RECOVERING POSITIVES FROM LEIE ROWS WITH NO NPI")
con.execute(f"""
    CREATE TABLE leie_nonpi AS
    SELECT * FROM leie
    WHERE npi IS NULL AND excltype IN ({inlist})
      AND ln <> '' AND fn <> '' AND st <> '' AND excl_year IS NOT NULL""")
n_nonpi = con.execute("SELECT COUNT(*) FROM leie_nonpi").fetchone()[0]
print(f"LEIE rows with selected fraud types, a person's name, state, and NO valid NPI: {n_nonpi:,}")

# CMS identity keys that point to exactly ONE NPI (avoids common-name ambiguity)
con.execute("""
    CREATE TABLE key_t1 AS
    SELECT ln, fn, st, MIN(npi) AS npi FROM cms
    GROUP BY 1,2,3 HAVING COUNT(DISTINCT npi) = 1""")
tiers = {"tier1 name+state": ("key_t1", "l.ln=k.ln AND l.fn=k.fn AND l.st=k.st")}
if has_zip:
    con.execute("""
        CREATE TABLE key_t2 AS
        SELECT ln, fn, st, zip5, MIN(npi) AS npi FROM cms WHERE zip5 IS NOT NULL
        GROUP BY 1,2,3,4 HAVING COUNT(DISTINCT npi) = 1""")
    tiers["tier2 name+state+ZIP"] = (
        "key_t2", "l.ln=k.ln AND l.fn=k.fn AND l.st=k.st AND l.zip5=k.zip5")

def split_counts(tbl):
    g = lambda a, b: con.execute(
        f"SELECT COUNT(*) FROM {tbl} WHERE cms_year BETWEEN {a} AND {b}").fetchone()[0]
    return g(*TRAIN), g(*VAL), g(*TEST)

summary = []
for name, (key, cond) in tiers.items():
    t = "rec_" + key
    con.execute(f"""
        CREATE OR REPLACE TABLE {t} AS
        SELECT DISTINCT c.cms_year, c.npi
        FROM leie_nonpi l JOIN {key} k ON {cond}
        JOIN cms c ON c.npi = k.npi
        WHERE l.excl_year > c.cms_year AND l.excl_year <= c.cms_year + {WINDOW}""")
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    u = con.execute(f"SELECT COUNT(DISTINCT npi) FROM {t}").fetchone()[0]
    tr, va, te = split_counts(t)
    summary.append((name, n, u, tr, va, te))
table(["recovery method", "extra provider-years", "unique NPIs", "train", "val", "test"], summary)

base_py = con.execute("SELECT COUNT(*) FROM posA").fetchone()[0]
base_u = con.execute("SELECT COUNT(DISTINCT npi) FROM posA").fetchone()[0]
print(f"\nBaseline Scenario A: {base_py:,} provider-years, {base_u:,} unique NPIs.")

if has_zip:
    print("\n20 sample tier-2 matches to hand-check (LEIE vs CMS):")
    table(["LEIE last", "LEIE first", "LEIE city", "LEIE zip", "CMS last", "CMS first", "CMS zip"],
          con.execute("""
            SELECT DISTINCT l.ln_raw, l.fn_raw, l.city, l.zip5, c.ln_raw, c.fn_raw, c.zip5
            FROM leie_nonpi l JOIN key_t2 k
              ON l.ln=k.ln AND l.fn=k.fn AND l.st=k.st AND l.zip5=k.zip5
            JOIN cms c ON c.npi = k.npi AND c.cms_year = 2019
            LIMIT 20""").fetchall())

print("""
How to read section 4:
  * Tier 2 (name+state+ZIP) is the safer recovery; tier 1 adds more but with more
    false matches (common names).
  * If tier 2 adds a few hundred provider-years with real test counts, use
    A + tier-2 as the primary label ("Scenario E") and keep A as a sensitivity check.
""")
banner("DONE")
con.close()