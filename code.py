"""
FraudLens - Phase 2 label feasibility audit (go / no-go)  -  CORRECTED VERSION

Fixes vs. previous version
  1. Name normalisation: upper() now runs BEFORE the regex that strips
     characters. Previously mixed-case CMS names (2020-2022) were destroyed,
     which is why scenarios C and D had 0 validation/test positives.
  2. Adds unique positive NPIs per scenario (the real sample size).
  3. Adds positives per billing year (survivorship check).
  4. Adds LEIE exclusions per year (survivorship check on the label source).
  5. Adds "name agreement": how many NPI matches (A) are confirmed by name+state.

Run:  python label_feasibility_fixed.py
"""
from pathlib import Path
import csv

import duckdb


# ============================================================
# 1. CONFIGURATION
# ============================================================

CMS_FOLDER = Path(r"C:\Users\DELL\Downloads\cms\227664-V1")
LEIE_CSV = Path(r"C:\Users\DELL\Downloads\cms\UPDATED (1).csv")

OUTPUT_FOLDER = Path("reports")
OUTPUT_FILE = OUTPUT_FOLDER / "label_feasibility.csv"

YEARS = list(range(2016, 2023))
WINDOW = 3  # positive if exclusion falls within the next 3 years

# Verify these against OIG's exclusion-authorities table.
FRAUD_TYPES = ["1128a1", "1128a3", "1128b1", "1128b6", "1128b7"]

TRAIN_YEARS = (2016, 2019)
VALIDATION_YEARS = (2020, 2020)
TEST_YEARS = (2021, 2022)


# ============================================================
# 2. HELPERS
# ============================================================

def sql_path(path):
    return str(path.resolve()).replace("\\", "/")


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def norm_name(col):
    """Uppercase FIRST, then keep only A-Z and 0-9 (the bug fix)."""
    return (
        f"regexp_replace(upper(trim(CAST({col} AS VARCHAR))), "
        f"'[^A-Z0-9]', '', 'g')"
    )


def count_provider_years(connection, table, start_year, end_year):
    return connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE cms_year BETWEEN ? AND ?",
        [start_year, end_year],
    ).fetchone()[0]


def banner(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

banner("FRAUDLENS PHASE 2 - LABEL FEASIBILITY AUDIT (CORRECTED)")

if not CMS_FOLDER.exists():
    raise FileNotFoundError(f"CMS folder not found: {CMS_FOLDER}")
if not LEIE_CSV.exists():
    raise FileNotFoundError(f"LEIE file not found: {LEIE_CSV}")

cms_files = []
for year in YEARS:
    year_folder = CMS_FOLDER / str(year)
    if not year_folder.exists():
        print(f"WARNING: Folder not found for {year}: {year_folder}")
        continue
    csvs = sorted(year_folder.glob("*.csv"))
    if not csvs:
        print(f"WARNING: No CSV file found for {year}")
        continue
    for f in csvs:
        cms_files.append((year, f))

if not cms_files:
    raise FileNotFoundError(f"No CMS CSV files found under: {CMS_FOLDER}")

print(f"\nFound {len(cms_files)} CMS CSV file(s):")
for year, f in cms_files:
    print(f"  {year}: {f.name}")
print(f"\nFound LEIE file: {LEIE_CSV.name}")


# ============================================================
# 4. DUCKDB
# ============================================================

con = duckdb.connect(database=":memory:")


# ============================================================
# 5. LOAD CMS PROVIDER DATA
# ============================================================

banner("LOADING CMS PROVIDER DATA")

CMS_COLUMNS = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "Rndrng_Prvdr_Last_Org_Name",
    "Rndrng_Prvdr_First_Name",
    "Rndrng_Prvdr_Ent_Cd",
]

cms_tables = []

for year, file in cms_files:
    print(f"\nLoading CMS data for {year}...")
    table = f"cms_{year}"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table}_raw AS
        SELECT * FROM read_csv_auto(
            {sql_string(sql_path(file))}, header = true, all_varchar = true)
        """
    )

    actual = {r[0] for r in con.execute(f"DESCRIBE {table}_raw").fetchall()}
    missing = [c for c in CMS_COLUMNS if c not in actual]
    if missing:
        raise ValueError(
            f"Missing CMS columns in {file.name}: {missing}\n"
            f"Available: {sorted(actual)}"
        )

    # Individuals only (Ent_Cd = 'I'); NPI digits only.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT DISTINCT
            {year} AS cms_year,
            regexp_replace(trim(CAST(Rndrng_NPI AS VARCHAR)), '[^0-9]', '', 'g') AS npi,
            {norm_name('Rndrng_Prvdr_Last_Org_Name')}  AS last_name,
            {norm_name('Rndrng_Prvdr_First_Name')}     AS first_name,
            upper(trim(CAST(Rndrng_Prvdr_State_Abrvtn AS VARCHAR))) AS state,
            trim(CAST(Rndrng_Prvdr_Type AS VARCHAR))   AS provider_type
        FROM {table}_raw
        WHERE upper(trim(CAST(Rndrng_Prvdr_Ent_Cd AS VARCHAR))) = 'I'
          AND Rndrng_NPI IS NOT NULL
          AND trim(CAST(Rndrng_NPI AS VARCHAR)) <> ''
        """
    )

    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    named = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE last_name <> '' AND first_name <> ''"
    ).fetchone()[0]
    print(f"  Loaded {n:,} individual provider records ({named:,} with usable names).")
    cms_tables.append(table)

print("\nCombining CMS data across years...")
union_query = "\nUNION ALL\n".join(f"SELECT * FROM {t}" for t in cms_tables)
con.execute(
    f"CREATE OR REPLACE TEMP TABLE cms_all AS SELECT DISTINCT * FROM ({union_query})"
)
total_cms = con.execute("SELECT COUNT(*) FROM cms_all").fetchone()[0]
print(f"Total CMS provider-year records (individuals): {total_cms:,}")


# ============================================================
# 6. LOAD OIG LEIE DATA
# ============================================================

banner("LOADING OIG LEIE DATA")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE leie_raw AS
    SELECT * FROM read_csv_auto(
        {sql_string(sql_path(LEIE_CSV))}, header = true, all_varchar = true)
    """
)

leie_cols = {r[0] for r in con.execute("DESCRIBE leie_raw").fetchall()}
required = ["NPI", "EXCLTYPE", "EXCLDATE", "LASTNAME", "FIRSTNAME", "STATE"]
missing = [c for c in required if c not in leie_cols]
if missing:
    raise ValueError(f"Missing LEIE columns: {missing}\nAvailable: {sorted(leie_cols)}")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE leie_clean AS
    SELECT DISTINCT
        regexp_replace(trim(CAST(NPI AS VARCHAR)), '[^0-9]', '', 'g') AS npi,
        lower(trim(CAST(EXCLTYPE AS VARCHAR))) AS excltype,
        try_cast(substr(regexp_replace(trim(CAST(EXCLDATE AS VARCHAR)),
                 '[^0-9]', '', 'g'), 1, 4) AS INTEGER) AS excl_year,
        {norm_name('LASTNAME')}  AS last_name,
        {norm_name('FIRSTNAME')} AS first_name,
        upper(trim(CAST(STATE AS VARCHAR))) AS state
    FROM leie_raw
    WHERE NPI IS NOT NULL
      AND trim(CAST(NPI AS VARCHAR)) <> ''
      AND regexp_replace(trim(CAST(NPI AS VARCHAR)), '[^0-9]', '', 'g')
          NOT IN ('', '0000000000')
    """
)

total_leie_rows = con.execute("SELECT COUNT(*) FROM leie_raw").fetchone()[0]
leie_count = con.execute("SELECT COUNT(*) FROM leie_clean").fetchone()[0]
print(f"LEIE rows in file:               {total_leie_rows:,}")
print(f"LEIE rows with a usable NPI:     {leie_count:,} "
      f"({leie_count / total_leie_rows:.1%})")

fraud_type_sql = ", ".join(sql_string(c.lower()) for c in FRAUD_TYPES)

print("\nSURVIVORSHIP CHECK - LEIE exclusions per year (usable NPI).")
print("A steep drop in older years = reinstated providers missing from file.")
result = con.execute(
    f"""
    SELECT excl_year,
           COUNT(*) AS rows_with_npi,
           SUM(CASE WHEN excltype IN ({fraud_type_sql}) THEN 1 ELSE 0 END)
               AS fraud_types
    FROM leie_clean
    WHERE excl_year BETWEEN 2012 AND 2025
    GROUP BY 1
    ORDER BY 1
    """
).fetchall()

print(f"{'Year':<12}{'Rows with NPI':<20}{'Selected types':<20}")
print("-" * 52)

for year, rows_with_npi, fraud_types in result:
    print(f"{year:<12}{rows_with_npi:<20}{fraud_types:<20}")


# ============================================================
# 7. BASELINE COUNTS
# ============================================================

banner("PREPARING BASELINE COUNTS")

total_train = count_provider_years(con, "cms_all", *TRAIN_YEARS)
total_validation = count_provider_years(con, "cms_all", *VALIDATION_YEARS)
total_test = count_provider_years(con, "cms_all", *TEST_YEARS)

print(f"Training provider-years:   {total_train:,}")
print(f"Validation provider-years: {total_validation:,}")
print(f"Test provider-years:       {total_test:,}")


# ============================================================
# 8. SCENARIOS
# ============================================================
# A: selected (fraud) exclusion types, NPI match only
# B: all exclusion types,              NPI match only
# C: selected types, NPI match AND name+state agree (verification)
# D: all types,      NPI match AND name+state agree (verification)

SCENARIOS = [
    {"name": "A_Selected_Types_NPI_Only", "selected_types": True, "name_state": False},
    {"name": "B_All_Types_NPI_Only", "selected_types": False, "name_state": False},
    {"name": "C_Selected_Types_NPI_Name_State", "selected_types": True, "name_state": True},
    {"name": "D_All_Types_NPI_Name_State", "selected_types": False, "name_state": True},
]


# ============================================================
# 9. RUN SCENARIOS
# ============================================================

banner("RUNNING FOUR MATCHING SCENARIOS")

rows = []
per_year = {}

for sc in SCENARIOS:
    name = sc["name"]
    print(f"\nRunning scenario: {name}")

    type_condition = (
        f"AND l.excltype IN ({fraud_type_sql})" if sc["selected_types"] else ""
    )
    identity_condition = (
        """
        AND c.last_name <> '' AND c.first_name <> '' AND c.state <> ''
        AND l.last_name <> '' AND l.first_name <> '' AND l.state <> ''
        AND c.last_name = l.last_name
        AND c.first_name = l.first_name
        AND c.state = l.state
        """
        if sc["name_state"]
        else ""
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE scenario_matches AS
        SELECT DISTINCT c.cms_year, c.npi
        FROM cms_all c
        INNER JOIN leie_clean l ON c.npi = l.npi
        WHERE c.npi <> ''
          AND l.excl_year IS NOT NULL
          AND l.excl_year > c.cms_year
          AND l.excl_year <= c.cms_year + {WINDOW}
          {type_condition}
          {identity_condition}
        """
    )

    pos_train = count_provider_years(con, "scenario_matches", *TRAIN_YEARS)
    pos_val = count_provider_years(con, "scenario_matches", *VALIDATION_YEARS)
    pos_test = count_provider_years(con, "scenario_matches", *TEST_YEARS)
    pos_total = con.execute("SELECT COUNT(*) FROM scenario_matches").fetchone()[0]
    unique_npis = con.execute(
        "SELECT COUNT(DISTINCT npi) FROM scenario_matches"
    ).fetchone()[0]
    test_unique_npis = con.execute(
        "SELECT COUNT(DISTINCT npi) FROM scenario_matches "
        "WHERE cms_year BETWEEN ? AND ?", list(TEST_YEARS)
    ).fetchone()[0]

    per_year[name] = con.execute(
        "SELECT cms_year, COUNT(*) AS positives FROM scenario_matches "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()

    rows.append({
        "scenario": name,
        "positive_provider_years": pos_total,
        "unique_positive_npis": unique_npis,
        "pos_train_2016-2019": pos_train,
        "pos_validation_2020": pos_val,
        "pos_test_2021-2022": pos_test,
        "unique_npis_in_test_years": test_unique_npis,
        "total_train_provider_years": total_train,
        "total_validation_provider_years": total_validation,
        "total_test_provider_years": total_test,
        "train_positive_pct": round(pos_train / total_train * 100, 4) if total_train else 0,
        "validation_positive_pct": round(pos_val / total_validation * 100, 4) if total_validation else 0,
        "test_positive_pct": round(pos_test / total_test * 100, 4) if total_test else 0,
    })

    print(f"  Total positive provider-years: {pos_total:,}")
    print(f"  Unique positive NPIs:          {unique_npis:,}")
    print(f"  Training / Validation / Test:  "
          f"{pos_train:,} / {pos_val:,} / {pos_test:,}")


# ============================================================
# 10. COMPARISON TABLE
# ============================================================

banner("FOUR SCENARIO COMPARISON")

for r in rows:
    print(f"\n{r['scenario']}")
    print(f"  Positive provider-years: {r['positive_provider_years']:,}")
    print(f"  Unique positive NPIs:    {r['unique_positive_npis']:,}")
    print(f"  Train / Val / Test:      {r['pos_train_2016-2019']:,} / "
          f"{r['pos_validation_2020']:,} / {r['pos_test_2021-2022']:,}")
    print(f"  Positive rate (train / val / test): "
          f"{r['train_positive_pct']:.4f}% / {r['validation_positive_pct']:.4f}% / "
          f"{r['test_positive_pct']:.4f}%")

print("\nPositives per billing year:")
years_hdr = "  ".join(str(y) for y in YEARS)
print(f"  {'scenario':36s} {years_hdr}")
for name, data in per_year.items():
    d = dict(data)
    vals = "  ".join(f"{d.get(y, 0):>4}" for y in YEARS)
    print(f"  {name:36s} {vals}")

# Name agreement: how many NPI-only positives (A) are confirmed by C?
a = next(r for r in rows if r["scenario"].startswith("A_"))
c = next(r for r in rows if r["scenario"].startswith("C_"))
if a["positive_provider_years"]:
    agree = c["positive_provider_years"] / a["positive_provider_years"]
    print(f"\nNAME AGREEMENT: {c['positive_provider_years']:,} of "
          f"{a['positive_provider_years']:,} Scenario-A positives "
          f"({agree:.1%}) are confirmed by name + state.")
    print("  ~80%+ = NPI matches are trustworthy. "
          "Much lower = inspect mismatches before trusting A.")


# ============================================================
# 11. DECISIONS
# ============================================================

banner("LABEL FEASIBILITY DECISIONS")

for r in rows:
    positives = r["positive_provider_years"]
    test_pos = r["pos_test_2021-2022"]

    if positives >= 300:
        decision = "PROCEED (SUPERVISED LEARNING)"
    elif positives >= 100:
        decision = "PROCEED CAUTIOUSLY (PU LEARNING)"
    else:
        decision = "FALLBACK (ANOMALY DETECTION / BROADEN LABELS)"

    warning = ""
    if test_pos < 30:
        warning = " | WARNING: Fewer than 30 test positives; metrics may be unstable"

    print(f"\n{r['scenario']}\n  Positives: {positives:,}\n  Decision: {decision}{warning}")


# ============================================================
# 12. EXPORT
# ============================================================

banner("EXPORTING RESULTS")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")

con.close()
banner("LABEL FEASIBILITY AUDIT COMPLETED")