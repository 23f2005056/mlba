
import re
import pandas as pd

from config import *

# ============================================
# 1. LOAD RAW LEIE DATA
# ============================================

print("=" * 65)
print("FRAUDLENS - LEIE DATA AUDIT")
print("=" * 65)

df = pd.read_csv(
    LEIE_MAIN,
    dtype=str,
    keep_default_na=False,
    encoding="latin-1"
)

df.columns = [c.strip().upper() for c in df.columns]

print("\nDATASET OVERVIEW")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", df.columns.tolist())

print("\nFirst five rows:")
print(df.head().to_string())

# ============================================
# 2. VERIFY REQUIRED COLUMNS
# ============================================

required = [
    "LASTNAME",
    "FIRSTNAME",
    "BUSNAME",
    "GENERAL",
    "SPECIALTY",
    "NPI",
    "STATE",
    "EXCLTYPE",
    "EXCLDATE",
    "REINDATE"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Required LEIE columns missing: {missing}"
    )

# ============================================
# 3. NPI QUALITY
# ============================================

def npi_luhn_ok(value):
    """Validate the NPI check digit using the 80840 prefix."""

    if not re.fullmatch(r"\d{10}", value or ""):
        return False

    if value == "0000000000":
        return False

    total = 0

    for i, digit in enumerate(
        reversed("80840" + value)
    ):
        n = int(digit)

        if i % 2 == 1:
            n *= 2

            if n > 9:
                n -= 9

        total += n

    return total % 10 == 0


npi = df["NPI"].str.strip()

has_npi = (
    npi.ne("")
    & npi.ne("0000000000")
)

valid_npi = npi.map(npi_luhn_ok)

print("\nNPI QUALITY")
print("Total records:", len(df))
print("Records with usable-format NPI:", has_npi.sum())
print("Missing/zero NPI:", (~has_npi).sum())
print("Valid NPI check digits:", valid_npi.sum())

print(
    "Usable-format NPI percentage:",
    round(has_npi.mean() * 100, 2),
    "%"
)

print(
    "Check-digit-valid percentage:",
    round(valid_npi.mean() * 100, 2),
    "%"
)

print("Unique usable NPIs:", npi[has_npi].nunique())

# ============================================
# 4. INDIVIDUALS VS ORGANISATIONS
# ============================================

entities = df["BUSNAME"].str.strip().ne("")

print("\nENTITY ANALYSIS")
print("Organisation records:", entities.sum())
print("Individual records:", (~entities).sum())

# ============================================
# 5. EXCLUSION TYPES
# ============================================

print("\nEXCLUSION TYPE DISTRIBUTION")

excl_counts = (
    df["EXCLTYPE"]
    .str.strip()
    .value_counts()
    .rename_axis("excl_type")
    .reset_index(name="record_count")
)

print(excl_counts.head(30).to_string(index=False))

excl_counts.to_csv(
    REPORTS / "exclusion_type_counts.csv",
    index=False
)

# ============================================
# 6. EXCLUSION DATE QUALITY
# ============================================

excl_raw = df["EXCLDATE"].str.strip()

excl_date = pd.to_datetime(
    excl_raw,
    format="%Y%m%d",
    errors="coerce"
)

invalid_excl = excl_raw.ne("") & excl_date.isna()

print("\nEXCLUSION DATE QUALITY")
print("Invalid nonblank dates:", invalid_excl.sum())
print("Missing dates:", excl_raw.eq("").sum())

if excl_date.notna().any():
    print("Earliest exclusion:", excl_date.min())
    print("Latest exclusion:", excl_date.max())

exclusions_by_year = pd.DataFrame({
    "year": excl_date.dt.year,
    "has_npi": has_npi
})

year_audit = exclusions_by_year.groupby("year").agg(
    all_records=("has_npi", "size"),
    records_with_npi=("has_npi", "sum")
).reset_index()

print("\nEXCLUSIONS BY YEAR")
print(year_audit.to_string(index=False))

year_audit.to_csv(
    REPORTS / "npi_coverage.csv",
    index=False
)

# ============================================
# 7. REINSTATEMENT DATE QUALITY
# ============================================

rein_raw = df["REINDATE"].str.strip()

rein_date = pd.to_datetime(
    rein_raw,
    format="%Y%m%d",
    errors="coerce"
)

invalid_rein = rein_raw.ne("") & rein_date.isna()

print("\nREINSTATEMENT ANALYSIS")
print("Rows with reinstatement date:", rein_date.notna().sum())
print("Invalid nonblank reinstatement dates:", invalid_rein.sum())

# ============================================
# 8. DUPLICATE NPI RECORDS
# ============================================

print("\nDUPLICATE NPI ANALYSIS")

print(
    "Duplicate NPI rows:",
    npi[has_npi].duplicated(keep=False).sum()
)

print(
    "Unique NPI count:",
    npi[has_npi].nunique()
)

# ============================================
# 9. SAVE REPORT
# ============================================

summary = [
    f"Total records: {len(df)}",
    f"Total columns: {len(df.columns)}",
    f"Records with usable-format NPI: {has_npi.sum()}",
    f"Records with valid NPI check digit: {valid_npi.sum()}",
    f"Unique usable NPIs: {npi[has_npi].nunique()}",
    f"Organisation records: {entities.sum()}",
    f"Missing/invalid exclusion dates: {invalid_excl.sum() + int(excl_raw.eq('').sum())}",
    f"Records with reinstatement date: {rein_date.notna().sum()}",
    f"Invalid reinstatement dates: {invalid_rein.sum()}"
]

(REPORTS / "leie_profile_summary.txt").write_text(
    "\n".join(summary),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("LEIE PROFILE COMPLETED")
print("Reports saved in:", REPORTS)
print("=" * 65)
