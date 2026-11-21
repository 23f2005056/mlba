
from pathlib import Path

import pandas as pd

from config import LEIE_MAIN, OUT, REPORTS


# ---------------------------------------------------------
# 1. SET UP DIRECTORIES
# ---------------------------------------------------------

OUT = Path(OUT)
REPORTS = Path(REPORTS)

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD LEIE DATA
# ---------------------------------------------------------

df = pd.read_csv(
    LEIE_MAIN,
    dtype="string",
    keep_default_na=False,
    low_memory=False
)

print("Original LEIE records:", len(df))


# ---------------------------------------------------------
# 3. VALIDATE NPI NUMBERS
# ---------------------------------------------------------

def valid_npi(value):
    """Validate an NPI using the CMS Luhn check-digit method."""

    if len(value) != 10 or not value.isdigit():
        return False

    if value == "0000000000":
        return False

    digits = "80840" + value[:9]
    total = 0

    for i, char in enumerate(digits):
        digit = int(char)

        if i % 2 == 1:
            digit *= 2
            digit = digit // 10 + digit % 10

        total += digit

    expected = (10 - total % 10) % 10

    return expected == int(value[-1])


npi_raw = df["NPI"].str.strip()

df["NPI_RAW"] = npi_raw

npi_valid = npi_raw.map(valid_npi).fillna(False).astype(bool)

df["NPI_VALID"] = npi_valid

# Invalid and placeholder values become missing.
# The original value is preserved in NPI_RAW.
df["NPI"] = npi_raw.where(npi_valid, pd.NA)


# ---------------------------------------------------------
# 4. STANDARDISE EXCLUSION TYPES
# ---------------------------------------------------------

df["EXCLTYPE_RAW"] = df["EXCLTYPE"].str.strip()

df["EXCLTYPE"] = (
    df["EXCLTYPE_RAW"]
    .str.lower()
)


# ---------------------------------------------------------
# 5. PARSE EXCLUSION AND REINSTATEMENT DATES
# ---------------------------------------------------------

def parse_leie_date(column):

    raw = df[column].str.strip()

    # The values 0 and 00000000 are placeholders.
    placeholder = raw.isin(["", "0", "00000000"])

    parsed = pd.to_datetime(
        raw.where(~placeholder),
        format="%Y%m%d",
        errors="coerce"
    )

    invalid = (~placeholder) & parsed.isna()

    return parsed, placeholder, invalid


df["EXCLDATE_DT"], excl_placeholder, excl_invalid = (
    parse_leie_date("EXCLDATE")
)

df["REINDATE_DT"], rein_placeholder, rein_invalid = (
    parse_leie_date("REINDATE")
)


# Extract exclusion year for future label construction.
df["EXCLUSION_YEAR"] = df["EXCLDATE_DT"].dt.year.astype("Int64")


# ---------------------------------------------------------
# 6. SEPARATE NPI-LINKABLE RECORDS
# ---------------------------------------------------------

leie_npi = df.loc[df["NPI_VALID"]].copy()

print("\nNPI-linkable records:", len(leie_npi))
print("Unique valid NPIs:", leie_npi["NPI"].nunique())


# ---------------------------------------------------------
# 7. INVESTIGATE REPEATED NPIs
# ---------------------------------------------------------

duplicate_npis = (
    leie_npi.groupby("NPI")
    .agg(
        record_count=("NPI", "size"),
        first_exclusion=("EXCLDATE_DT", "min"),
        last_exclusion=("EXCLDATE_DT", "max"),
        distinct_exclusion_types=("EXCLTYPE", "nunique")
    )
    .reset_index()
)

duplicate_npis = duplicate_npis[
    duplicate_npis["record_count"] > 1
]

duplicate_npis.to_csv(
    REPORTS / "leie_duplicate_npi_groups.csv",
    index=False
)


# ---------------------------------------------------------
# 8. CHECK REPEATED EXCLUSION EVENTS
# ---------------------------------------------------------

duplicate_event_rows = leie_npi.duplicated(
    subset=["NPI", "EXCLDATE_DT", "EXCLTYPE"],
    keep="first"
)

duplicate_event_count = int(duplicate_event_rows.sum())


# ---------------------------------------------------------
# 9. SAVE CLEANED DATASETS
# ---------------------------------------------------------

# Full LEIE data, including records without NPIs.
df.to_parquet(
    OUT / "leie_clean.parquet",
    index=False
)

# Only records suitable for NPI-based CMS matching.
leie_npi.to_parquet(
    OUT / "leie_npi_events.parquet",
    index=False
)


# ---------------------------------------------------------
# 10. SAVE CLEANING SUMMARY
# ---------------------------------------------------------

summary = {
    "original_records": len(df),
    "valid_npi_records": int(npi_valid.sum()),
    "records_without_valid_npi": int((~npi_valid).sum()),
    "unique_valid_npis": int(leie_npi["NPI"].nunique()),
    "repeated_npi_groups": len(duplicate_npis),
    "extra_repeated_npi_rows": (
        len(leie_npi) - leie_npi["NPI"].nunique()
    ),
    "duplicate_exclusion_event_rows": duplicate_event_count,
    "exclusion_date_placeholders": int(excl_placeholder.sum()),
    "invalid_exclusion_dates": int(excl_invalid.sum()),
    "reinstatement_date_placeholders": int(rein_placeholder.sum()),
    "invalid_reinstatement_dates": int(rein_invalid.sum())
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    REPORTS / "leie_cleaning_summary.csv",
    index=False
)


# ---------------------------------------------------------
# 11. PRINT RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LEIE CLEANING COMPLETED")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved:")
print(OUT / "leie_clean.parquet")
print(OUT / "leie_npi_events.parquet")
print(REPORTS / "leie_duplicate_npi_groups.csv")
print(REPORTS / "leie_cleaning_summary.csv")

