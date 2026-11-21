
import re
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SCENARIO_A = {
    "1128a1",
    "1128a3",
    "1128b1",
    "1128b6",
    "1128b7"
}


# --------------------------------------------------
# NPI VALIDATION
# --------------------------------------------------

def npi_ok(value):
    value = str(value).strip()

    # A real NPI must contain exactly 10 digits.
    # The LEIE file uses a single "0" for missing NPI.
    if not re.fullmatch(r"\d{10}", value):
        return False

    if value == "0000000000":
        return False

    total = 0

    for i, char in enumerate(reversed("80840" + value)):
        digit = int(char)

        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9

        total += digit

    return total % 10 == 0


# --------------------------------------------------
# LOAD LEIE
# --------------------------------------------------

df = pd.read_csv(
    "raw/leie/UPDATED.csv",
    dtype=str,
    keep_default_na=False,
    encoding="latin-1"
)

df.columns = [c.strip().upper() for c in df.columns]

npi = df["NPI"].str.strip()

df["valid_npi"] = npi.map(npi_ok)

df["year"] = pd.to_datetime(
    df["EXCLDATE"].str.strip(),
    format="%Y%m%d",
    errors="coerce"
).dt.year

df["type"] = (
    df["EXCLTYPE"]
    .str.lower()
    .str.replace(r"[^a-z0-9]", "", regex=True)
)

df["scenario_A"] = df["type"].isin(SCENARIO_A)


# --------------------------------------------------
# OVERALL COVERAGE
# --------------------------------------------------

print("=" * 65)
print("CORRECTED LEIE NPI COVERAGE")
print("=" * 65)

print("Total LEIE records:", len(df))

print("Records with NPI = 0:", int(npi.eq("0").sum()))

print("Blank NPI records:", int(npi.eq("").sum()))

print(
    "Records with valid NPI:",
    int(df["valid_npi"].sum()),
    f"({df['valid_npi'].mean():.2%})"
)

print(
    "Unique valid NPIs:",
    df.loc[df["valid_npi"], "NPI"].nunique()
)

print(
    "Scenario A records with valid NPI:",
    int((df["scenario_A"] & df["valid_npi"]).sum())
)

print(
    "Unique Scenario A NPIs:",
    df.loc[df["scenario_A"] & df["valid_npi"], "NPI"].nunique()
)


# --------------------------------------------------
# EXCLUSIONS BY YEAR
# --------------------------------------------------

all_years = df.groupby("year").size()

valid_years = (
    df.loc[df["valid_npi"]]
    .groupby("year")
    .size()
)

scenario_years = (
    df.loc[df["valid_npi"] & df["scenario_A"]]
    .groupby("year")
    .size()
)

year_table = pd.concat(
    [
        all_years.rename("all_records"),
        valid_years.rename("valid_npi"),
        scenario_years.rename("scenario_A_valid_npi")
    ],
    axis=1
).fillna(0).astype(int)

year_table = year_table.loc[year_table.index >= 2008]

print("\nEXCLUSIONS PER YEAR (2008 onwards)")
print(year_table.to_string())


# --------------------------------------------------
# PROVIDER CATEGORY ANALYSIS
# --------------------------------------------------

print("\nTOP GENERAL CATEGORIES WITH VALID NPIs")

print(
    df.loc[df["valid_npi"], "GENERAL"]
    .str.strip()
    .value_counts()
    .head(15)
    .to_string()
)

print("\nTOP GENERAL CATEGORIES IN SCENARIO A")

print(
    df.loc[df["valid_npi"] & df["scenario_A"], "GENERAL"]
    .str.strip()
    .value_counts()
    .head(10)
    .to_string()
)


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

Path("reports").mkdir(exist_ok=True)

year_table.to_csv(
    "reports/npi_coverage_corrected.csv"
)

print("\nReport saved: reports/npi_coverage_corrected.csv")
