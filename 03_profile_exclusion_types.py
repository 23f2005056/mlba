
import pandas as pd
from pathlib import Path

# -------------------------------------------------------
# 1. LOAD CLEANED NPI-LINKABLE LEIE RECORDS
# -------------------------------------------------------

df = pd.read_parquet("out/leie_npi_events.parquet")

df["EXCLTYPE"] = (
    df["EXCLTYPE"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("NPI-linkable records:", len(df))
print("Unique NPIs:", df["NPI"].nunique())


# -------------------------------------------------------
# 2. MAP OFFICIAL OIG EXCLUSION DESCRIPTIONS
# -------------------------------------------------------

CODE_MEANINGS = {
    "1128a1": "Program-related criminal conviction",
    "1128a2": "Patient abuse or neglect conviction",
    "1128a3": "Felony health-care fraud conviction",
    "1128a4": "Felony controlled-substance conviction",
    "1128b1": "Misdemeanor health-care or other program fraud conviction",
    "1128b2": "Obstruction of investigation or audit",
    "1128b3": "Misdemeanor controlled-substance conviction",
    "1128b4": "Licence revocation, suspension or surrender",
    "1128b5": "Exclusion or suspension by another health-care programme",
    "1128b6": "Excessive charges, unnecessary services or substandard care",
    "1128b7": "Fraud, kickbacks and other prohibited activities",
    "1128b8": "Entity controlled by a sanctioned individual",
    "1128b11": "Failure to provide required information",
    "1128b12": "Failure to grant immediate access",
    "1128b13": "Failure to take corrective action",
    "1128b14": "Default on health education loan or scholarship",
    "1128b15": "Individual controlling a sanctioned entity",
    "1128b16": "False statement or material misrepresentation",
    "1156": "Failure to meet statutory medical-care obligations"
}


# -------------------------------------------------------
# 3. PROVISIONAL SCENARIO A DEFINITION
# -------------------------------------------------------

SCENARIO_A_TYPES = {
    "1128a1",
    "1128a3",
    "1128b1",
    "1128b6",
    "1128b7"
}

df["SCENARIO_A_CANDIDATE"] = (
    df["EXCLTYPE"].isin(SCENARIO_A_TYPES)
)


# -------------------------------------------------------
# 4. IDENTIFY DISTINCT EXCLUSION EVENTS
# -------------------------------------------------------

event_columns = [
    "NPI",
    "EXCLDATE_DT",
    "EXCLTYPE"
]

distinct_events = df.drop_duplicates(
    subset=event_columns
).copy()


# -------------------------------------------------------
# 5. COUNT EACH EXCLUSION TYPE
# -------------------------------------------------------

summary = (
    df.groupby("EXCLTYPE")
    .agg(
        raw_records=("NPI", "size"),
        unique_npis=("NPI", "nunique")
    )
)

event_counts = (
    distinct_events.groupby("EXCLTYPE")
    .size()
    .rename("distinct_events")
)

summary = summary.join(event_counts)

summary["description"] = [
    CODE_MEANINGS.get(
        code,
        "Unmapped code: requires further investigation"
    )
    for code in summary.index
]

summary["scenario_A_candidate"] = (
    summary.index.isin(SCENARIO_A_TYPES)
)

summary = (
    summary.reset_index()
    .sort_values("raw_records", ascending=False)
)


# -------------------------------------------------------
# 6. PRINT RESULTS
# -------------------------------------------------------

print("\n" + "=" * 85)
print("EXCLUSION TYPE ANALYSIS")
print("=" * 85)

print(
    summary.to_string(index=False)
)

print("\n" + "=" * 85)
print("SCENARIO A CANDIDATE SUMMARY")
print("=" * 85)

a_events = distinct_events[
    distinct_events["EXCLTYPE"].isin(SCENARIO_A_TYPES)
]

print("Raw Scenario A records:",
      df["SCENARIO_A_CANDIDATE"].sum())

print("Distinct Scenario A exclusion events:",
      len(a_events))

print("Unique Scenario A NPIs:",
      a_events["NPI"].nunique())

print("\nScenario B - all exclusion types")

print("Distinct exclusion events:",
      len(distinct_events))

print("Unique NPIs:",
      distinct_events["NPI"].nunique())


# -------------------------------------------------------
# 7. SAVE REPORT
# -------------------------------------------------------

output_path = Path(
    "reports/leie_exclusion_type_npi_profile.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    output_path,
    index=False
)

print("\nReport saved to:", output_path)
