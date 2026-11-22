"""Pick random positives and show the LEIE details so you can verify a few by hand."""
import pandas as pd

YEAR, COL = 2022, "y_A_h2"          # prediction year and label column to inspect
N = 8

lab = pd.read_parquet("out/labels.parquet", columns=["npi", "year", "eligible", COL])
pos = lab[(lab["year"] == YEAR) & (lab[COL] == 1)].sample(N, random_state=7)

leie = pd.read_csv("raw/leie/UPDATED.csv", dtype=str, keep_default_na=False, encoding="latin-1")
leie.columns = [c.strip().upper() for c in leie.columns]
leie["NPI"] = leie["NPI"].str.strip()
cols = ["NPI", "LASTNAME", "FIRSTNAME", "BUSNAME", "GENERAL", "SPECIALTY", "STATE", "EXCLTYPE", "EXCLDATE"]
out = pos[["npi"]].merge(leie[cols], left_on="npi", right_on="NPI", how="left").drop(columns="NPI")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(f"Positives for prediction year {YEAR}, label {COL}: exclusion must fall in {YEAR+1}-{YEAR+2}\n")
print(out.to_string(index=False))
bad = out[~out["EXCLDATE"].str[:4].astype(int).between(YEAR + 1, YEAR + 2)]
print("\nrows whose EXCLDATE is outside the expected window (should be 0 unless the NPI has several records):", len(bad))