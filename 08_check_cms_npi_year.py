"""Run AFTER any script that writes out/cms_npi_year.parquet. Compares against the verified counts."""
import pandas as pd

EXPECTED = {2013: 956251, 2014: 986657, 2015: 1019377, 2016: 1053958, 2017: 1088687, 2018: 1121462,
            2019: 1155870, 2020: 1161542, 2021: 1198754, 2022: 1230293, 2023: 1259343}   # from your own successful run

d = pd.read_parquet("out/cms_npi_year.parquet")
print("rows:", f"{len(d):,}", "| unique NPIs:", f"{d['npi'].nunique():,}", "| expected rows:", f"{sum(EXPECTED.values()):,}")
print("duplicate (npi, year) pairs:", int(d.duplicated(["npi", "year"]).sum()))
print("NPIs that are not exactly 10 digits:", int((~d["npi"].astype(str).str.fullmatch(r"\d{10}")).sum()))
got = d.groupby("year").size().to_dict()
bad = {y: (got.get(y), n) for y, n in EXPECTED.items() if got.get(y) != n}
print("\nYEAR COUNTS:", "ALL MATCH" if not bad and set(got) == set(EXPECTED) else "MISMATCH")
for y in sorted(set(got) | set(EXPECTED)):
    print(f"  {y}: got {got.get(y, 0):>9,}  expected {EXPECTED.get(y, 0):>9,}", "" if got.get(y) == EXPECTED.get(y) else "  <-- DIFFERENT")