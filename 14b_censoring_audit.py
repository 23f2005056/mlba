"""Censoring audit for labels.parquet (extended version of the quick check)."""
import pandas as pd

df = pd.read_parquet("out/labels.parquet")
print("Total label rows:", f"{len(df):,}", "| unique NPIs:", f"{df['npi'].nunique():,}",
      "| duplicate (npi, year):", int(df.duplicated(['npi', 'year']).sum()))

ok = True
for h in (1, 2, 3):
    cens = df[f"censored_h{h}"].fillna(False).astype(bool)
    years = sorted(df.loc[cens, "year"].unique().tolist())
    print(f"\nHORIZON {h}: censored rows = {int(cens.sum()):,} | censored years = {years}")
    for S in ("A", "B"):
        col = f"y_{S}_h{h}"
        print(f"  {col}: NaN among censored = {int(df.loc[cens, col].isna().sum()):,} | "
              f"positives among censored (incomplete window) = {int(df.loc[cens, col].fillna(0).sum())}")
    if h in (1, 2) and cens.any():
        ok = False; print("  !! unexpected: censored rows for this horizon")
    if h == 3 and years not in ([], [2023]):
        ok = False; print("  !! unexpected censored years for h=3")

print("\nRULE: never train or evaluate with y_*_h3 on year 2023 (window incomplete). Primary label y_A_h2 is fully observed.")
print("AUDIT RESULT:", "as expected" if ok else "CHECK THE LINES MARKED !!")