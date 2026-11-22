
import pandas as pd

df = pd.read_parquet("out/labels.parquet")

print("Total label rows:", len(df))
print("Unique NPIs:", df["npi"].nunique())
print("Duplicate NPI-year pairs:",
      df.duplicated(["npi", "year"]).sum())

for h in [1, 2, 3]:

    censored = df[f"censored_h{h}"].fillna(False)

    print(f"\nHORIZON: {h} YEARS")
    print("Censored rows:", int(censored.sum()))

    for scenario in ["A", "B"]:

        col = f"y_{scenario}_h{h}"

        missing = int(df.loc[censored, col].isna().sum())
        nonmissing = int(df.loc[censored, col].notna().sum())

        print(
            f"{col}: "
            f"censored unknown={missing:,}, "
            f"censored with a label={nonmissing:,}"
        )

print("\nCensored years for 3-year horizon:")
print(sorted(df.loc[
    df["censored_h3"].fillna(False), "year"
].unique()))
