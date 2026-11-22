"""
FraudLens - P3 model pipeline (supervised ranking + anomaly ranking)

Run:
    python src/p3_models.py --data stub
    python src/p3_models.py --data real
    python src/p3_models.py --data real --data-dir "files from team"

Step 2 (this version): load features + labels, keep eligible rows,
report how many eligible rows have CMS features.
"""

# argparse is a Python library used to handle command-line arguments
import argparse
from pathlib import Path

import pandas as pd
# as data is stored in Parquet files, and we need to load and process large datasets efficiently
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO=Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Feature list: the 30 numeric columns from docks/FEATURE_DICTIONARY.md
# (34 columns minus npi, year, specialty, state).
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # operational / volume / financial
    "medicare_participating", "tot_benes", "tot_srvcs",
    "submitted_charges", "allowed_amount", "medicare_payment",
    # ratios
    "pymt_per_bene", "srvcs_per_bene", "pymt_per_srvc", "submitted_per_bene",
    "charge_to_allowed", "allowed_to_pymt", "pymt_to_charge",
    # log-scaled volume
    "log_medicare_payment", "log_tot_benes", "log_tot_srvcs",
    # specialty peer robust z-scores
    "pymt_per_bene_peer_z", "srvcs_per_bene_peer_z", "charge_to_allowed_peer_z",
    # year-over-year (uses t-1 only)
    "medicare_payment_yoy_diff", "medicare_payment_yoy_growth",
    "tot_benes_yoy_diff", "tot_benes_yoy_growth",
    "tot_srvcs_yoy_diff", "tot_srvcs_yoy_growth",
    "submitted_charges_yoy_diff", "submitted_charges_yoy_growth",
    "pymt_per_bene_yoy_diff", "pymt_per_bene_yoy_growth",
    "is_new_billing_provider",
]

# Columns that must NEVER be model inputs (leak the label or are metadata).
BANNED_EXACT = {"npi", "year", "specialty", "state", "eligible", "y", "censored"}
BANNED_PREFIXES = ("y_", "censored_", "excluded_", "label_")

# Stub uses different names -> map them onto the real names.
STUB_RENAME = {
    "NPI": "npi",
    "label_scenario_a": "y",
    "total_payment": "medicare_payment",
    "beneficiaries": "tot_benes",
    "services": "tot_srvcs",
    "payment_per_beneficiary": "pymt_per_bene",
    "services_per_beneficiary": "srvcs_per_bene",
}

def check_no_leakage(cols):
    '''It ensures that ML model's feature list does not accidentally contain columns
      that could reveal the target variable or information that shouldn't be used for prediction.    
    '''
    bad=[c for c in cols if c in BANNED_EXACT or c.startswith(BANNED_PREFIXES)]

    # An assert checks whether a condition is true. If it's false, Python raises an AssertionError.
    assert not bad, f'Banned columns in feature list: {bad}'

def _cast(table,col,typ):
    '''Changes the data type of a column in a PyArrow Table'''
    # table:	The PyArrow Table containing your data
    # col:	    The name of the column whose type you want to change
    # typ:	    The target data type
    i=table.schema.get_field_index(col) # The variable i is needed to tell PyArrow where to place the converted column in the table.
    return table.set_column(i,col,pc.cast(table[col],typ)) # pc.cast() is the operation that actually performs the data type conversion.

# ---------------------------------------------------------------------------
# Loaders. Both return (df, feature_cols, horizon) where df has columns:
#   npi (str), year (int), y (0/1), censored (bool), has_features (bool),
#   + feature columns (float32, NaN where missing)
# and contains ONLY eligible provider-years.
# ---------------------------------------------------------------------------

def load_real(data_dir:Path):
    '''
    main data-loading function for real FraudLens dataset. It loads labels and features from Parquet files, 
    filters the data, performs validation, optimizes data types, joins the datasets, and finally converts everything 
    into a Pandas DataFrame for downstream ML.
    '''

    print(f'--> Loading real data from {data_dir}')

    # Labels: keep eligible rows only. Primary target y_A_h2.
    lab=pq.read_table(data_dir / "labels.parquet", 
                      columns=["npi", "year", "eligible", "y_A_h2", "censored_h2"]) # only load the five columns you need instead of loading the entire file.
    
    lab = lab.filter(
        pc.equal(
            lab["eligible"], True) # Creates a Boolean mask indicating which rows have eligible == True.
            ).drop(["eligible"]) # After filtering, the eligible column is no longer necessary, so you remove it.
    lab = _cast(lab, "year", pa.int32())

    # Rename the columns
    lab = lab.rename_columns(["npi", "year", "y", "censored"])
    print(f"eligible label rows: {lab.num_rows:,}")

    # Cross-check: candidate_pool should be exactly the eligible keys.
    n_cand = pq.ParquetFile(data_dir / "candidate_pool.parquet").metadata.num_rows # retrieving the number of rows in candidate_pool.parquet.
    print(f"candidate_pool rows: {n_cand:,}")
    assert n_cand == lab.num_rows, "candidate_pool and eligible labels disagree - ask P1"

    # Features: only the model columns, cast to float32 to save RAM.
    feat = pq.read_table(data_dir / "provider_features.parquet",
                         columns=["npi", "year"] + FEATURE_COLS # FEATURE_COLS contains your 30 numerical model features.
                        )
    feat = _cast(feat, "year", pa.int32())

    # Iterates over all 30 features and converts each one to float32.
    for c in FEATURE_COLS:
        feat = _cast(feat, c, pa.float32())
    feat = feat.append_column("has_features", pa.array([True] * feat.num_rows))
    print(f"    feature rows: {feat.num_rows:,}")

    # Left join: keep every eligible row, features are null if CMS row missing.
    joined = lab.join(feat, 
                      keys=["npi", "year"], 
                      join_type="left outer")
    df = joined.to_pandas()
    df["has_features"] = df["has_features"].fillna(False).astype(bool) # Remember that some providers had no matching feature row.
    return df, FEATURE_COLS, 2

def load_stub(stub_dir: Path):
    '''Loads synthetic/test data into a format compatible with your real-data pipeline.'''

    print(f"--> Loading STUB data from {stub_dir} (synthetic - never report results)")
    feat = pd.read_parquet(stub_dir / "provider_features.parquet").rename(columns=STUB_RENAME)
    cand = pd.read_parquet(stub_dir / "candidate_pool.parquet").rename(columns=STUB_RENAME)
    lab = pd.read_parquet(stub_dir / "labels.parquet").rename(columns=STUB_RENAME)

    # merging the candidate pool with the labels.
    df = cand.merge(
        lab[["npi", "year", "y"]], 
        on=["npi", "year"], 
        validate="one_to_one")

    # Identify available features
    feat_cols = [c for c in FEATURE_COLS if c in feat.columns]
    # Create a feature-only DataFrame
    f = feat[["npi", "year"] + feat_cols].copy()
    f[feat_cols] = f[feat_cols].astype("float32")

    f["has_features"] = True
    df = df.merge(
        f, 
        on=["npi", "year"], 
        how="left", validate="one_to_one")
    df["has_features"] = df["has_features"].fillna(False).astype(bool)

    # stub dataset doesn't contain a censoring column.
    # But real-data loader returns: censored
    df["censored"] = False   # stub has no censoring column
    return df, feat_cols, 3  # stub labels use a 3-year window

# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def coverage_report(df, out_csv: Path):
    '''Analyzes how many eligible providers have features available, broken down by year.'''

    # This groups the dataset by observation year.
    g = df.groupby("year").agg(
        eligible_rows=("y", "size"),
        with_features=("has_features", "sum"),
        positives=("y", "sum"),
    )

    g["without_features"] = g["eligible_rows"] - g["with_features"]
    g["positives_without_features"] = df[~df["has_features"]].groupby("year")["y"].sum()
    g = g.fillna(0).astype(int).reset_index()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Creates the output directory if it doesn't already exist.
    g.to_csv(out_csv, index=False)
    print("\n=== SCORING COVERAGE (eligible rows) ===")
    print(g.to_string(index=False))
    print(f"saved {out_csv}")
    return g


# ---------------------------------------------------------------------------
def main():
    # Creates an argument parser object.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["stub", "real"], required=True)
    ap.add_argument("--data-dir", default=None,
                    help="folder with the input parquet files")
    args = ap.parse_args()

    # If the user provides a custom data directory → use it.
    # Otherwise → use the default stub directory inside the repository.
    if args.data == "stub":
        data_dir = Path(args.data_dir) if args.data_dir else REPO / "shared_p3_p4_stub"
        out_dir = report_dir = REPO / "out" / "stub"
        df, feat_cols, h = load_stub(data_dir)
    else:
        data_dir = Path(args.data_dir) if args.data_dir else REPO / "files from team"
        out_dir, report_dir = REPO / "out", REPO / "reports"
        df, feat_cols, h = load_real(data_dir)

    check_no_leakage(feat_cols)
    assert not df.duplicated(["npi", "year"]).any(), "duplicate (npi, year) rows"

    print(f"\nRows: {len(df):,} | features used: {len(feat_cols)} | horizon h={h}")
    print(f"Positives: {int(df['y'].sum()):,} | positive NPIs: {df.loc[df['y'] == 1, 'npi'].nunique():,}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    coverage_report(df, report_dir / "p3_scoring_coverage.csv")


if __name__ == "__main__":
    main()