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
    df["has_features"] = df["has_features"].fillna(False).astype(bool)
    return df, FEATURE_COLS, 2