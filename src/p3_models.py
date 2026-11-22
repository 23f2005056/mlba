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