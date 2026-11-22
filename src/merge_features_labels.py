"""
FraudLens: Feature & Label Integration Utility
Author: Person 2 (CMS Data & Feature Engineering)
Deliverable: CP-13 Reusable Integration Utility

Purpose:
Memory-efficient joining of provider features (P2) with candidate pool
eligibility and LEIE labels (P1) on key (npi, year).

Usage:
    # 1. Fast prototype mode using 2% development sample:
    python src/merge_features_labels.py --sample

    # 2. Full production join:
    python src/merge_features_labels.py --full

    # 3. Direct Python import in P3 training script:
    from src.merge_features_labels import load_modeling_data
    df_train, df_val, df_test = load_modeling_data(sample=True)
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out"

def load_modeling_data(
    sample: bool = False,
    feature_cols: list = None,
    target_col: str = "y_A_h2",
    censor_col: str = "censored_h2",
    require_eligible: bool = True
):
    """
    Loads features and joins with labels/candidate pool with zero memory bloat.
    """
    t0 = time.time()
    
    # 1. Select feature file
    feat_file = OUT_DIR / ("provider_features_sample_2pct.parquet" if sample else "provider_features.parquet")
    if not feat_file.exists():
        raise FileNotFoundError(f"Feature file not found at: {feat_file}")
        
    print(f"--> Ingesting features from {feat_file.name}...")
    
    # Use PyArrow dataset for zero-copy column filtering if specified
    if feature_cols:
        cols_to_load = list(set(['npi', 'year'] + feature_cols))
        feat_df = pq.read_table(feat_file, columns=cols_to_load).to_pandas()
    else:
        feat_df = pd.read_parquet(feat_file)
        
    print(f"    Loaded {len(feat_df):,} feature rows in {time.time() - t0:.2f}s (RAM: {feat_df.memory_usage().sum() / (1024**2):.1f} MB)")
    
    # Ensure key types match contract
    feat_df['npi'] = feat_df['npi'].astype(str).str.zfill(10)
    feat_df['year'] = feat_df['year'].astype('int32')
    
    # 2. Check for P1 Labels
    labels_file = OUT_DIR / "labels.parquet"
    cand_file = OUT_DIR / "candidate_pool.parquet"
    
    if not labels_file.exists() or not cand_file.exists():
        print(f"\n[NOTE] Labels or Candidate Pool not found in {OUT_DIR}.")
        print("       (P1 datasets are hosted on Team Drive. Returning unmerged feature set).")
        return feat_df
        
    t1 = time.time()
    print("--> Joining with P1 candidate pool and labels...")
    
    # Ingest Candidate Pool
    cand_cols = ['npi', 'year', 'eligible']
    cand_df = pd.read_parquet(cand_file, columns=cand_cols)
    cand_df['npi'] = cand_df['npi'].astype(str).str.zfill(10)
    cand_df['year'] = cand_df['year'].astype('int32')
    
    # Ingest Labels
    lab_cols = ['npi', 'year', target_col, censor_col]
    lab_df = pd.read_parquet(labels_file, columns=lab_cols)
    lab_df['npi'] = lab_df['npi'].astype(str).str.zfill(10)
    lab_df['year'] = lab_df['year'].astype('int32')
    
    # Merge
    merged = pd.merge(feat_df, cand_df, on=['npi', 'year'], how='inner')
    merged = pd.merge(merged, lab_df, on=['npi', 'year'], how='left')
    
    if require_eligible:
        n_before = len(merged)
        merged = merged[merged['eligible'] == True].copy()
        print(f"    Filtered eligible==True: {len(merged):,} rows kept ({n_before - len(merged):,} dropped)")
        
    # Uncensored rows for training
    if censor_col in merged.columns:
        n_censored = merged[censor_col].fillna(False).sum()
        print(f"    Censored observations ({censor_col}==True): {n_censored:,}")
        
    print(f"--> Join completed successfully in {time.time() - t1:.2f}s! Total shape: {merged.shape}")
    return merged

def main():
    parser = argparse.ArgumentParser(description="FraudLens Feature & Label Integration Utility")
    parser.add_argument("--sample", action="store_true", help="Use 2% fast development sample")
    parser.add_argument("--full", action="store_true", help="Use full 11.55M provider features dataset")
    args = parser.parse_args()
    
    use_sample = not args.full  # Default to sample if not full
    mode_str = "2% Development Sample" if use_sample else "Full 11.55M Provider Dataset"
    
    print("=" * 65)
    print(f"FRAUDLENS DATA INTEGRATION UTILITY ({mode_str})")
    print("=" * 65)
    
    df = load_modeling_data(sample=use_sample)
    print(f"\n[SUMMARY] Data shape: {df.shape}")
    print(f"[SUMMARY] Distinct NPIs: {df['npi'].nunique():,}")
    print(f"[SUMMARY] Years present: {sorted(df['year'].unique())}")
    print("=" * 65)

if __name__ == "__main__":
    main()
