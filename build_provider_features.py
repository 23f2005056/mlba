"""
FraudLens - Person 2 Feature Engineering Pipeline
Builds: out/provider_features.parquet (2013-2023)

Key Guarantees:
1. Zero Temporal Leakage: Lag features use strictly past years (t-1).
2. Specialty Context: Robust Z-scores using (X - Median) / (0.7413 * IQR) per (year, specialty).
3. Population Consistency: Individual providers ('I') with valid 10-digit NPIs.
4. Division Safety: Division by zero returns NaN, not fabricated zeroes.
"""

import os
import glob
import re
import gc
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CMS_DIR = BASE_DIR / "227664-V1"
OUTPUT_DIR = BASE_DIR / "out"
OUTPUT_FILE = OUTPUT_DIR / "provider_features.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD & HARMONIZE ANNUAL CMS PROVIDER FILES
# ----------------------------------------------------------------------
def load_and_clean_year(file_path):
    filename = os.path.basename(file_path)
    
    # Extract year (D13-D23 or 2013-2023)
    d_match = re.search(r'_D(\d{2})_', filename)
    if d_match:
        year = 2000 + int(d_match.group(1))
    else:
        y_match = re.search(r'(20\d\d)', file_path)
        year = int(y_match.group(1))
        
    print(f"--> Ingesting CMS Year {year}: {filename}")
    
    use_cols = [
        'Rndrng_NPI',
        'Rndrng_Prvdr_Ent_Cd',
        'Rndrng_Prvdr_Type',
        'Rndrng_Prvdr_State_Abrvtn',
        'Rndrng_Prvdr_Mdcr_Prtcptg_Ind',
        'Tot_Benes',
        'Tot_Srvcs',
        'Tot_Sbmtd_Chrg',
        'Tot_Mdcr_Alowd_Amt',
        'Tot_Mdcr_Pymt_Amt'
    ]
    
    df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)
    
    # Filter 1: Individual providers only ('I')
    df = df[df['Rndrng_Prvdr_Ent_Cd'] == 'I'].copy()
    
    # Clean NPI: 10-digit zero-padded string
    df['npi'] = (
        df['Rndrng_NPI']
        .astype(str)
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)
        .str.zfill(10)
    )
    valid_npi = df['npi'].str.match(r'^\d{10}$') & (df['npi'] != '0000000000')
    df = df[valid_npi].copy()
    
    # Rename core columns
    df['year'] = int(year)
    df.rename(columns={
        'Rndrng_Prvdr_Type': 'specialty',
        'Rndrng_Prvdr_State_Abrvtn': 'state',
        'Rndrng_Prvdr_Mdcr_Prtcptg_Ind': 'medicare_participating',
        'Tot_Benes': 'tot_benes',
        'Tot_Srvcs': 'tot_srvcs',
        'Tot_Sbmtd_Chrg': 'submitted_charges',
        'Tot_Mdcr_Alowd_Amt': 'allowed_amount',
        'Tot_Mdcr_Pymt_Amt': 'medicare_payment'
    }, inplace=True)
    
    # Clean Numerics (preserves adjustments, converts negative/invalid if any)
    numeric_cols = ['tot_benes', 'tot_srvcs', 'submitted_charges', 'allowed_amount', 'medicare_payment']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df['specialty'] = df['specialty'].fillna('Unknown').astype(str).str.strip()
    df['state'] = df['state'].fillna('XX').astype(str).str.strip()
    df['medicare_participating'] = (df['medicare_participating'] == 'Y').astype('int8')
    
    # Deduplicate strictly on (npi, year)
    df = df.drop_duplicates(subset=['npi', 'year'])
    
    keep_cols = [
        'npi', 'year', 'specialty', 'state', 'medicare_participating',
        'tot_benes', 'tot_srvcs', 'submitted_charges', 'allowed_amount', 'medicare_payment'
    ]
    return df[keep_cols]

# ----------------------------------------------------------------------
# 2. COMPUTE BASE FINANCIAL RATIOS
# ----------------------------------------------------------------------
def compute_base_ratios(df):
    print("--> Computing Base Financial & Operational Ratios...")
    
    # Safe division helper: returns NaN when denominator <= 0
    def safe_div(num, denom):
        return np.where(denom > 0, num / denom, np.nan)

    # 1. Billing intensity
    df['pymt_per_bene'] = safe_div(df['medicare_payment'], df['tot_benes'])
    df['srvcs_per_bene'] = safe_div(df['tot_srvcs'], df['tot_benes'])
    df['pymt_per_srvc'] = safe_div(df['medicare_payment'], df['tot_srvcs'])
    df['submitted_per_bene'] = safe_div(df['submitted_charges'], df['tot_benes'])
    
    # 2. Markup and pricing
    df['charge_to_allowed'] = safe_div(df['submitted_charges'], df['allowed_amount'])
    df['allowed_to_pymt'] = safe_div(df['allowed_amount'], df['medicare_payment'])
    df['pymt_to_charge'] = safe_div(df['medicare_payment'], df['submitted_charges'])
    
    # 3. Log-scaled volume measures
    df['log_medicare_payment'] = np.log1p(df['medicare_payment'].clip(lower=0))
    df['log_tot_benes'] = np.log1p(df['tot_benes'].clip(lower=0))
    df['log_tot_srvcs'] = np.log1p(df['tot_srvcs'].clip(lower=0))
    
    return df

# ----------------------------------------------------------------------
# 3. SPECIALTY PEER ROBUST Z-SCORES (IQR)
# ----------------------------------------------------------------------
def compute_peer_zscores(df, min_group_size=30):
    print("--> Computing Specialty Peer-Group Robust Z-Scores (0.7413 * IQR)...")
    
    features_to_standardize = ['pymt_per_bene', 'srvcs_per_bene', 'charge_to_allowed']
    
    for feat in features_to_standardize:
        z_col = f"{feat}_peer_z"
        
        # Group by year and specialty
        grp = df.groupby(['year', 'specialty'])[feat]
        
        # Calculate robust median and IQR
        med = grp.transform('median')
        q25 = grp.transform(lambda s: s.quantile(0.25))
        q75 = grp.transform(lambda s: s.quantile(0.75))
        counts = grp.transform('count')
        
        iqr = q75 - q25
        norm_factor = iqr * 0.7413
        
        # Valid groups: size >= min_group_size and norm_factor > 0
        valid_mask = (counts >= min_group_size) & (norm_factor > 1e-4) & (~df[feat].isna())
        
        df[z_col] = np.nan
        df.loc[valid_mask, z_col] = (df.loc[valid_mask, feat] - med[valid_mask]) / norm_factor[valid_mask]
        
    return df

# ----------------------------------------------------------------------
# 4. LONGITUDINAL YEAR-OVER-YEAR VELOCITY (LAG t-1)
# ----------------------------------------------------------------------
def compute_longitudinal_features(df):
    print("--> Computing Longitudinal Year-Over-Year Trends (Historical Lags Only)...")
    
    # Sort strictly by NPI and year to enforce time direction
    df = df.sort_values(by=['npi', 'year']).reset_index(drop=True)
    
    # Group by NPI for lag lookbacks
    npi_grp = df.groupby('npi')
    
    # Check if previous row is strictly year - 1
    prev_year = npi_grp['year'].shift(1)
    is_consecutive = (df['year'] - prev_year) == 1
    
    metrics = ['medicare_payment', 'tot_benes', 'tot_srvcs', 'submitted_charges', 'pymt_per_bene']
    
    for m in metrics:
        prev_m = npi_grp[m].shift(1)
        
        # 1. Absolute Change: Delta = X_t - X_{t-1}
        delta_col = f"{m}_yoy_diff"
        df[delta_col] = np.where(is_consecutive, df[m] - prev_m, np.nan)
        
        # 2. Percentage Growth: Growth = Delta / (|X_{t-1}| + epsilon)
        growth_col = f"{m}_yoy_growth"
        denom = np.abs(prev_m) + (100.0 if 'payment' in m or 'charge' in m else 10.0)
        df[growth_col] = np.where(is_consecutive, (df[m] - prev_m) / denom, np.nan)
        
    # Flag new billing providers (first appearance in Medicare)
    df['is_new_billing_provider'] = np.where(prev_year.isna(), 1, 0).astype('int8')
    
    return df

# ----------------------------------------------------------------------
# 5. PIPELINE EXECUTION & EXPORT
# ----------------------------------------------------------------------
def main():
    files = sorted(glob.glob(str(CMS_DIR / "**" / "*_Prov.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CMS provider files found in {CMS_DIR}")
        
    print(f"Starting Feature Engineering on {len(files)} CMS Annual Files...")
    
    yearly_dfs = []
    for f in files:
        ydf = load_and_clean_year(f)
        yearly_dfs.append(ydf)
        gc.collect()
        
    master_df = pd.concat(yearly_dfs, ignore_index=True)
    del yearly_dfs
    gc.collect()
    
    print(f"\nTotal Individual Provider-Years: {len(master_df):,}")
    
    # Module 2: Base Ratios
    master_df = compute_base_ratios(master_df)
    
    # Module 3: Specialty Peer Z-scores
    master_df = compute_peer_zscores(master_df)
    
    # Module 4: Longitudinal Lags
    master_df = compute_longitudinal_features(master_df)
    
    # Downcast floats to float32 to save RAM for modeling
    float_cols = master_df.select_dtypes(include=['float64']).columns
    master_df[float_cols] = master_df[float_cols].astype('float32')
    master_df['npi'] = master_df['npi'].astype('string')
    master_df['year'] = master_df['year'].astype('int32')
    
    # Export to Parquet
    print(f"\n--> Exporting to {OUTPUT_FILE}...")
    master_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"\n[DONE] Successfully saved {len(master_df):,} rows to {OUTPUT_FILE} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
