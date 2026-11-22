"""
FraudLens: Top Provider Interpretability & Case Study Extraction Utility
Author: Person 2 (CMS Data & Feature Engineering)
Deliverable: CP-14 Interpretability Support

Purpose:
Generates factual, peer-benchmarked audit dossiers for Top-K high-risk providers
selected by P3/P4 models, providing context without unsupported fraud claims.

Usage:
    # 1. Run demo on top billing providers:
    python src/explain_top_providers.py

    # 2. Python API in P4 evaluation script:
    from src.explain_top_providers import explain_provider_risk
    dossier = explain_provider_risk(npi="1234567890", year=2023)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out"

def explain_provider_risk(npi: str, year: int = 2023, feat_df: pd.DataFrame = None):
    """
    Extracts provider features and compares against specialty peer benchmarks.
    """
    if feat_df is None:
        feat_path = OUT_DIR / "provider_features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found at {feat_path}")
        feat_df = pd.read_parquet(feat_path)
        
    npi_clean = str(npi).strip().zfill(10)
    row = feat_df[(feat_df['npi'] == npi_clean) & (feat_df['year'] == int(year))]
    
    if row.empty:
        return f"Provider {npi_clean} not found in reporting year {year}."
        
    r = row.iloc[0]
    spec = r['specialty']
    
    # Compute specialty benchmarks for this specific year
    spec_peers = feat_df[(feat_df['year'] == int(year)) & (feat_df['specialty'] == spec)]
    n_peers = len(spec_peers)
    
    metrics = [
        ('medicare_payment', 'Total Medicare Payment ($)', True),
        ('tot_srvcs', 'Total Services Rendered', False),
        ('tot_benes', 'Distinct Beneficiaries', False),
        ('pymt_per_bene', 'Payment per Beneficiary ($)', True),
        ('srvcs_per_bene', 'Services per Beneficiary', False),
        ('charge_to_allowed', 'Charge-to-Allowed Markup', False),
    ]
    
    lines = [
        f"=================================================================",
        f"AUDIT INTERPRETABILITY DOSSIER: NPI {npi_clean} (Year: {year})",
        f"Specialty: {spec} | Peer Group Size: {n_peers:,} individual physicians",
        f"=================================================================",
        f"{'Metric':<32} | {'Provider':<12} | {'Peer Median':<12} | {'Peer IQR':<10} | {'Robust Z':<10}",
        f"---------------------------------------------------------------------------------------------------"
    ]
    
    for col, label, is_curr in metrics:
        val = r.get(col, np.nan)
        z_col = f"{col}_peer_z"
        z_val = r.get(z_col, np.nan)
        
        peer_med = spec_peers[col].median()
        peer_iqr = spec_peers[col].quantile(0.75) - spec_peers[col].quantile(0.25)
        
        val_str = f"${val:,.2f}" if is_curr else f"{val:,.1f}"
        med_str = f"${peer_med:,.2f}" if is_curr else f"{peer_med:,.1f}"
        iqr_str = f"${peer_iqr:,.2f}" if is_curr else f"{peer_iqr:,.1f}"
        z_str = f"{z_val:+.2f}" if pd.notna(z_val) else "N/A"
        
        lines.append(f"{label:<32} | {val_str:<12} | {med_str:<12} | {iqr_str:<10} | {z_str:<10}")
        
    lines.append(f"---------------------------------------------------------------------------------------------------")
    lines.append("LONGITUDINAL YEAR-OVER-YEAR (YoY) DYNAMICS:")
    
    yoy_metrics = [
        ('medicare_payment', 'Payment Growth'),
        ('tot_srvcs', 'Services Growth'),
        ('tot_benes', 'Beneficiary Growth')
    ]
    
    for col, lbl in yoy_metrics:
        diff = r.get(f"{col}_yoy_diff", np.nan)
        growth = r.get(f"{col}_yoy_growth", np.nan)
        diff_str = f"{diff:+,.1f}" if pd.notna(diff) else "N/A"
        growth_str = f"{growth * 100:+.1f}%" if pd.notna(growth) else "N/A (Baseline/Gap)"
        lines.append(f"  • {lbl:<20}: Change: {diff_str:<12} | Velocity: {growth_str}")
        
    lines.append(f"=================================================================\n")
    return "\n".join(lines)

def main():
    feat_path = OUT_DIR / "provider_features.parquet"
    if not feat_path.exists():
        print(f"Error: {feat_path} not found.")
        sys.exit(1)
        
    print("Loading features to demo top provider interpretability...")
    df = pd.read_parquet(feat_path)
    
    # Pick a sample high payment provider in 2023 for demo
    sample_p = df[df['year'] == 2023].sort_values(by='medicare_payment', ascending=False).iloc[0]
    sample_npi = sample_p['npi']
    
    dossier = explain_provider_risk(sample_npi, 2023, feat_df=df)
    print(dossier)

if __name__ == "__main__":
    main()
