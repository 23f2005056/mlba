import os
import glob
import re
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "227664-V1"
OUTPUT_DIR = BASE_DIR / "out"
OUTPUT_FILE = OUTPUT_DIR / "cms_npi_year.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all CMS provider CSV files (recursively in yearly subfolders)
files = sorted(glob.glob(str(INPUT_DIR / "**" / "*_Prov.csv"), recursive=True))

if not files:
    # Fallback to any csv/parquet in INPUT_DIR
    files = sorted(glob.glob(str(INPUT_DIR / "**" / "*.*"), recursive=True))
    files = [f for f in files if f.endswith(('.csv', '.parquet'))]

print(f"Found {len(files)} CMS files to process.")

yearly_records = []

for file_path in files:
    filename = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    # Try to extract 4-digit year from directory or filename (e.g. 2013 or D13)
    match = re.search(r'(20\d\d)', file_path)
    if match:
        year = int(match.group(1))
    else:
        # Match D13 -> 2013, D23 -> 2023
        d_match = re.search(r'_D(\d{2})_', filename)
        if d_match:
            year = 2000 + int(d_match.group(1))
        else:
            print(f"Skipping {filename}: unable to determine year")
            continue

    print(f"--> Processing Year {year}: {filename}")

    try:
        if file_path.endswith('.parquet'):
            schema_cols = pd.read_parquet(file_path, engine='pyarrow').columns
            npi_col = next((c for c in schema_cols if 'npi' in c.lower() or 'rndrng' in c.lower()), None)
            df = pd.read_parquet(file_path, columns=[npi_col])
        else:
            header = pd.read_csv(file_path, nrows=0).columns
            # Prioritize Rndrng_NPI or exact NPI column
            npi_col = next((c for c in header if c.lower() in ['rndrng_npi', 'npi']), None)
            if not npi_col:
                npi_col = next((c for c in header if 'npi' in c.lower()), None)
            df = pd.read_csv(file_path, usecols=[npi_col], dtype={npi_col: str}, low_memory=False)
    except Exception as e:
        print(f"    Failed to read {filename}: {e}")
        continue

    # 1. Clean & format NPI: strip spaces, pad with zeros to exactly 10 digits
    npi_series = (
        df[npi_col]
        .astype(str)
        .str.strip()
        .str.replace(r'\.0$', '', regex=True) # remove .0 if read as float
        .str.zfill(10)
    )

    # 2. Filter out invalid/dummy NPIs (must be 10 digits, not 0000000000)
    valid_mask = npi_series.str.match(r'^\d{10}$') & (npi_series != '0000000000')
    npi_clean = npi_series[valid_mask].drop_duplicates()

    # 3. Create single-year dataframe
    sub_df = pd.DataFrame({
        'npi': npi_clean,
        'year': year
    })

    print(f"    Extracted {len(sub_df):,} unique NPIs for {year}")
    yearly_records.append(sub_df)

# Combine and save
if yearly_records:
    final_df = pd.concat(yearly_records, ignore_index=True)
    # Ensure one unique row per NPI per year
    final_df = final_df.drop_duplicates(subset=['npi', 'year'])
    final_df['year'] = final_df['year'].astype('int32')
    final_df['npi'] = final_df['npi'].astype('string')
    
    # Save to parquet
    final_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    print(f"\n[DONE] Saved {len(final_df):,} rows to {OUTPUT_FILE}")
else:
    print("\n[ERROR] No valid data extracted. Check input paths/filenames.")
