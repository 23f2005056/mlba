
from pathlib import Path

# ============================================
# PROJECT ROOT
# ============================================

ROOT = Path(__file__).resolve().parent

# ============================================
# DATA DIRECTORIES
# ============================================

RAW = ROOT / "raw"
OUT = ROOT / "out"
REPORTS = ROOT / "reports"

LEIE_MAIN = RAW / "leie" / "UPDATED.csv"
REIN_DIR = RAW / "leie" / "rein"

# Original CMS dataset
CMS_SOURCE_DIR = ROOT / "227664-V1"

# Available CMS years based on downloaded folders.
CMS_YEARS_AVAILABLE = list(range(2013, 2024))

# Actual modelling years will be frozen after
# P2 confirms compatible annual files.
YEARS = None

# ============================================
# LABEL SETTINGS
# ============================================

HORIZONS = [1, 2, 3]

# Latest LEIE reporting cutoff must be verified
# against the actual file before label construction.
LEIE_CUTOFF = None

# Scenario A codes will be confirmed after
# inspecting the actual LEIE exclusion categories.
SCENARIO_A_TYPES = set()

# Scenario B includes all valid exclusion types.

# ============================================
# CREATE OUTPUT DIRECTORIES
# ============================================

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

if not LEIE_MAIN.exists():
    raise FileNotFoundError(
        f"LEIE file not found: {LEIE_MAIN}"
    )

if not CMS_SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"CMS directory not found: {CMS_SOURCE_DIR}"
    )

print("FraudLens configuration loaded.")
print("Project root:", ROOT)
print("CMS available years:", CMS_YEARS_AVAILABLE)
print("LEIE file:", LEIE_MAIN)
