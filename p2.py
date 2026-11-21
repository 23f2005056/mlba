
"""
FraudLens - Phase 2 validation checks

Checks:
1. Duplicate NPI-year records in the raw CMS files
2. NPI format validation (10 digits + Luhn check)
3. LEIE historical completeness (survivorship) + coverage of CMS
4. The five selected exclusion codes: do they exist, how common are they
5. Entity type: are positives lost because we kept only individuals?
6. Billing AFTER exclusion (sanity check on label timing)

Run:
    python p2.py

No pandas dependency required.
"""

from pathlib import Path

import duckdb


# ============================================================
# 1. CONFIGURATION
# ============================================================

CMS_FOLDER = Path(r"C:\Users\DELL\Downloads\cms\227664-V1")
LEIE_CSV = Path(r"C:\Users\DELL\Downloads\cms\UPDATED (1).csv")

YEARS = list(range(2016, 2023))
WINDOW = 3

FRAUD_TYPES = [
    "1128a1",
    "1128a3",
    "1128b1",
    "1128b6",
    "1128b7"
]


# What OIG's exclusion-authorities page says each code means.

CODE_MEANING = {
    "1128a1": "Mandatory: conviction of program-related crimes",
    "1128a2": "Mandatory: patient abuse or neglect",
    "1128a3": "Mandatory: felony conviction, health care fraud",
    "1128a4": "Mandatory: felony conviction, controlled substance",
    "1128b1": "Permissive: misdemeanor health care fraud (b1A) / fraud in non-health-care programs (b1B)",
    "1128b2": "Permissive: obstruction of investigation or audit",
    "1128b3": "Permissive: misdemeanor, controlled substance",
    "1128b4": "Permissive: license revocation/suspension/surrender",
    "1128b5": "Permissive: excluded/suspended under other program",
    "1128b6": "Permissive: excessive charges / unnecessary services",
    "1128b7": "Permissive: fraud, kickbacks, other prohibited activities",
    "1128b8": "Permissive: entities controlled by sanctioned individual",
    "1128b15": "Permissive: individuals controlling a sanctioned entity",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def sq(path):
    """Convert a Windows path into a SQL-safe string."""

    return (
        "'" +
        str(Path(path).resolve())
        .replace("\\", "/")
        .replace("'", "''")
        + "'"
    )


def banner(title):
    """Print a section heading."""

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def print_table(headers, rows):
    """
    Print SQL query results as a formatted table.

    Uses ordinary Python instead of pandas.
    """

    if not rows:
        print("No records found.")
        return

    rows = [
        [
            str(value) if value is not None else "NULL"
            for value in row
        ]
        for row in rows
    ]

    widths = [
        max(
            len(str(headers[i])),
            max(len(row[i]) for row in rows)
        )
        for i in range(len(headers))
    ]

    header_line = " | ".join(
        str(headers[i]).ljust(widths[i])
        for i in range(len(headers))
    )

    separator = "-+-".join(
        "-" * width
        for width in widths
    )

    print(header_line)
    print(separator)

    for row in rows:
        print(
            " | ".join(
                row[i].ljust(widths[i])
                for i in range(len(headers))
            )
        )


def luhn_npi_ok(npi: str) -> bool:
    """
    Validate NPI using the Luhn check-digit algorithm.

    The standard healthcare identifier prefix is 80840.
    """

    if len(npi) != 10 or not npi.isdigit():
        return False

    digits = [int(c) for c in "80840" + npi]

    total = 0

    for i, d in enumerate(reversed(digits)):

        if i % 2 == 1:
            d *= 2

            if d > 9:
                d -= 9

        total += d

    return total % 10 == 0


# ============================================================
# 3. INITIALISE DUCKDB
# ============================================================

con = duckdb.connect(":memory:")


# ============================================================
# 4. LOAD CMS AND LEIE DATA
# ============================================================

banner("LOADING ALL ENTITY TYPES - RAW CMS DATA")

print("Loading CMS files for 2016-2022...")

first = True
loaded_years = []

for year in YEARS:

    files = sorted(
        (CMS_FOLDER / str(year)).glob("*.csv")
    )

    if not files:
        print(f"WARNING: No CMS file found for {year}")
        continue

    # Use the first CSV file found for that year.
    cms_file = files[0]

    query = f"""
        SELECT
            {year} AS cms_year,

            trim(
                CAST(Rndrng_NPI AS VARCHAR)
            ) AS npi_raw,

            upper(
                trim(CAST(Rndrng_Prvdr_Ent_Cd AS VARCHAR))
            ) AS ent,

            trim(
                CAST(Rndrng_Prvdr_Type AS VARCHAR)
            ) AS provider_type

        FROM read_csv_auto(
            {sq(cms_file)},
            header = true,
            all_varchar = true
        )
    """

    if first:

        con.execute(
            "CREATE TABLE billing AS " + query
        )

        first = False

    else:

        con.execute(
            "INSERT INTO billing " + query
        )

    loaded_years.append(year)

    print(f"Loaded CMS {year}: {cms_file.name}")


if not loaded_years:
    raise FileNotFoundError(
        f"No CMS CSV files found under {CMS_FOLDER}"
    )


# Load OIG LEIE data.

banner("LOADING OIG LEIE DATA")

if not LEIE_CSV.exists():
    raise FileNotFoundError(
        f"LEIE file not found: {LEIE_CSV}"
    )


con.execute(
    f"""
    CREATE TABLE leie AS

    SELECT

        trim(
            CAST(NPI AS VARCHAR)
        ) AS npi_raw,

        lower(
            trim(CAST(EXCLTYPE AS VARCHAR))
        ) AS excltype,

        try_cast(
            substr(
                trim(CAST(EXCLDATE AS VARCHAR)),
                1,
                4
            ) AS INTEGER
        ) AS excl_year,

        trim(
            CAST(REINDATE AS VARCHAR)
        ) AS reindate

    FROM read_csv_auto(
        {sq(LEIE_CSV)},
        header = true,
        all_varchar = true
    )
    """
)


cms_count = con.execute(
    "SELECT COUNT(*) FROM billing"
).fetchone()[0]

leie_count = con.execute(
    "SELECT COUNT(*) FROM leie"
).fetchone()[0]

print(f"CMS rows:  {cms_count:,}")
print(f"LEIE rows: {leie_count:,}")


# ============================================================
# 5. DUPLICATE NPI-YEAR RECORDS
# ============================================================

banner("1. DUPLICATE NPI-YEAR RECORDS (RAW CMS)")


duplicate_count = con.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            npi_raw,
            cms_year

        FROM billing

        GROUP BY 1, 2

        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]


print(
    f"NPI-year pairs appearing more than once: "
    f"{duplicate_count:,}"
)


if duplicate_count:

    duplicate_rows = con.execute(
        """
        SELECT
            cms_year,
            COUNT(*) AS duplicated_npis

        FROM (
            SELECT
                npi_raw,
                cms_year

            FROM billing

            GROUP BY 1, 2

            HAVING COUNT(*) > 1
        )

        GROUP BY 1

        ORDER BY 1
        """
    ).fetchall()

    print_table(
        ["CMS Year", "Duplicated NPIs"],
        duplicate_rows
    )

    print(
        "-> Investigate before modelling. "
        "The previous scripts used SELECT DISTINCT, "
        "which could hide some duplicates."
    )

else:

    print("OK: One row per NPI per year.")


# ============================================================
# 6. NPI FORMAT VALIDATION
# ============================================================

banner("2. NPI FORMAT VALIDATION")


# CMS NPI validation.

cms_npis = [
    r[0]
    for r in con.execute(
        """
        SELECT DISTINCT npi_raw
        FROM billing
        WHERE npi_raw IS NOT NULL
        """
    ).fetchall()
]


bad_format = [
    n for n in cms_npis
    if not (len(n) == 10 and n.isdigit())
]


bad_luhn = [
    n for n in cms_npis
    if len(n) == 10
    and n.isdigit()
    and not luhn_npi_ok(n)
]


print(f"CMS distinct NPIs:            {len(cms_npis):,}")

print(
    f"  Not 10 digits:              {len(bad_format):,} "
    f"  Examples: {bad_format[:3]}"
)

print(
    f"  10 digits, fail Luhn check: {len(bad_luhn):,} "
    f"  Examples: {bad_luhn[:3]}"
)


# LEIE NPI validation.

leie_npis = [
    r[0]
    for r in con.execute(
        """
        SELECT DISTINCT npi_raw
        FROM leie
        WHERE npi_raw IS NOT NULL
        """
    ).fetchall()
]


placeholder = [
    n for n in leie_npis
    if n in ("", "0000000000")
]


real = [
    n for n in leie_npis
    if n not in ("", "0000000000")
]


l_bad_fmt = [
    n for n in real
    if not (len(n) == 10 and n.isdigit())
]


l_bad_luhn = [
    n for n in real
    if len(n) == 10
    and n.isdigit()
    and not luhn_npi_ok(n)
]


print(f"\nLEIE distinct NPI values:     {len(leie_npis):,}")
print(f"  Placeholder / empty:        {len(placeholder):,}")
print(f"  Real NPIs:                  {len(real):,}")

print(
    f"  Real but not 10 digits:     {len(l_bad_fmt):,} "
    f" Examples: {l_bad_fmt[:3]}"
)

print(
    f"  Real, fail Luhn check:      {len(l_bad_luhn):,} "
    f" Examples: {l_bad_luhn[:3]}"
)


# ============================================================
# 7. LEIE HISTORICAL COMPLETENESS
# ============================================================

banner("3. LEIE HISTORICAL COMPLETENESS")


print(
    "Fields in the current file only describe exclusions "
    "still in effect."
)

print(
    "Reinstated people are removed, so older years "
    "can be undercounted.\n"
)


inlist = ",".join(
    f"'{t}'"
    for t in FRAUD_TYPES
)


leie_year_rows = con.execute(
    f"""
    SELECT

        excl_year,

        COUNT(*) AS all_rows,

        SUM(
            CASE
                WHEN npi_raw NOT IN ('', '0000000000')
                THEN 1
                ELSE 0
            END
        ) AS with_npi,

        SUM(
            CASE
                WHEN excltype IN ({inlist})
                 AND npi_raw NOT IN ('', '0000000000')
                THEN 1
                ELSE 0
            END
        ) AS fraud_types_with_npi

    FROM leie

    WHERE excl_year BETWEEN 2010 AND 2025

    GROUP BY 1

    ORDER BY 1
    """
).fetchall()


print_table(
    [
        "Exclusion Year",
        "All Rows",
        "With NPI",
        "Selected Types with NPI"
    ],
    leie_year_rows
)


print(
    "\nHow to read: A steady decline going back in time "
    "(for example, 2013 much lower than 2022) "
    "suggests reinstatements or expiry have removed "
    "older records."
)


# ------------------------------------------------------------
# LEIE PROVIDERS THAT ALSO APPEAR IN CMS
# ------------------------------------------------------------

con.execute(
    """
    CREATE TABLE leie_real AS

    SELECT *

    FROM leie

    WHERE npi_raw NOT IN ('', '0000000000')
    """
)


in_cms = con.execute(
    """
    SELECT COUNT(DISTINCT l.npi_raw)

    FROM leie_real l

    JOIN (
        SELECT DISTINCT npi_raw
        FROM billing
    ) b

    ON b.npi_raw = l.npi_raw
    """
).fetchone()[0]


print(
    f"\nLEIE real-NPI providers that appear in CMS "
    f"2016-2022 at all: {in_cms:,} of {len(real):,} "
    f"({in_cms / max(len(real), 1):.1%})"
)


print(
    "(The rest were never billing Part B in these years, "
    "were excluded before 2016, or bill under other programs.)"
)


# ============================================================
# 8. THE FIVE SELECTED EXCLUSION CODES
# ============================================================

banner("4. THE FIVE SELECTED EXCLUSION CODES")


print(
    "Codes found in your file (top 25) vs. OIG meaning:\n"
)


codes = con.execute(
    """
    SELECT

        excltype,

        COUNT(*) AS n,

        SUM(
            CASE
                WHEN npi_raw NOT IN ('', '0000000000')
                THEN 1
                ELSE 0
            END
        ) AS with_npi

    FROM leie

    GROUP BY 1

    ORDER BY 2 DESC

    LIMIT 25
    """
).fetchall()


found_codes = set()


for code, n, with_npi in codes:

    found_codes.add(code)

    mark = (
        "<== SELECTED"
        if code in FRAUD_TYPES
        else ""
    )

    meaning = CODE_MEANING.get(
        code,
        "(not in my lookup - check OIG page)"
    )

    print(
        f"  {code:10s} "
        f"n={n:>6,}  "
        f"with_npi={with_npi:>5,}  "
        f"{meaning:70s} "
        f"{mark}"
    )


missing = [
    c for c in FRAUD_TYPES
    if c not in found_codes
]


print(
    f"\nSelected codes missing from the file's top 25: "
    f"{missing or 'none'}"
)


print(
    """
Notes (verified against OIG's exclusion-authorities page):

  * 1128a1 = program-related crimes (broader than billing fraud)
  * 1128a3 = felony health care fraud (strongest signal)
  * 1128b1 = misdemeanour HC fraud AND non-HC fraud
             (LEIE doesn't split b1A/b1B)
  * 1128b6 = excessive charges / unnecessary services
  * 1128b7 = fraud, kickbacks, other prohibited activities

  => Call the target "future fraud-related exclusion",
     not "fraud".
    """
)


# ============================================================
# 9. ENTITY TYPE - INDIVIDUALS VS ORGANISATIONS
# ============================================================

banner(
    "5. ENTITY TYPE - DID 'INDIVIDUALS ONLY' DISCARD POSITIVES?"
)


con.execute(
    f"""
    CREATE TABLE pos_all AS

    SELECT DISTINCT

        b.cms_year,

        b.npi_raw AS npi,

        b.ent

    FROM billing b

    JOIN leie_real l

        ON b.npi_raw = l.npi_raw

    WHERE l.excl_year > b.cms_year

      AND l.excl_year <= b.cms_year + {WINDOW}

      AND l.excltype IN ({inlist})
    """
)


print(
    "Scenario A positives (NPI match, 5 fraud types) "
    "by entity type:"
)


entity_rows = con.execute(
    """
    SELECT

        ent,

        COUNT(*) AS positive_provider_years,

        COUNT(DISTINCT npi) AS unique_npis

    FROM pos_all

    GROUP BY 1

    ORDER BY 2 DESC
    """
).fetchall()


print_table(
    [
        "Entity Type",
        "Positive Provider-Years",
        "Unique NPIs"
    ],
    entity_rows
)


print(
    "\nI = individual, O = organisation. "
    "If 'O' has many positives, the earlier scripts "
    "(individuals only) left them out. Decide deliberately "
    "whether to model individuals only or both."
)


# ------------------------------------------------------------
# ALL-ENTITY POSITIVES BY BILLING YEAR
# ------------------------------------------------------------

print("\nAll-entity positives per billing year:")


year_positive_rows = con.execute(
    """
    SELECT

        cms_year,

        COUNT(*) AS positives

    FROM pos_all

    GROUP BY 1

    ORDER BY 1
    """
).fetchall()


print_table(
    [
        "CMS Year",
        "Positive Provider-Years"
    ],
    year_positive_rows
)


# ------------------------------------------------------------
# TOP PROVIDER TYPES AMONG POSITIVES
# ------------------------------------------------------------

print("\nTop provider types among positives:")


provider_type_rows = con.execute(
    f"""
    SELECT

        b.provider_type,

        COUNT(DISTINCT p.npi) AS unique_npis

    FROM pos_all p

    JOIN (
        SELECT DISTINCT
            npi_raw,
            provider_type

        FROM billing
    ) b

        ON b.npi_raw = p.npi

    GROUP BY 1

    ORDER BY 2 DESC

    LIMIT 12
    """
).fetchall()


print_table(
    [
        "Provider Type",
        "Unique Positive NPIs"
    ],
    provider_type_rows
)


# ============================================================
# 10. BILLING AFTER EXCLUSION
# ============================================================

banner("6. BILLING AFTER EXCLUSION (SANITY CHECK)")


after = con.execute(
    f"""
    SELECT COUNT(
        DISTINCT b.npi_raw || '-' || b.cms_year
    )

    FROM billing b

    JOIN leie_real l

        ON b.npi_raw = l.npi_raw

    WHERE l.excltype IN ({inlist})

      AND b.cms_year > l.excl_year
    """
).fetchone()[0]


print(
    f"Provider-years billed in a year AFTER "
    f"the exclusion year: {after:,}"
)


print(
    "A few are expected (exclusion date vs. data timing; "
    "waivers). A large number means exclusion dates or "
    "NPI matches need checking."
)


# ============================================================
# 11. COMPLETION
# ============================================================

banner("VALIDATION COMPLETE")


print(
    """
All six Phase 2 validation checks have finished.

IMPORTANT:
These checks assess data quality and label feasibility.
They do not establish that a provider committed fraud.
    """
)


con.close()
