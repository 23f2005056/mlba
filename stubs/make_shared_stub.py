"""Generate small, entirely synthetic P1/P2 inputs for P3 and P4."""

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
YEARS = (2016, 2017, 2018, 2020, 2021, 2022)
PROVIDERS = (
    # NPI, specialty, base payment, beneficiaries, services, exclusion year, scenario
    ("9000000001", "Cardiology", 800_000, 400, 2_000, 2019, "A"),
    ("9000000002", "Cardiology", 1_300_000, 280, 2_100, 2024, "A"),
    ("9000000003", "Internal Medicine", 570_000, 650, 3_200, 2025, "A"),
    ("9000000004", "Orthopedics", 1_100_000, 520, 1_400, 2023, "B_only"),
    ("9000000005", "Internal Medicine", 420_000, 700, 3_600, 2021, "A"),
    ("9000000006", "Cardiology", 1_900_000, 900, 4_500, None, None),
    ("9000000007", "Orthopedics", 300_000, 210, 950, None, None),
    ("9000000008", "Internal Medicine", 960_000, 1_100, 5_200, None, None),
)


def main() -> None:
    features = []
    candidates = []
    labels = []

    for npi, specialty, base_payment, base_beneficiaries, base_services, exclusion_year, scenario in PROVIDERS:
        provider_offset = int(npi[-1])
        for year in YEARS:
            step = year - 2016
            total_payment = round(base_payment * (1 + 0.035 * step + 0.004 * provider_offset), 2)
            beneficiaries = base_beneficiaries + 7 * step + provider_offset
            services = base_services + 18 * step + 3 * provider_offset
            features.append(
                {
                    "NPI": npi,
                    "year": year,
                    "specialty": specialty,
                    "total_payment": total_payment,
                    "beneficiaries": beneficiaries,
                    "services": services,
                    "payment_per_beneficiary": round(total_payment / beneficiaries, 2),
                    "services_per_beneficiary": round(services / beneficiaries, 4),
                }
            )

            # An exclusion during or before the decision year removes the provider.
            if exclusion_year is not None and exclusion_year <= year:
                continue

            candidates.append({"NPI": npi, "year": year})
            future_exclusion = exclusion_year is not None and year < exclusion_year <= year + 3
            scenario_a = int(future_exclusion and scenario == "A")
            scenario_b = int(future_exclusion and scenario in {"A", "B_only"})
            labels.append(
                {
                    "NPI": npi,
                    "year": year,
                    "label_scenario_a": scenario_a,
                    "label_scenario_b": scenario_b,
                }
            )

    for filename, rows in (
        ("provider_features.parquet", features),
        ("candidate_pool.parquet", candidates),
        ("labels.parquet", labels),
    ):
        frame = pd.DataFrame(rows).sort_values(["year", "NPI"]).reset_index(drop=True)
        frame["NPI"] = frame["NPI"].astype("string")
        frame["year"] = frame["year"].astype("int64")
        frame.to_parquet(OUT / filename, index=False)
        print(f"{filename}: {len(frame)} rows")


if __name__ == "__main__":
    main()
