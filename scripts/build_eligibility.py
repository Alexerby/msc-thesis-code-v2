from bafoeg_calculator import BafoegCalculator
from misc.utility_functions import export_parquet

def main(overwrite: bool = False):
    calc = BafoegCalculator()
    tables = calc.run()

    # Select relevant columns from the full table
    required_cols = [
        "pid", "syear", "monthly_award", "receives_bafoeg", "lives_with_parent",
        "age", "sex", "bula", "num_siblings", "hgtyp1hh",
        "pgemplst", "migback", "east_background", "phrf", "eligible_for_bafoeg",
    ]

    # Filter the full table to only valid rows (e.g., students)
    df = tables["full"].copy()

    # Drop if any required variable is missing
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in BafoegCalculator output: {missing}")

    df = df[required_cols].dropna(subset=["monthly_award"])

    path = export_parquet(df, "eligibility")
    print(f"✅ Eligibility file written → {path}")

if __name__ == "__main__":
    main()
