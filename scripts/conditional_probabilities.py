from pathlib import Path
import pandas as pd

from misc.utility_functions import get_config_path, load_config


def main(parquet_file: Path | None = None) -> None:
    # ---------------------- Load data ----------------------
    if parquet_file is None:
        config_path = get_config_path(Path("config.json"))
        config = load_config(config_path)
        parquet_file = (
            Path(config["paths"]["results"]["dataframes"])
            .expanduser().resolve()
            / "eligibility.parquet"
        )
    else:
        parquet_file = parquet_file.expanduser().resolve()

    df = pd.read_parquet(parquet_file)

    # Rename for clarity
    df = df.rename(columns={
        "eligible_for_bafoeg": "T",             # Theoretically eligible
        "receives_bafoeg": "A",          # Actually received BAföG
    })

    # Ensure both are binary integers
    df = df[df["T"].notna() & df["A"].notna()].copy()
    df["T"] = df["T"].astype(int)
    df["A"] = df["A"].astype(int)

    # ---------------------- Raw Probabilities ----------------------
    # Pr(A = 1 | T = 1)
    takeup_rate = df.loc[df["T"] == 1, "A"].mean()

    # Pr(A = 0 | T = 1)
    non_takeup_rate = 1 - takeup_rate

    # Pr(T = 1 | A = 1)
    share_eligible_among_recipients = df.loc[df["A"] == 1, "T"].mean()

    # Pr(T = 0 | A = 1)
    share_ineligible_among_recipients = 1 - share_eligible_among_recipients


    false_positive_rate = df.loc[df["T"] == 0, "A"].mean()

    print(f"\nEmpirical Conditional Probabilities:")
    print(f"  Pr(A = 1 | T = 1) = {takeup_rate:.2%}")
    print(f"  Pr(A = 0 | T = 1) = {non_takeup_rate:.2%}")
    print(f"  Pr(T = 1 | A = 1) = {share_eligible_among_recipients:.2%}")
    print(f"  Pr(T = 0 | A = 1) = {share_ineligible_among_recipients:.2%}")
    print(f"  Pr(A = 1 | T = 0) = {false_positive_rate:.2%}")

    # Optional: output total counts
    print("\nCounts:")
    print(pd.crosstab(df["T"], df["A"], rownames=["Eligible (T)"], colnames=["Receives (A)"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute raw conditional probabilities between theoretical eligibility and actual BAföG receipt."
    )
    parser.add_argument(
        "-p", "--parquet-file",
        type=Path,
        metavar="FILE",
        help="Path to Parquet file. If omitted, use config.json default."
    )
    args = parser.parse_args()
    main(parquet_file=args.parquet_file)
