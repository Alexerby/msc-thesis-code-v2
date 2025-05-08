from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from misc.utility_functions import get_config_path, load_config

# CONFIGURATION FLAGS
PLOT_BY_YEAR = True          # Set to False to pool all years together
INCLUDE_SHARE_PLOT = True    # Plot (award / total_need)


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

    # ---------------------- Filter sample ----------------------
    df = df[df["eligible_for_bafoeg"] == 1].copy()
    df = df[df["monthly_award"].notna()]
    df = df[df["monthly_award"] > 0]

    # ---------------------- PDF and CDF plots ----------------------
    plt.figure(figsize=(14, 5))

    if PLOT_BY_YEAR:
        years = [2005, 2010, 2016, 2022]
        for year in years:
            subset = df[df["syear"] == year]
            sns.kdeplot(subset["monthly_award"], label=str(year))
            # plt.xlim(left=0)
            plt.xlim(0,1100)
        plt.title("PDF of Monthly BAföG Awards by Survey Year")
        plt.xlabel("Monthly Award (€)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 5))
        for year in years:
            subset = df[df["syear"] == year]
            sns.ecdfplot(subset["monthly_award"], label=str(year))
        plt.title("CDF of Monthly BAföG Awards by Survey Year")
        plt.xlabel("Monthly Award (€)")
        plt.ylabel("Cumulative Proportion")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        plt.subplot(1, 2, 1)
        sns.histplot(df["monthly_award"], bins=30, kde=True, stat="density", color="skyblue", edgecolor="black")
        plt.title("PDF: Distribution of BAföG Award Amounts")
        plt.xlabel("Monthly Award (€)")
        plt.ylabel("Density")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sns.histplot(df["monthly_award"], bins=200, cumulative=True, stat="density", element="step", fill=False)
        plt.title("CDF: Cumulative Distribution of BAföG Award Amounts")
        plt.xlabel("Monthly Award (€)")
        plt.ylabel("Cumulative Proportion")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # ---------------------- Plot award share ----------------------
    if INCLUDE_SHARE_PLOT and "total_need" in df:
        df["award_share"] = df["monthly_award"] / df["total_need"]
        df = df[df["award_share"].between(0, 1.5)]

        plt.figure(figsize=(6, 5))
        sns.histplot(df["award_share"], bins=30, kde=True, color="seagreen", edgecolor="black")
        plt.title("Distribution of Award / Total Need")
        plt.xlabel("Share of Need Covered by BAföG")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot distribution of BAföG awards among recipients."
    )
    parser.add_argument(
        "-p", "--parquet-file",
        type=Path,
        metavar="FILE",
        help="Path to Parquet file. If omitted, use config.json default."
    )
    args = parser.parse_args()
    main(parquet_file=args.parquet_file)
