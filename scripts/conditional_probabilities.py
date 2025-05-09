from pathlib import Path
import pandas as pd
from misc.utility_functions import get_config_path, load_config

def main(parquet_file: Path | None = None) -> None:
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

    # Rename and filter
    df = df.rename(columns={
        "eligible_for_bafoeg": "T",
        "receives_bafoeg": "A",
    })
    df = df[df["T"].notna() & df["A"].notna()].copy()
    df["T"] = df["T"].astype(int)
    df["A"] = df["A"].astype(int)

    print("Conditional probabilities per year:")
    results = []

    for year, group in df.groupby("syear"):
        p_takeup = group.loc[group["T"] == 1, "A"].mean()
        p_non_takeup = 1 - p_takeup
        p_true_positive = group.loc[group["A"] == 1, "T"].mean()
        p_false_positive = group.loc[group["T"] == 0, "A"].mean()
        p_false_negative = 1 - p_true_positive

        results.append({
            "syear": year,
            "Pr(A=1|T=1)": round(p_takeup * 100, 2),
            "Pr(A=0|T=1)": round(p_non_takeup * 100, 2),
            "Pr(T=1|A=1)": round(p_true_positive * 100, 2),
            "Pr(T=0|A=1)": round(p_false_negative * 100, 2),
            "Pr(A=1|T=0)": round(p_false_positive * 100, 2),
        })

    result_df = pd.DataFrame(results).set_index("syear")
    print(result_df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-year conditional probabilities.")
    parser.add_argument("-p", "--parquet-file", type=Path, metavar="FILE")
    args = parser.parse_args()
    main(parquet_file=args.parquet_file)
