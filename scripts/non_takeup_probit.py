from pathlib import Path
from statsmodels.iolib.summary2 import summary_col
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from misc.utility_functions import get_config_path, load_config

###############################################################################
# Helpers
###############################################################################


def plot_monthly_award_over_time(df: pd.DataFrame) -> None:
    """Plot the average monthly award (among recipients) over survey years."""
    # Filter for students who received BAföG
    received = df[df["eligible_for_bafoeg"] == 1].copy()

    # Compute mean award by survey year
    award_by_year = (
        received.groupby("syear")["monthly_award"]
        .mean()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(award_by_year["syear"], award_by_year["monthly_award"], marker="o")
    plt.xlabel("Survey Year")
    plt.ylabel("Average Monthly Award (€)")
    plt.title("Average BAföG Monthly Award Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_non_takeup_by_year(df: pd.DataFrame) -> None:
    # Filter for theoretically eligible students
    eligible = df[df["eligible_for_bafoeg"] == 1].copy()
    eligible["NTU"] = (eligible["receives_bafoeg"] == 0).astype(int)

    # Compute average NTU by year
    rate_by_year = (
        eligible.groupby("syear")["NTU"]
        .mean()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(rate_by_year["syear"], rate_by_year["NTU"], marker="o")
    plt.xlabel("Survey Year")
    plt.ylabel("Non-take-up Rate")
    plt.title("Non-take-up of BAföG Over Time")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _build_design_matrix(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build design matrix X and outcome NTU_i = 1{A_i = 0 | T_i = 1}.

    • Sample: restrict to theoretically eligible students (T_i = 1).
    • NTU_i = 1 if student did not receive BAföG (A_i = 0); 0 if received it.
    • Continuous covariates are z-standardized; categorical ones one-hot encoded.
    """

    df = df.loc[df["eligible_for_bafoeg"] == 1].copy()

    # ───────────────────── NTU indicator: 1 if A_i = 0 ────────────────────────
    NTU = (df["receives_bafoeg"] == 0).astype(int)
    NTU.name = "NTU"

    # ───────────────────────── continuous covariates ──────────────────────────
    continuous_cols = ["age", "num_siblings", "monthly_award"]
    scaler = StandardScaler()
    continuous_std = pd.DataFrame(
        scaler.fit_transform(df[continuous_cols]),
        columns=[f"z_{c}" for c in continuous_cols],
        index=df.index,
    )

    # ───────────────────────────── binary covariates ──────────────────────────
    binary = pd.DataFrame({
        "living_with_parent": df["lives_with_parent"].astype(int),
        "east_background": df["east_background"].astype(int),
    }, index=df.index)

    # ─────────────────────────── categorical dummies ──────────────────────────
    sex_dummies = pd.get_dummies(df["sex"].astype("category"), prefix="sex", drop_first=True)
    bula_dummies = pd.get_dummies(df["bula"].astype("category"), prefix="state", drop_first=True)
    migback_dummies = pd.get_dummies(df["migback"].astype("category"), prefix="migback", drop_first=True)

    df["pgemplst"] = pd.Categorical(
        df["pgemplst"],
        categories=[5, 1, 2, 3, 4],  # 5 = not employed (base)
        ordered=False
    )
    emp_dummies = pd.get_dummies(df["pgemplst"], prefix="empstat", drop_first=True)

    # ───────────────────────────── combine all features ───────────────────────
    X = pd.concat([continuous_std, binary, sex_dummies, emp_dummies, migback_dummies], axis=1)
    X = X.dropna()
    NTU = NTU.loc[X.index]

    # Add intercept
    X = sm.add_constant(X).astype(float)

    return NTU, X


def main(parquet_file: Path | None = None) -> None:
    # --------------------------- load data -----------------------------------
    if parquet_file is None:
        # fall back to the path defined in config.json
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

    # ------------------------ build model matrices ---------------------------
    NTU, X = _build_design_matrix(df)

    # ---------------------------- Probit estimation --------------------------
    model = sm.Probit(NTU, X)
    res = model.fit(disp=False, cov_type="HC2")
    print(res.summary())

    # ---------------------------- prediction ---------------------------------
    p_NTU = res.predict(X)  # predicted Pr(NTU_i = 1 | X_i)

    threshold = NTU.mean()  # ~ empirical prevalence
    # threshold = 0.5
    NTU_hat = (p_NTU >= threshold).astype(int)
    predicted_rate = NTU_hat.mean()

    print(f"\nSample non‑take‑up rate:          {NTU.mean():.2%} (proportion of students not taking up bafög even though eligible)")
    print(f"Predicted non‑take‑up rate (≥{threshold:.2f}): {predicted_rate:.2%} (the model's predicted rate using a threshold of 83.63% to classify non-take-up)")
    print(p_NTU)


    # Compute McFadden's pseudo-R²
    ll_full = res.llf
    ll_null = res.llnull
    r2_mcfadden = 1 - ll_full / ll_null

    print(f"\nMcFadden pseudo-R²: {r2_mcfadden:.4f}")

    # ---------------------------- marginal effects ---------------------------
    mfx = res.get_margeff()
    print(mfx.summary())

    # # ------------------------ Weighted GLM Probit ----------------------------
    # print("\nWeighted GLM Probit Model:")
    # weights = df.loc[NTU.index, "phrf"]
    # glm_model = sm.GLM(
    #     NTU,
    #     X,
    #     family=sm.families.Binomial(link=sm.families.links.probit()),
    #     weights=weights
    # )
    # glm_res = glm_model.fit()
    # print(glm_res.summary())

    # ----------------------- Plot NTU over years -----------------------------
    # plot_non_takeup_by_year(df)

    plot_monthly_award_over_time(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate non-take-up of BAföG and produce plots."
    )
    parser.add_argument(
        "-p", "--parquet-file",
        type=Path,
        metavar="FILE",
        help="Path to an alternative Parquet file. "
             "If omitted, the location from config.json is used."
    )
    args = parser.parse_args()

    # pass the user-supplied path (or None) into main()
    main(parquet_file=args.parquet_file)
