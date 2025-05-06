from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from misc.utility_functions import get_config_path, load_config

###############################################################################
# Helpers
###############################################################################

def _build_design_matrix(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build the design matrix **X** and dependent variable **y** for a Probit
    of *non‑take‑up* among BAföG‑eligible students.

    • Sample: keep *all* theoretically eligible students (`receives_bafoeg == 1`).
    • y = 1 ⇔ eligible but *did not* receive a grant; 0 ⇔ took‑up.
    • Continuous covariates are z‑standardised; categorical covariates are
      one‑hot encoded with the first level dropped.
    • Rare household‑type categories (fewer than 30 observations) are collapsed
      into an "other" bucket to avoid quasi‑separation.
    """

    # ────────────────────────────── sample ────────────────────────────────────
    df = df.loc[df["receives_bafoeg"] == 1].copy()

    # ───────────────────────── dependent variable ─────────────────────────────
    y = (df["received_student_grant"] == 0).astype(int)
    y.name = "non_takeup"

    # ────────────────────── collapse rare household types ─────────────────────
    hh_counts = df["hgtyp1hh"].value_counts()
    rare_hh = hh_counts[hh_counts < 30].index
    df.loc[df["hgtyp1hh"].isin(rare_hh), "hgtyp1hh"] = -1  # "other" category

    # ───────────────────────── continuous covariates ──────────────────────────
    continuous_cols = ["age", "num_siblings"]
    scaler = StandardScaler()
    continuous_std = pd.DataFrame(
        scaler.fit_transform(df[continuous_cols]),
        columns=[f"z_{c}" for c in continuous_cols],
        index=df.index,
    )

    # ───────────────────────────── binary covariate ───────────────────────────
    binary = pd.DataFrame(
        {"living_with_parent": df["lives_with_parent"].astype(int)}, index=df.index
    )

    # ─────────────────────────── categorical dummies ──────────────────────────
    sex_dummies = pd.get_dummies(df["sex"].astype("category"), prefix="sex", drop_first=True)
    bula_dummies = pd.get_dummies(df["bula"].astype("category"), prefix="state", drop_first=True)
    hh_dummies = pd.get_dummies(df["hgtyp1hh"].astype("category"), prefix="hh", drop_first=True)

    # ─────────────────────────────── combine & clean ──────────────────────────
    X = pd.concat([continuous_std, binary, sex_dummies, bula_dummies, hh_dummies], axis=1)
    X = X.dropna()
    y = y.loc[X.index]

    # Add constant (intercept)
    X = sm.add_constant(X).astype(float)

    return y, X

###############################################################################
# Script entry point
###############################################################################

def main() -> None:
    # --------------------------- load data -----------------------------------
    config_path = get_config_path(Path("config.json"))
    config = load_config(config_path)
    parquet_file = (
        Path(config["paths"]["results"]["dataframes"]).expanduser().resolve()
        / "eligibility.parquet"
    )
    df = pd.read_parquet(parquet_file)

    # ------------------------ build model matrices ---------------------------
    y, X = _build_design_matrix(df)

    # ---------------------------- estimation ---------------------------------
    model = sm.Probit(y, X)
    res = model.fit(disp=False)
    print(res.summary())

    # ----------------------- non‑take‑up prediction --------------------------
    p_hat = res.predict(X)

    # Use prevalence cut‑off to classify (better with imbalanced data)
    threshold = y.mean()
    non_takeup_hat = (p_hat >= threshold).astype(int)
    predicted_rate = non_takeup_hat.mean()

    print(f"\nSample non‑take‑up rate:          {y.mean():.2%}")
    print(f"Predicted non‑take‑up rate (≥{threshold:.2f}): {predicted_rate:.2%}")


if __name__ == "__main__":
    main()
