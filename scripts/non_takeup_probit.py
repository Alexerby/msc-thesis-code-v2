from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from misc.utility_functions import get_config_path, load_config

# Model:
# Pr(no actual take-up∣theoretical eligibility)=Pr(A=0∣T=1)

# TODO:
# Currently covariance type non-robust
# 

###############################################################################
# Helpers
###############################################################################

def _build_design_matrix(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build the design matrix X and dependent variable y for a Probit
    of non‑take‑up among BAföG‑eligible students.

    • Sample: keep all theoretically eligible students (`receives_bafoeg == 1`).
    • y = 1 ⇔ eligible but did not receive a grant; 0 ⇔ took‑up.
    • Continuous covariates are z‑standardised; categorical covariates are
      one‑hot encoded with the first level dropped.
    """

    # ────────────────────────────── sample ────────────────────────────────────
    # Sample for our condition that T = 1 (theoretically eligible)
    # We are modeling Pr(A = 0 | T = 1), i.e. the probability of *not* taking up BAföG
    # among those who are eligible according to our calculations.
    df = df.loc[df["receives_bafoeg"] == 1].copy()

    # ───────────────────────── dependent variable ─────────────────────────────
    # Define dependent variable:
    # This is our actual observed outcome from SOEP data.
    # y = 1 if the student *did not* receive BAföG (i.e. non-take-up)
    #     = 0 if they did receive it (i.e. take-up)
    # This sets the left-hand side of our Probit model: Pr(A = 0 | T = 1)
    y = (df["received_student_grant"] == 0).astype(int)
    y.name = "non_takeup"

    # ───────────────────────── continuous covariates ──────────────────────────
    # Although 'num_siblings' is a discrete count variable, we treat it as continuous
    # here for simplicity in the Probit model. This is common practice when the variable
    # takes a reasonable range of values and no strong nonlinearity is expected.
    continuous_cols = ["age", "num_siblings"]
    # StandardScaler standardizes variables to have mean 0 and standard deviation 1.
    # This helps with model convergence and makes the scale of coefficients comparable.
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
    # For each categorical variable, we use one-hot encoding (get_dummies).
    # We set drop_first=True to avoid the "dummy variable trap":
    # If all dummy variables are included along with a constant term, they
    # become linearly dependent (perfect multicollinearity), which breaks estimation.
    # Dropping the first category avoids this by using it as the reference group.
    sex_dummies = pd.get_dummies(df["sex"].astype("category"), prefix="sex", drop_first=True)
    bula_dummies = pd.get_dummies(df["bula"].astype("category"), prefix="state", drop_first=True)


    df["pgemplst"] = pd.Categorical(
        df["pgemplst"],
        categories=[5, 1, 2, 3, 4],  # make 5 = Not employed the base
        ordered=False
    )

    emp_dummies = pd.get_dummies(df["pgemplst"], prefix="empstat", drop_first=True)

    # REMOVE FOR NOW, MADE NO SENSE IN THE OUTCOME
    # hh_dummies = pd.get_dummies(df["hgtyp1hh"].astype("category"), prefix="hh", drop_first=True)

    # ─────────────────────────────── combine & clean ──────────────────────────
    # We now build the full design matrix X.
    # Conceptually, X = [X_continuous | X_binary | X_dummies], where:
    # - X_continuous holds standardised continuous covariates (z_age, z_num_siblings)
    # - X_binary holds variables like 'living_with_parent' (0/1)
    # - X_dummies holds one-hot encoded categories (sex, state, etc.)
    # This mirrors the common structure in econometrics: Xβ + Dγ
    X = pd.concat([continuous_std, binary, sex_dummies, bula_dummies, emp_dummies], axis=1)
    X = X.dropna()
    y = y.loc[X.index]

    # Add constant (intercept)
    # Including a constant allows the model to estimate a baseline probability
    # of non-take-up when all covariates are zero (or at their reference level).
    # It also ensures the model can correctly fit the average outcome unless
    # explicitly restricted. Without it, the model would be forced through the origin,
    # which is almost never appropriate in Probit/logit models.
    X = sm.add_constant(X).astype(float)

    return y, X

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
    res = model.fit(disp=False, cov_type = "HC0") # true for printing convergence messages
    print(res.summary())

    # ----------------------- non‑take‑up prediction --------------------------
    p_hat = res.predict(X)

    # Use prevalence cut‑off to classify (better with imbalanced data)
    # FIX: The cut off doesnt make sense to me, how does cutoff relate to the 
    # probit model?
    threshold = y.mean()
    non_takeup_hat = (p_hat >= threshold).astype(int)
    predicted_rate = non_takeup_hat.mean()

    print(f"\nSample non‑take‑up rate:          {y.mean():.2%}")
    print(f"Predicted non‑take‑up rate (≥{threshold:.2f}): {predicted_rate:.2%}")

    mfx = res.get_margeff()
    print(mfx.summary())

if __name__ == "__main__":
    main()



