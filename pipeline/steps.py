from __future__ import annotations

"""Composable, *side-effect-free* dataframe transformations used by
`BafoegPipeline`.  Each function:

* accepts a **DataFrame** (and possibly supplementary frames/params)
* returns a **new** DataFrame (no in-place mutation)
* contains *no* I/O, logging, or global state

That makes them safe for `df.pipe(...)`, trivial to unit-test, and easy
to reorder in a pipeline.
"""

from typing import Sequence
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Basic filters / enrichers
# ---------------------------------------------------------------------------

def filter_post_euro(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only survey years 2002 + (after EUR introduction)."""
    return df.loc[df["syear"] >= 2002].copy()


def add_demographics(
    df: pd.DataFrame,
    ppath_df: pd.DataFrame,
    region_df: pd.DataFrame,
    hgen_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join age, Bundesland, and household type."""
    out = df.copy()

    # Age ---------------------------------------------------------
    p = ppath_df[["pid", "hid", "syear", "gebjahr", "gebmonat"]].copy()
    p["age"] = p["syear"] - p["gebjahr"] - (p["gebmonat"] > 6).astype(int)
    out = (
        out.drop(columns=["gebjahr", "gebmonat"], errors="ignore")
           .merge(p[["pid", "syear", "age"]], on=["pid", "syear"], how="left")
    )

    # Bundesland --------------------------------------------------
    out = out.merge(region_df[["hid", "syear", "bula"]], on=["hid", "syear"], how="left")

    # Household type ---------------------------------------------
    out = out.merge(hgen_df[["hid", "syear", "hgtyp1hh"]], on=["hid", "syear"], how="left")

    return out

# ---------------------------------------------------------------------------
# Education & income helpers
# ---------------------------------------------------------------------------

def merge_education(df: pd.DataFrame, pl_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(pl_df, on=["pid", "syear"], how="left")


def merge_income(
    df: pd.DataFrame,
    pgen_df: pd.DataFrame,
    invalid_codes: Sequence[int],
) -> pd.DataFrame:
    pg = pgen_df.copy()
    pg["pglabgro"] = pg["pglabgro"].where(~pg["pglabgro"].isin(invalid_codes), np.nan)
    pg.rename(columns={"pglabgro": "gross_monthly_income"}, inplace=True)
    pg["gross_annual_income"] = pg["gross_monthly_income"] * 12
    return df.merge(pg[["pid", "syear", "gross_annual_income"]], on=["pid", "syear"], how="left")


def filter_students(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["plg0012_h"] == 1].drop(columns=["plg0012_h"])

# ---------------------------------------------------------------------------
# Parental links & income composition
# ---------------------------------------------------------------------------

def merge_parent_links(df: pd.DataFrame, bioparen_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(bioparen_df, on="pid", how="left")


def merge_parental_incomes(
    df: pd.DataFrame,
    pgen_df: pd.DataFrame,
    invalid_codes: Sequence[int],
    *,
    require_both_parents: bool = True,
) -> pd.DataFrame:
    pg = pgen_df.copy()
    pg["pglabgro"] = pg["pglabgro"].where(~pg["pglabgro"].isin(invalid_codes), np.nan)
    pg.rename(columns={"pglabgro": "parent_income"}, inplace=True)

    father = pg.rename(columns={"pid": "fnr", "parent_income": "father_income"})
    mother = pg.rename(columns={"pid": "mnr", "parent_income": "mother_income"})

    out = (
        df.merge(father[["fnr", "syear", "father_income"]], on=["fnr", "syear"], how="left")
          .merge(mother[["mnr", "syear", "mother_income"]], on=["mnr", "syear"], how="left")
    )

    if require_both_parents:
        out = out.loc[out["father_income"].notna() & out["mother_income"].notna()]
    else:
        out = out.loc[out[["father_income", "mother_income"]].notna().any(axis=1)]

    out["parental_income"] = out[["father_income", "mother_income"]].sum(axis=1, min_count=1)
    out["parental_annual_income"] = out["parental_income"] * 12
    return out

# ---------------------------------------------------------------------------
# Statutory deductions & allowances
# ---------------------------------------------------------------------------

def apply_lump_sum_deduction(df: pd.DataFrame, werbung_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(werbung_df.rename(columns={"Year": "syear"}), on="syear", how="left")
    out["adjusted_parental_income"] = out["parental_annual_income"] - out["werbungskostenpauschale"]
    return out.drop(columns="werbungskostenpauschale")


def apply_social_insurance_allowance(df: pd.DataFrame, rate: float = 0.223) -> pd.DataFrame:
    out = df.copy()
    out["parental_income_post_insurance_allowance"] = out["adjusted_parental_income"] * (1 - rate)
    return out


def flag_parent_relationship(df: pd.DataFrame, ppath_df: pd.DataFrame) -> pd.DataFrame:
    p = ppath_df[["pid", "syear", "parid"]].copy()
    father = p.rename(columns={"pid": "fnr", "parid": "parid_of_father"})
    mother = p.rename(columns={"pid": "mnr", "parid": "parid_of_mother"})
    out = (
        df.merge(father, on=["fnr", "syear"], how="left")
          .merge(mother, on=["mnr", "syear"], how="left")
    )
    out["parents_are_partnered"] = (
        out["parid_of_father"] == out["mnr"]
    ) & (
        out["parid_of_mother"] == out["fnr"]
    )
    return out.drop(columns=["parid_of_father", "parid_of_mother"])


def apply_basic_allowance_parents(df: pd.DataFrame, allowance_table: pd.DataFrame) -> pd.DataFrame:
    tab = allowance_table.rename(columns={
        "Valid from": "valid_from",
        "§ 25 (1) 1": "allowance_joint",
        "§ 25 (1) 2": "allowance_single",
    })
    tab["valid_from"] = pd.to_datetime(tab["valid_from"])
    tab = tab.sort_values("valid_from")

    out = df.copy()
    out["syear_date"] = pd.to_datetime(out["syear"].astype(str) + "-01-01")
    out = pd.merge_asof(
        out.sort_values("syear_date"),
        tab[["valid_from", "allowance_joint", "allowance_single"]],
        left_on="syear_date",
        right_on="valid_from",
        direction="backward",
    )
    out["parental_income_post_basic_allowance"] = np.maximum(
        np.where(
            out["parents_are_partnered"],
            out["parental_income"] - out["allowance_joint"],
            out["parental_income"] - 2 * out["allowance_single"],
        ),
        0,
    )
    return out.drop(columns=["valid_from", "syear_date", "allowance_joint", "allowance_single"])


# ---------------------------------------------------------------------------
# Siblings
# ---------------------------------------------------------------------------

def find_siblings(
    df: pd.DataFrame,
    bioparen_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach a list of sibling pids to each student row.
    Full siblings = same biological mother *and* father.
    """
    # bioparen: pid, fnr, mnr
    sib_map = (
        bioparen_df
        .dropna(subset=["fnr", "mnr"])
        .groupby(["fnr", "mnr"])["pid"]
        .apply(list)          # -> list of kids per parent pair
        .rename("sib_list")
        .reset_index()
    )

    out = df.merge(sib_map, on=["fnr", "mnr"], how="left")
    # drop own pid from the list
    out["sib_list"] = out.apply(
        lambda r: [s for s in (r.sib_list or []) if s != r.pid],
        axis=1,
    )
    return out


def merge_sibling_income(
    df: pd.DataFrame,
    pgen_df: pd.DataFrame,
    invalid_codes: Sequence[int]
) -> pd.DataFrame:
    """
    Add two columns:
      * num_siblings – number of siblings per student
      * siblings_income_sum – total income of those siblings
    """
    if "sib_list" not in df:
        raise KeyError("run find_siblings() first")

    # Pre-clean pgen
    income = pgen_df.copy()
    income["pglabgro"] = income["pglabgro"].where(
        ~income["pglabgro"].isin(invalid_codes), np.nan
    )
    income["gross_annual_income"] = income["pglabgro"] * 12
    income = income[["pid", "syear", "gross_annual_income"]]

    # Preserve student pid
    exploded = (
        df[["pid", "syear", "sib_list"]].copy()
        .assign(student_pid=lambda d: d["pid"])
        .explode("sib_list")
        .rename(columns={"sib_list": "sibling_pid"})
        .dropna(subset=["sibling_pid"])
        .astype({"sibling_pid": int})
    )

    exploded = exploded.merge(
        income, left_on=["sibling_pid", "syear"], right_on=["pid", "syear"], how="left"
    )

    sib_agg = (
        exploded.groupby(["student_pid", "syear"])["gross_annual_income"]
                .agg(num_siblings="count", siblings_income_sum="sum")
                .reset_index()
                .rename(columns={"student_pid": "pid"})
    )

    out = df.merge(sib_agg, on=["pid", "syear"], how="left")
    out[["num_siblings", "siblings_income_sum"]] = out[[
        "num_siblings", "siblings_income_sum"
    ]].fillna(0)

    return out


def apply_sibling_allowance(df: pd.DataFrame, rate: float = 2_000) -> pd.DataFrame:
    """
    Example: deduct €2 000 per sibling from parental income (placeholder logic!)
    """
    out = df.copy()
    out["parental_income_post_sibling_allowance"] = np.maximum(
        out["parental_income_post_basic_allowance"] - rate * out["num_siblings"],
        0,
    )
    return out
