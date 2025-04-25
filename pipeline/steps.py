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


def apply_income_tax(df: pd.DataFrame, tax_service) -> pd.DataFrame:
    tax_cols = df.apply(tax_service.compute_for_row, axis=1, result_type="expand")
    df[["parental_income_tax", "parental_church_tax", "parental_soli"]] = tax_cols
    df["parental_income_relevant_for_bafög"] = (
        df["parental_income_post_insurance_allowance"]
        - df["parental_income_tax"].fillna(0)
        - df["parental_church_tax"].fillna(0)
    )
    df["monthly_parental_income_relevant_for_bafoeg"] = df["parental_income_relevant_for_bafög"] / 12
    return df

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

# ---------------------------------------------------------------------------
# Statutory deductions & allowances
# ---------------------------------------------------------------------------

def apply_lump_sum_deduction_parents(df: pd.DataFrame, werbung_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(werbung_df.rename(columns={"Year": "syear"}), on="syear", how="left")
    out["adjusted_parental_income"] = out["parental_annual_income"] - out["werbungskostenpauschale"]
    return out.drop(columns="werbungskostenpauschale")


def apply_social_insurance_allowance(df: pd.DataFrame, rate: float = 0.223) -> pd.DataFrame:
    out = df.copy()
    out["parental_income_post_insurance_allowance"] = out["adjusted_parental_income"] * (1 - rate)
    return out


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

    out["monthly_parental_income_post_basic_allowance"] = np.maximum(
        np.where(
            out["parents_are_partnered"],
            out["monthly_parental_income_relevant_for_bafoeg"] - out["allowance_joint"],
            out["monthly_parental_income_relevant_for_bafoeg"] - 2 * out["allowance_single"],
        ),
        0,
    )

    return out.drop(columns=["valid_from", "syear_date", "allowance_joint", "allowance_single"])

def apply_additional_allowance_parents(df: pd.DataFrame, allowance_table: pd.DataFrame) -> pd.DataFrame:
    """
    Applies § 25 (4) 1 BAföG: 50% of income remaining after basic allowance.
    This uses the percentage stated in the table column "§ 25 (4) 1".
    """
    tab = allowance_table.rename(columns={
        "Valid from": "valid_from",
        "§ 25 (4) 1": "rate_50_percent"
    })
    tab["valid_from"] = pd.to_datetime(tab["valid_from"])
    tab["rate_50_percent"] = tab["rate_50_percent"].str.rstrip("% ").astype(float) / 100
    tab = tab.sort_values("valid_from")

    out = df.copy()
    out["syear_date"] = pd.to_datetime(out["syear"].astype(str) + "-01-01")
    out = pd.merge_asof(
        out.sort_values("syear_date"),
        tab[["valid_from", "rate_50_percent"]],
        left_on="syear_date",
        right_on="valid_from",
        direction="backward",
    )

    out["monthly_parental_income_post_additional_allowance"] = np.maximum(
        out["monthly_parental_income_post_basic_allowance"] * (1 - out["rate_50_percent"]),
        0,
    )

    return out.drop(columns=["valid_from", "syear_date", "rate_50_percent"])

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


def apply_sibling_allowance(df: pd.DataFrame, rate: float = 2000) -> pd.DataFrame:
    out = df.copy()
    out["monthly_parental_income_post_sibling_allowance"] = np.maximum(
        out["monthly_parental_income_post_basic_allowance"] - rate * out["num_siblings"],
        0
    )
    return out


# def apply_lump_sum_deduction_siblings(df: pd.DataFrame, werbung_df: pd.DataFrame) -> pd.DataFrame:
#     out = df.merge(werbung_df.rename(columns={"Year": "syear"}), on="syear", how="left")
#     out["adjusted_parental_income"] = out["parental_annual_income"] - out["werbungskostenpauschale"]
#     return out.drop(columns="werbungskostenpauschale")




# ---------------------------------------------------------------------------
# Living-with-parents flag  (§ 13 Abs. 2)
# ---------------------------------------------------------------------------

def flag_living_with_parents(
    df: pd.DataFrame,
    ppathl_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a boolean column  lives_with_parent  (True = shares 'hid' with at least
    one parent in the same year).

    Parameters
    ----------
    df          : student-level dataframe that already contains columns
                  'pid', 'fnr', 'mnr', 'hid', 'syear'.
    ppathl_df   : full SOEP ppathl table (needs cols pid, hid, syear).

    Returns
    -------
    DataFrame  – original columns +  lives_with_parent ,  hid_of_father ,
                 hid_of_mother  (the latter two can be dropped later).
    """
    out = df.copy()

    # -- build year-specific mapping pid → hid --------------------------------
    hh_map = ppathl_df[["pid", "syear", "hid"]]

    father_hid = hh_map.rename(columns={"pid": "fnr", "hid": "hid_of_father"})
    mother_hid = hh_map.rename(columns={"pid": "mnr", "hid": "hid_of_mother"})

    out = (
        out.merge(father_hid, on=["fnr",  "syear"], how="left")
           .merge(mother_hid, on=["mnr", "syear"], how="left")
    )

    # -- flag: lives with ≥1 parent ------------------------------------------
    out["lives_with_parent"] = (
        (out["hid"] == out["hid_of_father"]) |
        (out["hid"] == out["hid_of_mother"])
    )

    return out



def compute_bafög_monthly_award(df: pd.DataFrame, need_table: pd.DataFrame) -> pd.DataFrame:
    """
    § 13 BAföG:
        total_need  = base_need  +  housing_allowance
        award       = max(total_need − monthly_parental_income_post_additional_allowance, 0)

    housing_allowance:
        59 €  if lives_with_parent
        380 € otherwise
    """
    # ---- prep statute table -------------------------------------------------
    tab = need_table.rename(columns={
        "Valid from": "valid_from",
        "§ 13 (1) 2": "base_need",
        "§ 13 (2) 1": "housing_with_parents",
        "§ 13 (2) 2": "housing_away",
    })
    tab["valid_from"] = pd.to_datetime(tab["valid_from"])
    tab = tab.sort_values("valid_from")

    out = df.copy()
    out["syear_date"] = pd.to_datetime(out["syear"].astype(str) + "-01-01")
    out = pd.merge_asof(
        out.sort_values("syear_date"),
        tab[["valid_from", "base_need", "housing_with_parents", "housing_away"]],
        left_on="syear_date",
        right_on="valid_from",
        direction="backward",
    )

    # ---- housing -----------------------------------------------------------
    out["housing_allowance"] = np.where(
        out["lives_with_parent"],       # ← new flag
        out["housing_with_parents"],
        out["housing_away"],
    )

    # ---- final need / award -------------------------------------------------
    out["total_need"] = out["base_need"] + out["housing_allowance"]
    out["monthly_award"] = np.maximum(
        out["total_need"] - out["monthly_parental_income_post_additional_allowance"],
        0,
    )

    return out.drop(columns=[
        "valid_from", "syear_date",
        "base_need", "housing_with_parents", "housing_away",
    ])
