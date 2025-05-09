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
from misc.utility_functions import _norm, _auto_map

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
    """
    Join age, Bundesland, household type, and East German flag.

    Adds:
      - age (from gebjahr/gebmonat)
      - bula (state), from region table
      - hgtyp1hh (household type), from hgen table
      - east_background = 1 if state in former East Germany (bula ∈ 11–16)
    """
    out = df.copy()

    # Age
    p = ppath_df[["pid", "hid", "syear", "gebjahr", "gebmonat"]].copy()
    p["age"] = p["syear"] - p["gebjahr"] - (p["gebmonat"] > 6).astype(int)
    out = (
        out.drop(columns=["gebjahr", "gebmonat"], errors="ignore")
           .merge(p[["pid", "syear", "age"]], on=["pid", "syear"], how="left")
    )

    # Bundesland
    out = out.merge(region_df[["hid", "syear", "bula"]], on=["hid", "syear"], how="left")

    # East German flag
    east_states = {11, 12, 13, 14, 15, 16}
    out["east_background"] = out["bula"].isin(east_states).astype(int)

    # Household type
    out = out.merge(hgen_df[["hid", "syear", "hgtyp1hh"]], on=["hid", "syear"], how="left")

    return out

# ---------------------------------------------------------------------------
# Education & income helpers
# ---------------------------------------------------------------------------


def merge_employment_status(df: pd.DataFrame, pgen_df: pd.DataFrame) -> pd.DataFrame:
    """Merge employment status from pgen/pgemplst into main dataframe.

    Categories:
        1 = Full-time
        2 = Part-time
        3 = Vocational training
        4 = Marginal/irregular
        5 = Not employed
        6–8 = Other (less relevant or rare)
    """
    pg = pgen_df.copy()
    valid_codes = [1, 2, 3, 4, 5]  # Limit to most relevant categories
    pg = pg.loc[pg["pgemplst"].isin(valid_codes)]
    return df.merge(pg[["pid", "syear", "pgemplst"]], on=["pid", "syear"], how="left")

def merge_student_grant_dummy(df: pd.DataFrame, pkal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a dummy for whether the person received student grants (BAföG) in the previous year.

    Returns a new column: *received_student_grant* = 1 if ISTUY > 0, else 0.
    """
    tmp = pkal_df[["pid", "syear", "istuy"]].copy()
    tmp["received_student_grant"] = (tmp["istuy"] > 0).astype(int)
    return df.merge(
        tmp[["pid", "syear", "received_student_grant"]],
        on=["pid", "syear"],
        how="left",
    )


def merge_student_status_and_support(df: pd.DataFrame, pl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge student status and support-related fields from the PL file:

    Merged columns:
        - 'plg0012_h': Currently in education (yes/no)
        - 'plh0258_h': Religion affiliation
        - 'plc0168_h': BAföG/Grant gross monthly amount

    Joins on ['pid', 'syear'].
    """
    return df.merge(pl_df, on=["pid", "syear"], how="left")


def merge_income(
    df: pd.DataFrame,
    pgen_df: pd.DataFrame,
    invalid_codes: Sequence[int],
) -> pd.DataFrame:
    """
    Merge gross income from employment into the main student DataFrame.

    This function extracts each student's gross labor income from the pgen file
    and merges both monthly and annual income values into the main DataFrame.

    - Invalid codes (e.g., -1 to -8) are treated as missing.
    - Missing income is assumed to be zero (i.e., the student had no earnings).
    - Income is converted from monthly to annual.

    Parameters
    ----------
    df : pd.DataFrame
        The main student-level DataFrame containing at least 'pid' and 'syear'.
    pgen_df : pd.DataFrame
        The pgen table containing labor income ('pglabgro').
    invalid_codes : Sequence[int]
        A list or set of codes representing missing or invalid income data.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two new columns merged in:
        - 'gross_monthly_income': Monthly earnings from pgen, with NaN → 0
        - 'gross_annual_income' : gross_monthly_income × 12
    """
    pg = pgen_df.copy()

    # Treat invalid codes as missing
    pg["pglabgro"] = pg["pglabgro"].where(~pg["pglabgro"].isin(invalid_codes), np.nan)

    # Assume missing means zero income for BAföG calculation purposes
    pg["gross_monthly_income"] = pg["pglabgro"].fillna(0)
    pg["gross_annual_income"] = pg["gross_monthly_income"] * 12

    return df.merge(
        pg[["pid", "syear", "gross_monthly_income", "gross_annual_income"]],
        on=["pid", "syear"],
        how="left"
    )


def apply_student_income_tax(df: pd.DataFrame, tax_service, base_col="gross_annual_income") -> pd.DataFrame:
    """
    Apply income tax and contributions to student's gross income to compute
    net income relevant for BAföG.

    Ensures that net income is never negative.

    Returns a new column:
        'net_annual_student_income'
    """
    tax_cols = df.apply(tax_service.compute_for_row, axis=1, result_type="expand")
    df[["student_income_tax", "student_church_tax", "student_soli"]] = tax_cols

    net_income = (
        df[base_col]
        - df["student_income_tax"].fillna(0)
        - df["student_church_tax"].fillna(0)
        - df["student_soli"].fillna(0)
    )

    df["net_annual_student_income"] = np.maximum(net_income, 0)
    return df


def filter_students(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["plg0012_h"] == 1].drop(columns=["plg0012_h"])


# ---------------------------------------------------------------------------
# Parental links & income composition
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# § 21 – split parental contribution across co-supported children
# ---------------------------------------------------------------------------

def split_parental_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements § 21 BAföG:
        If several children of the same parents receive training support in the
        same calendar year, the parental income to be credited is divided equally
        among them.

    Requires columns
        • fnr, mnr   – biological father / mother pid (from merge_parent_links)
        • syear      – survey year
        • monthly_parental_income_post_additional_allowance
    """

    out = df.copy()

    # 1) how many of this parent pair's children are in *this* student frame & year
    out["children_in_training"] = (
        out.groupby(["fnr", "mnr", "syear"])["pid"].transform("count")
    ).clip(lower=1)        # guard against division by zero

    # 2) per-child parental share
    out["monthly_parental_contribution_split"] = (
        out["monthly_parental_income_post_additional_allowance"]
        / out["children_in_training"]
    )

    return out


def count_own_children(df: pd.DataFrame, bioparen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column `num_own_children` indicating how many children each student (pid) has,
    based on the bioparen table.

    This uses the inverse logic of the bioparen mapping:
    • bioparen normally maps child_pid → (fnr, mnr)
    • here, we reverse it: parent_pid → [children]

    Parameters
    ----------
    df           : DataFrame containing students (must include 'pid')
    bioparen_df  : SOEP bioparen dataset with columns ['pid', 'fnr', 'mnr']

    Returns
    -------
    DataFrame with one new column:
        • num_own_children: number of known biological children per student
    """
    # Build a long-form mapping of parent_pid → child_pid
    parent_child = (
        pd.concat([
            bioparen_df[["fnr", "pid"]].rename(columns={"fnr": "parent_pid"}),
            bioparen_df[["mnr", "pid"]].rename(columns={"mnr": "parent_pid"}),
        ])
        .dropna(subset=["parent_pid"])
        .astype({"parent_pid": int})
    )

    # Count how many times each parent appears (i.e. number of children)
    counts = (
        parent_child.groupby("parent_pid")
                    .size()
                    .rename("num_own_children")
                    .reset_index()
    )

    # Merge back to student rows
    out = df.merge(counts, how="left", left_on="pid", right_on="parent_pid")
    out["num_own_children"] = out["num_own_children"].fillna(0).astype(int)
    return out.drop(columns=["parent_pid"])


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


def apply_parental_income_tax(df: pd.DataFrame, tax_service) -> pd.DataFrame:
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


def apply_social_insurance_allowance(
    df: pd.DataFrame,
    *,
    input_var: str,
    output_var: str,
    rate: float = 0.223,
) -> pd.DataFrame:
    """
    Generic flat-rate social-insurance deduction (§ 21 Abs 2).

    Parameters
    ----------
    input_var   column that contains the **pre-deduction** annual income
    output_var  name of the **post-deduction** column to create
    rate        default 22.3 %
    """
    out = df.copy()
    out[output_var] = out[input_var] * (1 - rate)
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
        out["monthly_parental_income_post_sibling_allowance"] * (1 - out["rate_50_percent"]),
        0,
    )

    return out.drop(columns=["valid_from", "syear_date", "rate_50_percent"])



def apply_lump_sum_deduction_student(df: pd.DataFrame,
                                     werbung_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # merge pauschale by survey year
    tab = werbung_df.rename(columns={"Year": "syear"})
    out = out.merge(tab, on="syear", how="left")

    # deduct from *gross* annual income
    out["gross_after_werbung"] = np.maximum(
        out["gross_annual_income"] - out["werbungskostenpauschale"], 0
    )

    return out.drop(columns=["werbungskostenpauschale"])

#FIX: We are still not applying § 21 which I also think is relevant
def apply_student_income_deduction(
    df: pd.DataFrame,
    allowance_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply § 23 (1) BAföG: Freibeträge (allowances) from student's own income.

    Monthly income is reduced by:
      - Base allowance for the student (§ 23 Abs. 1 Nr. 1),
      - Allowance for a spouse or partner (§ 23 Abs. 1 Nr. 2),
      - Allowance per child (§ 23 Abs. 1 Nr. 3).

    Returns a new column:
        'monthly_income_post_student_allowance'

    Assumes:
        - 'gross_annual_income' column exists,
        - 'has_partner' and 'num_children' columns exist (or defaulted to 0).
    """
    # Prep statute table
    tab = allowance_table.rename(columns={
        "Valid from": "valid_from",
        "§ 23 (1) 1": "allowance_self",
        "§ 23 (1) 2": "allowance_spouse",
        "§ 23 (1) 3": "allowance_child"
    }).copy()

    tab["valid_from"] = pd.to_datetime(tab["valid_from"])
    for col in ["allowance_self", "allowance_spouse", "allowance_child"]:
        tab[col] = pd.to_numeric(tab[col], errors="coerce")
    tab = tab.sort_values("valid_from").ffill()

    # Merge allowance table by survey year
    out = df.copy()
    out["syear_date"] = pd.to_datetime(out["syear"].astype(str) + "-01-01")
    out = pd.merge_asof(
        out.sort_values("syear_date"),
        tab[["valid_from", "allowance_self", "allowance_spouse", "allowance_child"]],
        left_on="syear_date",
        right_on="valid_from",
        direction="backward"
    )

    # Monthly income and default fallback values
    out["monthly_student_income"] = out["net_annual_student_income"] / 12

    if "has_partner" not in out:
        out["has_partner"] = 0
    else:
        out["has_partner"] = out["has_partner"].fillna(0)

    if "num_children" not in out:
        out["num_children"] = 0
    else:
        out["num_children"] = out["num_children"].fillna(0)

    # Total deduction
    out["student_total_allowance"] = (
        out["allowance_self"]
        + out["has_partner"] * out["allowance_spouse"]
        + out["num_children"] * out["allowance_child"]
    )

    # Apply the deduction
    out["student_excess_income"] = np.maximum(
        out["monthly_student_income"] - out["student_total_allowance"], 0
    )

    return out.drop(columns=[
        "valid_from", "syear_date",
        "allowance_self", "allowance_spouse", "allowance_child"
    ])



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


def apply_sibling_allowance(df: pd.DataFrame, allowance_table: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the § 25 (3) 2 BAföG sibling allowance.

    For each sibling (not in training), a fixed deduction (per survey year) is
    subtracted from the monthly parental income. The allowance is defined yearly 
    in the § 25 policy table.

    Requires:
        • df must include 'syear', 'num_siblings', 
          and 'monthly_parental_income_post_basic_allowance'

    Returns:
        • DataFrame with new column:
            'monthly_parental_income_post_sibling_allowance'
    """
    out = df.copy()

    # Clean policy table: keep year and §25 (3)2
    table = allowance_table.copy()
    table = table.rename(columns={"§ 25 (3) 2": "sibling_allowance_yearly"})
    table["syear"] = pd.to_datetime(table["Valid from"]).dt.year
    table = table[["syear", "sibling_allowance_yearly"]]

    # Merge in the policy value for each row's year
    out = out.merge(table, on="syear", how="left")

    # Fill missing allowance values with 0
    out["sibling_allowance_yearly"] = out["sibling_allowance_yearly"].fillna(0)

    # Apply the deduction (convert yearly → monthly)
    out["monthly_parental_income_post_sibling_allowance"] = np.maximum(
        out["monthly_parental_income_post_basic_allowance"]
        - (out["sibling_allowance_yearly"] / 12) * out["num_siblings"],
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



def flag_partner_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'has_partner' = 1 if the person has a spouse/partner, 0 otherwise.
    Based on the SOEP 'partner' variable from ppathl.
    """
    out = df.copy()
    out["has_partner"] = (out["partner"] > 0).astype(int)
    return out



def compute_bafög_monthly_award(
    df: pd.DataFrame,
    need_table: pd.DataFrame,
    insurance_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute monthly BAföG award.

    total_need = base_need + housing_allowance + insurance_supplement
    award      = max(total_need
                     − parental_income_after_allowances
                     − own_income_after_allowances, 0)
    """

    # ───────────── § 13: base need & housing ─────────────
    need_patterns = {
        "valid_from":        [r"validfrom"],
        "base_need_parent":  [r"§?13\(1\)1"],
        "base_need_away":    [r"§?13\(1\)2"],
        "housing_with_parents": [r"§?13\(2\)1"],
        "housing_away":         [r"§?13\(2\)2"],
    }
    need = (
        need_table
        .rename(columns=_auto_map(need_table.columns, need_patterns))
        .assign(valid_from=lambda d: pd.to_datetime(d["valid_from"]))
        .sort_values("valid_from")
    )

    # ────────── § 13a: health & LTC supplements ──────────
    ins_patterns = {
        "valid_from_ins": [r"validfrom"],
        "kv_stat_mand":   [r"§?13a\(1\)1"],
        "pv_stat_mand":   [r"§?13a\(1\)2"],
    }
    # strip all column-name whitespace first for reliability
    insurance_table.columns = insurance_table.columns.str.strip()
    ins = (
        insurance_table
        .rename(columns=_auto_map(insurance_table.columns, ins_patterns))
        .assign(valid_from_ins=lambda d: pd.to_datetime(d["valid_from_ins"]))
        .sort_values("valid_from_ins")
    )

    # ───────────────────── merge by survey year ─────────────────────
    out = df.copy()
    out["syear_date"] = pd.to_datetime(out["syear"].astype(str) + "-01-01")

    out = pd.merge_asof(
        out.sort_values("syear_date"),
        need[
            ["valid_from",
             "base_need_parent", "base_need_away",
             "housing_with_parents", "housing_away"]
        ],
        left_on="syear_date",
        right_on="valid_from",
        direction="backward",
    )

    out = pd.merge_asof(
        out.sort_values("syear_date"),
        ins[["valid_from_ins", "kv_stat_mand", "pv_stat_mand"]],
        left_on="syear_date",
        right_on="valid_from_ins",
        direction="backward",
        allow_exact_matches=True,
    )

    # ───── choose base need & housing allowance ─────
    out["base_need"] = np.where(
        out["lives_with_parent"], out["base_need_parent"], out["base_need_away"]
    )
    out["housing_allowance"] = np.where(
        out["lives_with_parent"], out["housing_with_parents"], out["housing_away"]
    )

    # ───── health & LTC supplement (§ 13a) ─────
    out["insurance_supplement"] = (
        out["kv_stat_mand"].fillna(0) + out["pv_stat_mand"].fillna(0)
    )

    # ─────────────── final need & award ───────────────
    out["total_need"] = (
        out["base_need"]
        + out["housing_allowance"]
        + out["insurance_supplement"]
    )

    out["monthly_award"] = np.maximum(
        out["total_need"]
        - out["monthly_parental_contribution_split"]
        - out["student_excess_income"],
        0,
    )

    # ─────────── clean-up (ignore missing cols) ───────────
    return out.drop(
        columns=[
            "valid_from", "valid_from_ins", "syear_date",
            "base_need_parent", "base_need_away",
            "housing_with_parents", "housing_away",
            # "kv_stat_mand", "pv_stat_mand", "insurance_supplement",
        ],
        errors="ignore",
    )


def flag_theoretical_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a dummy column `eligible_for_bafoeg`:
        - 1 if the computed monthly_award > 0
        - 0 otherwise

    This reflects theoretical BAföG eligibility under current law,
    irrespective of whether the student actually received any support.
    """
    out = df.copy()
    out["eligible_for_bafoeg"] = (out["monthly_award"] > 0).astype(int)
    return out

def add_receives_bafoeg_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an integer column 'receives_bafoeg' = 1 if plc0168_h > 0, else 0.
    """
    out = df.copy()
    out["receives_bafoeg"] = (out["plc0168_h"] > 0).astype(int)
    return out



# ---------------------------------------------------------------------------
# PIPE WRAPPERS FOR build.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Demographics mini-pipeline
# ---------------------------------------------------------------------------

def demographics_pipeline(df, ppath_df, region_df, hgen_df, pl_df):
    """
    Combines background enrichments to build the student profile:

    1. Demographics (sex, age, household, region, east/west)
    2. Student status & support (education, religion, BAföG grant)

    Returns
    -------
    Enriched student-level DataFrame.
    """
    return (
        df
        .pipe(add_demographics, ppath_df, region_df, hgen_df)
    )


# ---------------------------------------------------------------------------
#  Student-status mini-pipeline
# ---------------------------------------------------------------------------

def student_status_pipeline(
    df,
    pequiv_df,
    ppath_df,
    pgen_df,
    bioparen_df,
    pl_df
):
    """
    Adds flags and status indicators for BAföG eligibility:

        1.  merge_student_grant_dummy  → received_student_grant
        2.  merge_student_status_and_support → BAföG, religion, school status
        3.  flag_living_with_parents   → affects housing allowance
        4.  add_receives_bafoeg_flag   → consistency check
        5.  merge_employment_status    → pgemplst
        6.  flag_partner_status        → has_partner
        7.  count_own_children         → num_children
        8.  filter_students             → (filter)

    Required for both the benefit formula and the take-up model.
    """
    return (
        df
        # .pipe(merge_student_grant_dummy, pequiv_df)
        .pipe(merge_student_status_and_support, pl_df)
        .pipe(flag_living_with_parents, ppath_df)
        .pipe(add_receives_bafoeg_flag)
        .pipe(merge_employment_status, pgen_df)
        .pipe(flag_partner_status)
        .pipe(count_own_children, bioparen_df)
        .pipe(filter_students)
    )


# ---------------------------------------------------------------------------
#  Student mini-pipeline
# ---------------------------------------------------------------------------

def student_income_pipeline(
    df, pgen_df, invalid_codes, tax_service, allowance_table, werbung_df
):
    """
    One-shot wrapper that replaces 5 individual pipes related to student's own income:

        1.  merge_income
        2.  apply_lump_sum_deduction_student      (§ 9a EStG)
        3.  apply_social_insurance_allowance      (§ 21 Abs. 2 Nr. 1 BAföG)
        4.  apply_student_income_tax              (income tax computation)
        5.  apply_student_income_deduction        (§ 23 Abs. 1 BAföG)

    Computes net BAföG-relevant student income after all statutory deductions.
    """
    return (
        df
        # 1  gross labour income
        .pipe(merge_income, pgen_df, invalid_codes)

        # 2  Werbungskosten-Pauschale (§ 9a EStG)
        .pipe(apply_lump_sum_deduction_student, werbung_df)

        # 3  flat 22.3 % social-insurance allowance (§ 21 II Nr. 1)
        .pipe(apply_social_insurance_allowance,
              input_var="gross_after_werbung",
              output_var="income_post_si")

        # 4  income-tax calculation on the SI-reduced base
        .pipe(apply_student_income_tax, tax_service, base_col="income_post_si")

        # 5  § 23 personal / spouse / child allowances
        .pipe(apply_student_income_deduction, allowance_table)
    )


# ---------------------------------------------------------------------------
#  Parental mini-pipeline
# ---------------------------------------------------------------------------

def parental_income_pipeline(
    df: pd.DataFrame,
    pgen_df: pd.DataFrame,
    invalid_codes: list[int],
    werbung_df: pd.DataFrame,
    bioparen_df: pd.DataFrame,
    tax_service,
    ppath_df: pd.DataFrame,
    allowance_table: pd.DataFrame,
    require_both_parents: bool = False,
) -> pd.DataFrame:
    """
    One-shot wrapper that replaces 10 individual pipes:

        1.  merge_parental_incomes
        2.  apply_lump_sum_deduction_parents
        3.  apply_social_insurance_allowance
        4.  find_siblings
        5.  merge_sibling_income
        6.  apply_parental_income_tax
        7.  flag_parent_relationship
        8.  apply_basic_allowance_parents
        9.  apply_sibling_allowance
       10.  apply_additional_allowance_parents
       11.  split_parental_contribution   ( § 25 Abs. 3 )

    The output is identical to the sequence above, but we do only one merge
    on `pgen` and avoid ~10 intermediate DataFrame copies.
    """
    return (
        df
        .pipe(
            merge_parental_incomes,
            pgen_df,
            invalid_codes,
            require_both_parents=require_both_parents,
        )
        .pipe(apply_lump_sum_deduction_parents, werbung_df)
        .pipe(apply_social_insurance_allowance,
              input_var="adjusted_parental_income",
              output_var="parental_income_post_insurance_allowance")
        .pipe(find_siblings, bioparen_df)
        .pipe(merge_sibling_income, pgen_df, invalid_codes)
        .pipe(apply_parental_income_tax, tax_service)
        .pipe(flag_parent_relationship, ppath_df)
        .pipe(apply_basic_allowance_parents, allowance_table)
        .pipe(apply_sibling_allowance, allowance_table)
        .pipe(apply_additional_allowance_parents, allowance_table)
        .pipe(split_parental_contribution)          # § 25 Abs. 3
    )
