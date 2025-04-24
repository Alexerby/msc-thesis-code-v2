from __future__ import annotations

"""High‑level pipeline that wires loaders, pure steps and services."""

import pandas as pd
from data_handler import SOEPStatutoryInputs
from loaders.registry import LoaderRegistry
from services.tax import TaxService
from pipeline import steps as S

INVALID_CODES = {-1, -2, -3, -4, -5, -6, -7, -8}

class BafoegPipeline:
    """Compose the final student DataFrame in one go."""

    def __init__(self, loaders: LoaderRegistry):
        self.loaders = loaders
        self.tax = TaxService()
        self._load_policy_tables()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def build(self) -> dict[str, pd.DataFrame]:
        # 1) run the pipeline exactly as before
        df_full = (
            self.loaders.ppath().copy()
            .pipe(S.filter_post_euro)
            .pipe(S.add_demographics,
                  self.loaders.ppath(),
                  self.loaders.region(),
                  self.loaders.hgen())
            .pipe(S.merge_education, self.loaders.pl())
            .pipe(S.merge_income, self.loaders.pgen(), INVALID_CODES)
            .pipe(S.filter_students)
            .pipe(S.merge_parent_links, self.loaders.bioparen())
            .pipe(S.merge_parental_incomes,
                  self.loaders.pgen(),
                  INVALID_CODES,
                  require_both_parents=False)
            .pipe(S.apply_lump_sum_deduction, self._werbung_df)
            .pipe(S.apply_social_insurance_allowance)
        )

        # add row-wise tax columns, flag relationship, basic allowance…
        # (exactly the same two blocks you already have in build())
        tax_cols = df_full.apply(self.tax.compute_for_row, axis=1, result_type="expand")
        df_full[["parental_income_tax", "parental_church_tax", "parental_soli"]] = tax_cols
        df_full["parental_income_post_income_tax"] = (
            df_full["parental_income_post_insurance_allowance"]
            - df_full["parental_income_tax"].fillna(0)
            - df_full["parental_church_tax"].fillna(0)
        )

        df_full = (
            df_full.pipe(S.flag_parent_relationship, self.loaders.ppath())
                   .pipe(S.apply_basic_allowance_parents, self._allowance_table)
        )

        # 2) Split into logical views
        parents = df_full[[
            "pid", "syear",
            "parental_annual_income",
            "adjusted_parental_income",
            "parental_income_post_insurance_allowance",
            "parental_income_tax", "parental_church_tax", "parental_soli",
            "parental_income_post_income_tax",
            "parental_income_post_basic_allowance"
        ]].copy()

        students = df_full[[
            "pid", "syear", "age", "bula",
            "gross_annual_income", "hgtyp1hh"
        ]].copy()

        return {
            "parents":  parents,
            "students": students,
            "full":     df_full      # keep complete table for debugging
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_policy_tables(self):
        werb = SOEPStatutoryInputs("Werbungskostenpauschale")
        werb.load_dataset(columns=["Year", "werbungskostenpauschale"])
        self._werbung_df = werb.data.copy()

        allow = SOEPStatutoryInputs("Basic Allowances - § 25")
        allow.load_dataset(columns=lambda _: True)
        self._allowance_table = allow.data.copy()
