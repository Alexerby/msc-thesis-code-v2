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
        df_full = (
            self.loaders.ppath().copy()
              .pipe(S.filter_post_euro)
              .pipe(
                  S.add_demographics,
                  self.loaders.ppath(),
                  self.loaders.region(),
                  self.loaders.hgen()
              )
              .pipe(S.merge_education, self.loaders.pl())
              .pipe(S.merge_income, self.loaders.pgen(), INVALID_CODES)
              .pipe(S.filter_students)
              .pipe(S.merge_parent_links, self.loaders.bioparen())
              .pipe(
                  S.merge_parental_incomes,
                  self.loaders.pgen(),
                  INVALID_CODES,
                  require_both_parents=False
              )
              .pipe(S.apply_lump_sum_deduction_parents, self._werbung_df)
              .pipe(S.apply_social_insurance_allowance)
              .pipe(S.find_siblings, self.loaders.bioparen())
              .pipe(S.merge_sibling_income, self.loaders.pgen(), INVALID_CODES)
              .pipe(S.apply_income_tax, self.tax)
              .pipe(S.flag_parent_relationship, self.loaders.ppath())
              .pipe(S.apply_basic_allowance_parents, self._allowance_table)
              .pipe(S.apply_sibling_allowance)
              .pipe(S.apply_additional_allowance_parents, self._allowance_table)
              .pipe(S.flag_living_with_parents, self.loaders.ppath())
              .pipe(S.compute_bafög_monthly_award, self._needs_table)
              .pipe(S.add_receives_bafoeg_flag)
        )

        # 2) Split into logical views
        parents = df_full[[
            "pid", "syear",
            "parental_annual_income",
            "adjusted_parental_income",
            "parental_income_post_insurance_allowance",
            "parental_income_tax", "parental_church_tax", "parental_soli",
            "parental_income_relevant_for_bafög",
            "monthly_parental_income_relevant_for_bafoeg",
            "monthly_parental_income_post_basic_allowance", 
            "monthly_parental_income_post_additional_allowance"
        ]].copy()

        students = df_full[[
            "pid", "syear", "age", "bula",
            "gross_annual_income", "hgtyp1hh"
        ]].copy()

        siblings_view = df_full[[
            "pid", "syear", "num_siblings", "siblings_income_sum"
        ]].copy()

        return {
            "parents": parents,
            "students": students,
            "siblings": siblings_view,
            "full": df_full,
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

        needs = SOEPStatutoryInputs("Basic Allowances - § 13")
        needs.load_dataset(columns=lambda _: True)
        self._needs_table = needs.data.copy()
