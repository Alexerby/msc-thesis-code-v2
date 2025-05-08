from __future__ import annotations

from numpy import load
from pandas.core.indexes.interval import le

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
              .pipe(S.filter_post_euro)               # 1
              .pipe(
                  S.demographics_pipeline,            # 2
                  self.loaders.ppath(),
                  self.loaders.region(),
                  self.loaders.hgen(),
                  self.loaders.pl(),
              )
              .pipe(S.filter_students)                # 3
              .pipe(S.merge_parent_links, self.loaders.bioparen())
              .pipe(                                  # 4
                  S.student_status_pipeline,
                  self.loaders.pequiv(),
                  self.loaders.ppath(),
                  self.loaders.pgen(),
                  self.loaders.bioparen(),
              )
              .pipe(                                  # 5
                  S.parental_income_pipeline,
                  self.loaders.pgen(),
                  INVALID_CODES,
                  self._werbung_df,
                  self.loaders.bioparen(),
                  self.tax,
                  self.loaders.ppath(),
                  self._allowance_table,
                  require_both_parents=False,
              )
            .pipe(
                S.student_income_pipeline,
                self.loaders.pgen(),
                INVALID_CODES,
                self.tax,
                self._student_allowance_table,
                self._werbung_df
            )
              .pipe(S.compute_bafög_monthly_award,
                    self._needs_table,
                    self._insurance_table)
              .pipe(S.flag_theoretical_eligibility)
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
            "pid", 
            "syear", 
            "age", 
            "bula",
            "gross_annual_income", 
            "hgtyp1hh"
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

        # Student allowances
        student_allow = SOEPStatutoryInputs("Basic Allowances - § 23")
        student_allow.load_dataset(columns=lambda _: True)
        self._student_allowance_table = student_allow.data.copy()

        # Health insurances etc
        ins = SOEPStatutoryInputs("Basic Allowances - §13a")
        ins.load_dataset(columns=lambda _: True)
        self._insurance_table = ins.data.copy()
