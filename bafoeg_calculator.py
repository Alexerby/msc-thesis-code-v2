from __future__ import annotations

"""Minimal façade ‑ constructs the pipeline and exports the result."""

from misc.utility_functions import export_data, Literal
from loaders.registry import LoaderRegistry
from pipeline.build import BafoegPipeline

ExportType = Literal["csv", "excel"]


class BafoegCalculator:
    def __init__(self, loaders: LoaderRegistry | None = None):
        self.loaders = loaders or LoaderRegistry()
        self.pipeline = BafoegPipeline(self.loaders)
        self.main_df = None

    def run(self):
        self.loaders.load_all()
        self.main_df = self.pipeline.build()
        return self.main_df


if __name__ == "__main__":
    calc = BafoegCalculator()
    df = calc.run()
    if df is not None:
        print("Exporting → student_parental_income.xlsx …")
        export_data("excel", df, "student_parental_income", sheet_name="main_df")
        print("✅ Done.")
