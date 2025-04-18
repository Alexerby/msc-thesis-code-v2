
# Package imports
import numpy as np

# Project imports
from misc.utility_functions import Literal, export_data
from data_handler import SOEPDataHandler


ExportType = Literal["csv", "excel"]

class BafoegCalculator:
    def __init__(self):
        self.invalid_codes = set([-1, -2, -3, -4, -5, -6, -7, -8])
        self.datasets = {}
        self.df = None

    def load_all_data(self):
        self._load_dataset("ppathl", ["pid", "syear", "gebjahr", "sex", "gebmonat", "parid", "partner"])
        self._load_dataset("pl", ["pid", "syear", "plg0012_h"])
        self._load_dataset("pgen", ["pid", "syear", "pglabgro"])
        self._load_dataset("bioparen", ["pid", "fnr", "mnr"])

    def _load_dataset(self, key, columns):
        self.datasets[key] = SOEPDataHandler(key)
        self.datasets[key].load_dataset(columns)

    def _add_demographics(self, df):
        ppathl = self.datasets["ppathl"].data[["pid", "syear", "gebjahr", "gebmonat", "sex"]].copy()
        ppathl["age"] = ppathl["syear"] - ppathl["gebjahr"]
        ppathl["age"] = ppathl["age"] - (ppathl["gebmonat"] > 6).astype(int)

        df = df.merge(ppathl[["pid", "syear", "age", "sex"]], on=["pid", "syear"], how="left")

        return df


    def process_data(self):
        df = self.datasets["ppathl"].data.copy()
        df = self._add_demographics(df)
        df = self._merge_education(df)
        df = self._merge_income(df)
        df = self._filter_students(df)
        df = self._merge_parental_links(df)
        df = self._merge_parental_incomes(df)

        self.df = df

    def _merge_education(self, df):
        edu = self.datasets["pl"].data
        return df.merge(edu, on=["pid", "syear"], how="left")

    def _merge_income(self, df):
        income = self.datasets["pgen"].data.copy()
        income["pglabgro"] = income["pglabgro"].where(~income["pglabgro"].isin(self.invalid_codes), np.nan)
        income.rename(columns={"pglabgro": "gross_income_monthly"}, inplace=True)
        return df.merge(income[["pid", "syear", "gross_income_monthly"]], on=["pid", "syear"], how="left")

    def _filter_students(self, df):
        df = df[df["plg0012_h"] == 1]
        df.rename(columns={"plg0012_h": "currently_in_education"}, inplace=True)
        return df

    def _merge_parental_links(self, df):
        parent_links = self.datasets["bioparen"].data
        return df.merge(parent_links, on="pid", how="left")

    def _merge_parental_incomes(self, df):
        pgen = self.datasets["pgen"].data.copy()
        pgen["pglabgro"] = pgen["pglabgro"].where(~pgen["pglabgro"].isin(self.invalid_codes), np.nan)
        pgen.rename(columns={"pglabgro": "parent_income"}, inplace=True)

        father_income = pgen.rename(columns={"pid": "fnr", "parent_income": "father_income"})
        mother_income = pgen.rename(columns={"pid": "mnr", "parent_income": "mother_income"})

        df = df.merge(father_income[["fnr", "syear", "father_income"]], on=["fnr", "syear"], how="left")
        df = df.merge(mother_income[["mnr", "syear", "mother_income"]], on=["mnr", "syear"], how="left")
        df["parental_income"] = df[["father_income", "mother_income"]].sum(axis=1, skipna=True)
        return df

    def export(self, filename: str, format: ExportType = "csv"):
        if self.df is None:
            raise ValueError("Data has not been processed yet.")
        export_data(format, df=self.df, output_filename=filename)



if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Estimate theoretical BAföG from SOEP data")
    # parser.add_argument("--output", type=str, default="bafoeg_data.csv", help="Output filename")
    # parser.add_argument("--format", type=str, choices=["csv", "excel", "parquet"], default="csv", help="Export format")
    # args = parser.parse_args()

    # Instantiate and run the calculator
    calculator = BafoegCalculator()
    print("📦 Loading SOEP datasets...")
    calculator.load_all_data()

    print("🔧 Processing and merging data...")
    calculator.process_data()

    print(f"💾 Exporting")
    calculator.export(filename="student_parental_income.csv", format="csv")

    print("✅ Done.")
