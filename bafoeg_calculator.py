
# Package imports
import numpy as np

# Project imports
from misc.utility_functions import Literal, export_data
from data_handler import SOEPDataHandler, SOEPStatutoryInputs


ExportType = Literal["csv", "excel"]


#TODO: 
# Currently not looking at income of 
# previous year for parents or spouse.
# BAföG § 24 (1).

# TODO: 
# Add spouse income. Make this contingent on the being in a "relationship in law". 
# as this makes sense.


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
        df = df[df["syear"] >= 2002] # Filter for only 2002 onwards
        df = self._add_demographics(df)
        df = self._merge_education(df)
        df = self._merge_income(df)
        df = self._filter_students(df)
        df = self._merge_parental_links(df)
        df = self._merge_parental_incomes(df)
        df = self._apply_lump_sum_tax_deduction(df)
        self.df = df

    def _apply_lump_sum_tax_deduction(self, df):
        """
        Merges in the year-specific Werbungskostenpauschale and 
        applies it to parental income.
        """
        # Load the statutory deduction table
        statutory_input = SOEPStatutoryInputs("Werbungskostenpauschale")
        statutory_input.load_dataset(columns=["Year", "werbungskostenpauschale"])

        # Rename for clarity before merge
        deduction_df = statutory_input.data.rename(columns={"Year": "syear"}) 

        # Merge the deduction into the main dataframe
        df = df.merge(deduction_df, on="syear", how="left")

        # You can now use df["werbungskostenpauschale"] wherever needed, e.g.:
        df["adjusted_parental_income"] = df["parental_annual_income"] - df["werbungskostenpauschale"]

        return df

    def _merge_education(self, df):
        edu = self.datasets["pl"].data
        return df.merge(edu, on=["pid", "syear"], how="left")

    def _merge_income(self, df):
        income = self.datasets["pgen"].data.copy()
        income["pglabgro"] = income["pglabgro"].where(~income["pglabgro"].isin(self.invalid_codes), np.nan)
        income.rename(columns={"pglabgro": "gross_monthly_income"}, inplace=True)
        income["gross_annual_income"] = income["gross_monthly_income"] * 12
        return df.merge(income[["pid", "syear", "gross_annual_income"]], on=["pid", "syear"], how="left")

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
        df["parental_income"] = df[["father_income", "mother_income"]].sum(axis=1, min_count=1)
        df["parental_annual_income"] = df["parental_income"] * 12
        return df

    def export(self, filename: str, format: ExportType = "csv"):
        if self.df is None:
            raise ValueError("Data has not been processed yet.")
        export_data(format, df=self.df, output_filename=filename)



if __name__ == "__main__":
    # Instantiate and run the calculator
    calculator = BafoegCalculator()
    print("📦 Loading SOEP datasets...")
    calculator.load_all_data()

    print("🔧 Processing and merging data...")
    calculator.process_data()

    print(f"💾 Exporting")
    calculator.export(filename="student_parental_income.csv", format="csv")

    print("✅ Done.")
