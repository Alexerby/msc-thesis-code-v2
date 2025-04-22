# Built-in packages
import math

# Package imports
import numpy as np
import pandas as pd

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
        self._load_income_tax_sheet()


    def _load_income_tax_sheet(self):
        # Load tax table
        tax_brackets_data = SOEPStatutoryInputs("Income Tax")
        tax_brackets_data.load_dataset(columns=lambda col: True)
        
        # Clean and convert: Replace commas, then coerce to numeric
        raw_df = tax_brackets_data.data
        cleaned_df = raw_df.apply(lambda col: col.astype(str).str.replace(",", "."))  # fix decimal commas
        numeric_df = cleaned_df.apply(pd.to_numeric, errors="coerce")  # convert strings to numbers

        # Save cleaned version
        self.tax_rate_df = numeric_df

        # Optional: warn if any values are still NaN
        if self.tax_rate_df.isna().any().any():
            raise ValueError("🚨 Tax rate table contains NaNs — check formatting.")

    def load_all_data(self):
        self._load_dataset("ppathl", ["pid", "hid", "syear", "gebjahr", "sex", "gebmonat", "parid", "partner"])
        self._load_dataset("pl", ["pid", "syear", "plg0012_h", "plh0258_h"])
        self._load_dataset("pgen", ["pid", "syear", "pglabgro"])
        self._load_dataset("bioparen", ["pid", "fnr", "mnr"])
        self._load_dataset("regionl", ["hid", "bula", "syear"])

    def process_data(self):
        df = self.datasets["ppathl"].data.copy()
        df = df[df["syear"] >= 2002] # Filter for only 2002 onwards (post Euro implementation)
        df = self._add_demographics(df)
        df = self._merge_education(df)
        df = self._merge_income(df)
        df = self._filter_students(df)
        df = self._merge_parental_links(df)
        df = self._merge_parental_incomes(df, require_both_parents=True)
        df = self._apply_lump_sum_tax_deduction(df)
        df = self._apply_allowances_for_social_insurance_payments(df)
        df = self.calculate_income_tax(df)

        self.df = df

    def _load_dataset(self, key, columns):
        self.datasets[key] = SOEPDataHandler(key)
        self.datasets[key].load_dataset(columns)

    def _add_demographics(self, df):
        ppathl = self.datasets["ppathl"].data[["pid", "hid", "syear", "gebjahr", "gebmonat"]].copy()
        ppathl["age"] = ppathl["syear"] - ppathl["gebjahr"]
        ppathl["age"] = ppathl["age"] - (ppathl["gebmonat"] > 6).astype(int)
        df = df.merge(ppathl[["pid", "syear", "age"]], on=["pid", "syear"], how="left")

        # Find what state (Bundesland) the individual lives in 
        regionl = self.datasets["regionl"].data[["hid", "syear", "bula"]].copy()
        df = df.merge(regionl[["hid", "syear", "bula"]], on=["hid", "syear"], how="left")

        return df

    def calculate_income_tax(self, df):
        """
        NOTE: 
            Church tax is simulated only for individuals who 
            self-report Catholic or Protestant affiliation. 
            This likely underestimates true church tax 
            incidence due to unobserved registration status.

            TODO: Change so that individuals who have not 
            responded to this question are assumed to be 
            subject to church tax.
        """
        def compute_row(row):
            year = int(row["syear"])
            income = row["parental_income_post_insurance_allowance"]
            bula = row.get("bula", None)
            religion_code = row.get("plh0258_h", None)

            income_tax = self.compute_income_tax(income, year)

            # Determine church tax
            if income_tax is None:
                church_tax = None

            elif religion_code in [1, 2]:
                # Catholic or Protestant
                church_rate = 0.08 if bula in [8, 9] else 0.09 if not pd.isna(bula) else 0.09
                church_tax = math.floor(income_tax * church_rate)

            elif religion_code in self.invalid_codes:
                # Missing/filtered response — assume default 9%
                church_tax = math.floor(income_tax * 0.09)

            else:
                # All others — no church tax
                church_tax = 0

            return income_tax, church_tax

        tax_result = df.apply(compute_row, axis=1, result_type="expand")
        df[["parental_income_tax", "parental_church_tax"]] = tax_result

        df["parental_income_post_income_tax"] = (
            df["parental_income_post_insurance_allowance"]
            - df["parental_income_tax"]
            - df["parental_church_tax"]
        )

        return df

    #TODO: Add solidarity surcharge
    def compute_income_tax(self, income, year):
        if pd.isna(income):
            return None

        match = self.tax_rate_df[self.tax_rate_df["year"] == year]
        if match.empty:
            raise ValueError(f"No tax parameters found for year {year}")

        params = match.iloc[0].apply(pd.to_numeric, errors="coerce")
        income = math.floor(income)

        A = params["basic_allowance"]
        B = params["first_bracket_upper"]
        C = params["second_bracket_upper"]
        D = params["top_rate_threshold"]

        if income <= A:
            tax = 0
        elif A < income <= B:
            y = (income - A) / 10_000
            tax = (params["bracket2_a"] * y + params["bracket2_b"]) * y
        elif B < income <= C:
            z = (income - B) / 10_000
            tax = (params["bracket3_a"] * z + params["bracket3_b"]) * z + params["bracket3_c"]
        elif C < income <= D:
            tax = params["rate_4"] * income - params["deduct_4"]
        else:
            tax = params["rate_5"] * income - params["deduct_5"]

        return math.floor(tax) if not pd.isna(tax) else None


    def _apply_allowances_for_social_insurance_payments(self, df):
        """
        Applies the social insurance deduction to adjusted parental income.
        BAFöG § 21, (2) 1 states that 22.3% is deducted to account for 
        social insurance payments.
        
        TODO: Update rate dynamically based on statutory changes (BAFöGÄndG).
        """
        if "adjusted_parental_income" not in df.columns:
            raise KeyError("Missing 'adjusted_parental_income' column. Ensure previous steps ran correctly.")
        
        df["parental_income_post_insurance_allowance"] = df["adjusted_parental_income"] * (1 - 0.223)
        return df

    def _apply_lump_sum_tax_deduction(self, df):
        """
        Merges in the year-specific Werbungskostenpauschale and 
        applies it to parental income.

        PAUSCHBETRÄGE FUR WERBUNGSKOSTEN

        Legal Basis	                    BGBl. Citation
        -------------------------------------------------------
        Conversion to Euro	            BGBl. I 2000, p. 1790
        Alterseinkünftegesetz	        BGBl. I 2004, p. 1427
        Steuervereinfachungsgesetz 2011	BGBl. I 2011, p. 2131
        Steuerentlastungsgesetz 2022	BGBl. I 2022, p. 749
        Inflationsausgleichsgesetz 2022	BGBl. I 2022, p. 2294
        -------------------------------------------------------
        """
        statutory_input = SOEPStatutoryInputs("Werbungskostenpauschale")
        statutory_input.load_dataset(columns=["Year", "werbungskostenpauschale"])
        deduction_df = statutory_input.data.rename(columns={"Year": "syear"}) 
        df = df.merge(deduction_df, on="syear", how="left")
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

    def _merge_parental_incomes(self, df, require_both_parents=False):
        pgen = self.datasets["pgen"].data.copy()
        pgen["pglabgro"] = pgen["pglabgro"].where(~pgen["pglabgro"].isin(self.invalid_codes), np.nan)
        pgen.rename(columns={"pglabgro": "parent_income"}, inplace=True)

        father_income = pgen.rename(columns={"pid": "fnr", "parent_income": "father_income"})
        mother_income = pgen.rename(columns={"pid": "mnr", "parent_income": "mother_income"})

        df = df.merge(father_income[["fnr", "syear", "father_income"]], on=["fnr", "syear"], how="left")
        df = df.merge(mother_income[["mnr", "syear", "mother_income"]], on=["mnr", "syear"], how="left")

        if require_both_parents:
            df = df[df["father_income"].notna() & df["mother_income"].notna()]
        else:
            # Keep rows with at least one parent
            df = df[df[["father_income", "mother_income"]].notna().any(axis=1)]

        df["parental_income"] = df[["father_income", "mother_income"]].sum(axis=1, min_count=1)
        df["parental_annual_income"] = df["parental_income"] * 12

        return df

    def export(self, filename: str, format: ExportType = "csv"):
        if self.df is None:
            raise ValueError("Data has not been processed yet.")
        export_data(format, df=self.df, output_filename=filename)



if __name__ == "__main__":

    calculator = BafoegCalculator()
    print("📦 Loading SOEP datasets...")
    calculator.load_all_data()

    print("🔧 Processing and merging data...")
    calculator.process_data()

    print(f"💾 Exporting")
    calculator.export(filename="student_parental_income.csv", format="csv")

    print("✅ Done.")
