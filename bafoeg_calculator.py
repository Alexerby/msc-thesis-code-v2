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
        self.main_df = None
        self._load_income_tax_sheet()
        self._load_soli_thresholds()

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
        self._load_dataset("hgen", ["hid", "hgtyp1hh", "syear"])

    def process_data(self):
        # Starting df
        df = self.datasets["ppathl"].data.copy()
    
        # Add information
        df = df[df["syear"] >= 2002] # Filter for only 2002 onwards (post Euro implementation)
        df = self._add_demographics(df)
        df = self._merge_education(df)
        df = self._merge_income(df)
        df = self._filter_students(df)
        df = self._merge_parental_links(df)
        df = self._merge_parental_incomes(df, require_both_parents=False)
        df = self._apply_lump_sum_tax_deduction(df)
        df = self._apply_allowances_for_social_insurance_payments(df)
        df = self.calculate_income_tax(df)
        df = self._flag_parent_relationship(df)
        df = self._apply_basic_allowance_parents(df)
        # df = self._get_siblings(df)
        self.main_df = df

    # def _get_siblings(self, df):
    #     # TODO: Create new dataframe for siblings, get income etcetera bla bla bla
    #     bioparen = self.datasets["bioparen"].data[["pid", "fnr", "mnr"]]
    #
    #     # Rename for clarity
    #     siblings = bioparen.rename(columns={"pid": "sibling_pid"})
    #
    #     # Match siblings: same parents (full siblings only)
    #     df = df.merge(siblings, on=["fnr", "mnr"], how="inner")
    #
    #     # Exclude self
    #     df = df[df["pid"] != df["sibling_pid"]]
    #
    #     return df


    def _apply_basic_allowance_parents(self, df):
        # Load and prep the allowance table
        allowances = SOEPStatutoryInputs("Basic Allowances - § 25")
        allowances.load_dataset(columns=lambda col: True)

        table = allowances.data.rename(columns={
            "Valid from": "valid_from",
            "§ 25 (1) 1": "allowance_joint",
            "§ 25 (1) 2": "allowance_single"
        })

        table["valid_from"] = pd.to_datetime(table["valid_from"])
        table = table[["valid_from", "allowance_joint", "allowance_single"]].sort_values("valid_from")
        table[["allowance_joint", "allowance_single"]] = table[["allowance_joint", "allowance_single"]].apply(pd.to_numeric, errors="coerce")

        # Add a datetime version of syear to df
        df["syear_date"] = pd.to_datetime(df["syear"].astype(str) + "-01-01")
        df = df.sort_values("syear_date")

        # Merge: for each syear, get the latest applicable allowance
        df = pd.merge_asof(
            df,
            table,
            left_on="syear_date",
            right_on="valid_from",
            direction="backward"
        )

        # Apply the appropriate allowance
        df["parental_income_post_basic_allowance"] = np.maximum(
            np.where(
                df["parents_are_partnered"],
                df["parental_income"] - df["allowance_joint"],
                df["parental_income"] - (2 * df["allowance_single"])
            ),
            0
        )

        # Clean up temporary columns
        df.drop(columns=["valid_from", "syear_date", "allowance_joint", "allowance_single"], inplace=True)

        return df

    def _flag_parent_relationship(self, df):
        # Load parid (partner ID) from ppathl
        ppathl = self.datasets["ppathl"].data[["pid", "syear", "parid"]].copy()

        father_parid = ppathl.rename(columns={"pid": "fnr", "parid": "parid_of_father"})
        df = df.merge(father_parid, on=["fnr", "syear"], how="left")

        mother_parid = ppathl.rename(columns={"pid": "mnr", "parid": "parid_of_mother"})
        df = df.merge(mother_parid, on=["mnr", "syear"], how="left")

        # Relationship TRUE if father and mother are partnered with each other
        df["parents_are_partnered"] = (df["parid_of_father"] == df["mnr"]) & (df["parid_of_mother"] == df["fnr"])

        # Drop intermediate columns if you want a clean df
        df.drop(columns=["parid_of_father", "parid_of_mother"], inplace=True)

        return df

    def _load_dataset(self, key, columns):
        self.datasets[key] = SOEPDataHandler(key)
        self.datasets[key].load_dataset(columns)

    def _add_demographics(self, df):

        # Calculate age and merge into df
        ppathl = self.datasets["ppathl"].data[["pid", "hid", "syear", "gebjahr", "gebmonat"]].copy()
        ppathl["age"] = ppathl["syear"] - ppathl["gebjahr"]
        ppathl["age"] = ppathl["age"] - (ppathl["gebmonat"] > 6).astype(int)
        df.drop(columns=["gebjahr", "gebmonat"], inplace=True)
        df = df.merge(ppathl[["pid", "syear", "age"]], on=["pid", "syear"], how="left")

        # Find what state (Bundesland) the individual lives in 
        regionl = self.datasets["regionl"].data[["hid", "syear", "bula"]].copy()
        df = df.merge(regionl[["hid", "syear", "bula"]], on=["hid", "syear"], how="left")

        # Find household type 
        hgen = self.datasets["hgen"].data[["hid", "syear", "hgtyp1hh"]].copy()
        df = df.merge(hgen[["hid", "syear", "hgtyp1hh"]], on=["hid", "syear"], how="left")

        return df

    def calculate_income_tax(self, df):
        """
        NOTE: 
            People that are have no registered beliefs are assumed 
            to be paying the church tax rate in line with their state.
        """
        def compute_row(row):
            year = int(row["syear"])
            income = row["parental_income_post_insurance_allowance"]
            bula = row.get("bula", None)
            religion_code = row.get("plh0258_h", None)
            household_type = row.get("hgtyp1hh", None)

            income_tax = self._compute_income_tax(income, year)

            # Church tax
            if income_tax is None:
                church_tax = None
                soli = None
            else:
                if religion_code in {1, 2} | self.invalid_codes:
                    church_rate = 0.08 if bula in [8, 9] else 0.09
                    church_tax = math.floor(income_tax * church_rate)
                else:
                    church_tax = 0

                # Solidarity surcharge
                soli = self._compute_solidarity_surcharge(income_tax, household_type, year)

            return income_tax, church_tax, soli

        tax_result = df.apply(compute_row, axis=1, result_type="expand")
        df[["parental_income_tax", "parental_church_tax", "parental_soli"]] = tax_result

        df["parental_income_post_income_tax"] = (
            df["parental_income_post_insurance_allowance"]
            - df["parental_income_tax"]
            - df["parental_church_tax"]
        )

        df.drop(columns=["plh0258_h"], inplace=True)

        return df

    def _load_soli_thresholds(self):
        soli = SOEPStatutoryInputs("Solidaritätszuschlag")
        soli.load_dataset(columns=lambda col: True)
        df = soli.data.rename(columns={
            "In force": "year",
            "§ 32a Abs. 5 & 6 (joint)": "threshold_joint",
            "Otherwise (single)": "threshold_single"
        })
        df[["year", "threshold_joint", "threshold_single"]] = df[["year", "threshold_joint", "threshold_single"]].apply(pd.to_numeric, errors="coerce")
        self.soli_thresholds = df.set_index("year")

    def _compute_solidarity_surcharge(self, income_tax, household_type, year):
        """
        TODO: 
            Double check this. Think it's not considering the taxes 
            of the mother/father respectively when finding the threshold, it's is 
            just looking at them as one unit.
        """
        if pd.isna(income_tax):
            return None

        joint_filing = household_type in {2, 4, 5, 6}

        try:
            thresholds = self.soli_thresholds.loc[year]
        except KeyError:
            # Use nearest past year if no exact match
            past_years = self.soli_thresholds.index[self.soli_thresholds.index <= year]
            if past_years.empty:
                return math.floor(income_tax * 0.055)  # fallback for pre-2020
            thresholds = self.soli_thresholds.loc[past_years.max()]

        threshold = thresholds["threshold_joint"] if joint_filing else thresholds["threshold_single"]

        return 0 if income_tax <= threshold else math.floor(income_tax * 0.055)

    def _compute_income_tax(self, income, year):
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
        df.drop(columns=["werbungskostenpauschale"], inplace=True)
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
        return df[df["plg0012_h"] == 1].drop(columns=["plg0012_h"])

    def _merge_parental_links(self, df):
        parent_links = self.datasets["bioparen"].data
        return df.merge(parent_links, on="pid", how="left")

    def _merge_parental_incomes(self, df, require_both_parents=True):
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


if __name__ == "__main__":

    calculator = BafoegCalculator()
    print("\n📦 Loading SOEP datasets...")
    calculator.load_all_data()

    print("\nProcessing and merging data...")
    calculator.process_data()

    if calculator.main_df is not None:
        print(f"Exporting")
        export_data("excel", calculator.main_df, "student_parental_income", sheet_name="main_df")
        print("✅ Done.")

