import os
import pandas as pd
import matplotlib.pyplot as plt

# Add parent dir
import bootstrap
bootstrap.add_project_root_to_path()
from misc.utility_functions import get_config_path, load_config

# ---- Functions ----

def load_data(file: str):
    """Loads CSV data using the configured directory path."""
    config_path = get_config_path("config.json")
    config = load_config(config_path)
    csv_path = os.path.join(config["paths"]["data"]["destatis"], file)
    return pd.read_csv(csv_path, sep=";")

def clean_cpi_data(df):
    """Filters and processes the CPI data dynamically without hardcoding CPI for 2023."""
    
    df = df[df["value_variable_label"] == "Consumer price index"]
    df = df.rename(columns={"time": "Year", "value": "CPI"})[["Year", "CPI"]]
    df = df.sort_values(by="Year").astype({"Year": "int32", "CPI": "float"})
    
    # Dynamically get the CPI for 2023
    cpi_2023 = df.loc[df["Year"] == 2023, "CPI"].values[0]  # Extract the actual value

    # Normalize CPI values to 2023 prices
    df["CPI 2023 factor"] = 1 / (df["CPI"] / cpi_2023)
    # df["CPI 2023 factor"] = 1 / df["CPI 2023"]
    
    return df  # Ensure function returns the cleaned DataFrame
    
def clean_payout_data(df):
    """Filters and processes the payout data."""
    df = df[(df["value_variable_code"] == "PER014") & 
            (df["2_variable_attribute_label"] == "Students")]
    
    df = df.rename(columns={"time": "Year", "value": "Payout (EUR)"})[["Year", "Payout (EUR)"]]
    df = df.sort_values(by="Year").astype({"Year": "int32", "Payout (EUR)": "int"})
    
    return df

def clean_finance_data(df): 
    df = df[(df["value_variable_code"] == "FIN001") &  
            (df["2_variable_attribute_label"] == "Students")]

    df = df.rename(columns={"time": "Year", "value": "Financial Expenditure (EUR 1000)"})[["Year", "Financial Expenditure (EUR 1000)"]]
    df = df.sort_values(by="Year").astype({"Year": "int32", "Financial Expenditure (EUR 1000)": "int"})
    return df

def merge_data(payout_df, cpi_df):
    """Merges payout data with CPI data and calculates real payout values."""
    merged = payout_df.merge(cpi_df, how="inner", on="Year")
    merged["Payout (EUR) 2023 prices"] = merged["Payout (EUR)"] * merged["CPI 2023 factor"]
    return merged

def save_to_latex(df, filename="payout_over_time.tex"):
    """Formats and saves the DataFrame as a LaTeX table with proper rounding and integer conversion."""
    
    # Convert integer-like columns
    df["Year"] = df["Year"].astype(int)
    df["Payout (EUR)"] = df["Payout (EUR)"].astype(int)
    df["CPI"] = df["CPI"].astype(int)  # If CPI should be an integer
    df["Payout (EUR) 2023 prices"] = df["Payout (EUR) 2023 prices"].round().astype(int)

    df["CPI 2023 factor"] = df["CPI 2023 factor"].round(3)

    cols = ["Year", "CPI", "CPI 2023 factor", "Payout (EUR)", "Payout (EUR) 2023 prices", "Financial Expenditure (EUR 1000)", "Financial Expenditure (EUR 1000) 2023 prices"]
    df = df[cols]

    # Save to LaTeX
    df.to_latex(filename, index=False, float_format="%.3f")  # Ensures floats use 3 decimal places

def plot_payout(merged_df):
    """Plots nominal and real payout over time, including CPI for comparison."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize CPI by setting the first year's CPI to match its payout
    first_year = merged_df["Year"].min()
    first_cpi = merged_df.loc[merged_df["Year"] == first_year, "CPI"].values[0]
    cpi_scaled = merged_df["CPI"] / first_cpi * merged_df["Payout (EUR)"].iloc[0]

    # Plot payout values
    ax.plot(merged_df["Year"], merged_df["Payout (EUR) 2023 prices"], 
            marker="s", linestyle='--', linewidth=1.5, color='black', 
            label="Payout (EUR) 2023 prices")
    
    ax.plot(merged_df["Year"], merged_df["Payout (EUR)"], 
            marker="o", linestyle='-', linewidth=1.5, color='black', 
            label="Nominal Payout (EUR)")

    # Plot CPI (scaled)
    ax.plot(merged_df["Year"], cpi_scaled, 
            marker="^", linestyle='-.', linewidth=1.5, color='gray', 
            label="CPI (scaled for comparison)")

    ax.legend(loc='best', frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()

# ---- Execution ----

def main():
    cpi_df = clean_cpi_data(load_data("61111-0001_en_flat.csv"))
    payout_df = clean_payout_data(load_data("21411-0001_en_flat.csv"))

    merged_df = merge_data(payout_df, cpi_df)

    # Finance expenditures
    finance_exp_df = clean_finance_data(load_data("21411-0001_en_flat.csv"))
    merged_df = merged_df.merge(finance_exp_df, how="inner", on="Year")
    merged_df["Financial Expenditure (EUR 1000) 2023 prices"] = merged_df["Financial Expenditure (EUR 1000)"] * merged_df["CPI 2023 factor"]


    plot_payout(merged_df)
    save_to_latex(merged_df)


if __name__ == "__main__":
    main()
