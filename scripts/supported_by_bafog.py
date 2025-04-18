import os

import pandas as pd

import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick


# Add parent dir
import bootstrap
bootstrap.add_project_root_to_path()
from misc.utility_functions import get_config_path, load_config


config_path = get_config_path("config.json")
config = load_config(config_path)
destatis_dir = config["paths"]["data"]["destatis"]


# BIL002
def enrolled_students_over_time():
    students_data_path = os.path.join(destatis_dir, "21311-0001_en_flat.csv")
    df = pd.read_csv(students_data_path, sep=";")[["time", "value", "3_variable_attribute_label", "2_variable_attribute_code"]]

    # Rename columns and clean data in one step
    df.rename(columns={"time": "Year", "value": "BIL002"}, inplace=True)
    df["Year"] = df["Year"].str.split("-").str[0].astype(int)  # Clean year and convert to int
    df["BIL002"] = pd.to_numeric(df["BIL002"], errors='coerce')  # Convert to float, replacing "-" with NaN

    # Filter data
    df = df[(df["3_variable_attribute_label"] == "Total") & 
            (df["2_variable_attribute_code"].isin(["NATD", "NATA"]))]

    # Group by year and sum enrolled students
    df = df.groupby("Year")["BIL002"].sum().reset_index()

    # Sort by year descending
    df = df.sort_values(by="Year", ascending=False)

    return df



# Received FTAA/BAfoG
ftaa_path = os.path.join(destatis_dir, "21411-0001_en_flat.csv")
df = pd.read_csv(ftaa_path, sep=";")
cols = ["time", "2_variable_attribute_code", "value_unit", "value", "value_variable_code"]

df = df[cols]
df.rename(columns= {"time": "Year"}, inplace=True)

df = df.astype(
{
"Year": "int32", 
"2_variable_attribute_code": "string",
"value": "float",
"value_unit": "string",
"value_variable_code": "string",
 })

df = df[df["2_variable_attribute_code"] == "STUDENT"]

def extract_variable(df: pd.DataFrame, variable_code: str):
    df = df[df["value_variable_code"] == variable_code]
    df = df.sort_values(by="Year", ascending=False)
    df = df[["Year", "value"]]
    df.rename(columns={"value": f"{variable_code}"}, inplace=True)
    return df



enrolled_students = enrolled_students_over_time()
supported_persons = extract_variable(df, "PER010")
supported_persons_full = extract_variable(df, "PER011")
supported_persons_partial = extract_variable(df, "PER012")
supported_persons_avg_monthly = extract_variable(df, "PER013")
average_monthly_check = extract_variable(df, "PER014")

merged = enrolled_students.merge(supported_persons, on="Year", how="inner")
merged = merged.merge(supported_persons_full, on="Year", how="inner")
merged = merged.merge(supported_persons_partial, on="Year", how="inner")
merged = merged.merge(supported_persons_avg_monthly, on="Year", how="inner")

merged["Ratio PER010"] = merged["PER010"] / merged["BIL002"]
merged["Ratio PER011"] = merged["PER011"] / merged["BIL002"]
merged["Ratio PER012"] = merged["PER012"] / merged["BIL002"]




# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Plot in black and white using different line styles and markers
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(merged["Year"], merged["Ratio PER010"] * 100, marker="o", linestyle='-', linewidth=1.5, markersize=6, color='black', label="Supported Persons (PER010)")
ax.plot(merged["Year"], merged["Ratio PER011"] * 100, marker="s", linestyle='--', linewidth=1.5, markersize=6, color='black', label="Full Assistance (PER011)")
ax.plot(merged["Year"], merged["Ratio PER012"] * 100, marker="^", linestyle=':', linewidth=1.5, markersize=6, color='black', label="Partial Assistance (PER012)")

# Labels and title (placed below the figure in APA style)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Students supported by BAföG (%)", fontsize=12)

# Convert y-axis to percentage format
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Legend
ax.legend(loc='best', frameon=False, fontsize=10)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a horizontal grid
ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.7)

# Ensure tight layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add figure caption
# fig.text(0.5, 0.01, "Fraction of BIL002 Receiving Assistance Over Time.", ha='center', fontsize=12)

# Show the plot
plt.show()




merged["BIL002"] = merged["BIL002"].apply(lambda x: f"{x:,}")
merged["PER010"] = merged["PER010"].apply(lambda x: f"{x:,}")
merged["PER011"] = merged["PER011"].apply(lambda x: f"{x:,}")
merged["PER012"] = merged["PER012"].apply(lambda x: f"{x:,}")
merged["PER013"] = merged["PER013"].apply(lambda x: f"{x:,}")

merged["Ratio PER010"] = merged["Ratio PER010"].apply(lambda x: f"{x:.1%}".replace("%", r""))
merged["Ratio PER011"] = merged["Ratio PER011"].apply(lambda x: f"{x:.1%}".replace("%", r""))
merged["Ratio PER012"] = merged["Ratio PER012"].apply(lambda x: f"{x:.1%}".replace("%", r""))

cols = ["Year", "BIL002", "PER010", "PER011", "PER012", "Ratio PER010", "Ratio PER011", "Ratio PER012"]

latex_df = merged[cols]
latex_df = latex_df.rename(columns={
    "Ratio PER010": "Ratio PER010 (\%)",
    "Ratio PER011": "Ratio PER011 (\%)",
    "Ratio PER012": "Ratio PER012 (\%)"
})





# Index(['Year', 'BIL002', 'PER010', 'PER011', 'PER012', 'PER013',
#        'Ratio PER010', 'Ratio PER011', 'Ratio PER012'],
#       dtype='object')


# Variable codes 
# PER 010 | Supported persons
# PER 011 | Persons receiving full assistance payments
# PER 012 | Persons receiving partial assistance payments
# PER 013 | Supported persons (average monthly stock)
# PER 014 | Average monthly assistance payment per person

# def main():
#     pass
#
# if __name__ == "__main__":

