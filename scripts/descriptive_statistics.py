import pandas as pd
import os

# Adjust path
PATH = os.path.expanduser("~/Downloads/dataframes/")
df = pd.read_excel(os.path.join(PATH, "bafoeg_results.xlsx"), sheet_name="full")

# Eligibility summary per year
print("================================================================")
print("Eligibility counts by year")
print("================================================================")
print(df.groupby("syear")["eligible_for_bafoeg"].value_counts().unstack().fillna(0).astype(int))
print("\n")

# === FILTER only eligible students ===
eligible = df[df["eligible_for_bafoeg"] == 1].copy()

# === FILTER only students who report receiving BAföG in SOEP ===
reported = df[df["plc0168_h"] > 0].copy()

# --- Theoretical award stats (eligible only) ---
award_stats = eligible.groupby("syear")["monthly_award"].agg(
    mean_award_model = "mean",
    std_award_model = "std",
    # count_model = "count"
).round(2)

# --- Reported SOEP BAföG stats (plc0168_h) ---
reported_stats = reported.groupby("syear")["plc0168_h"].agg(
    mean_award_soep = "mean",
    std_award_soep = "std",
    # count_soep = "count"
).round(2)

# --- Join for comparison ---
comparison = award_stats.join(reported_stats, how="outer")

# Add difference column (model - reported)
comparison["diff_model_vs_soep"] = (comparison["mean_award_model"] - comparison["mean_award_soep"]).round(2)

# Fill missing values for display only
print("================================================================")
print("Theoretical vs. Reported BAföG Awards")
print("================================================================")
print(comparison.fillna("-"))

