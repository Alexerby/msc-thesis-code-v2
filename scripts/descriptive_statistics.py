import pandas as pd
import os

# Adjust path
PATH = os.path.expanduser("~/Downloads/dataframes/")
df = pd.read_excel(os.path.join(PATH, "bafoeg_results.xlsx"), sheet_name="full")

# ============================================================================
print("Eligibility counts by year")
print("===========================================================================")
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
).round(2)

# --- Reported SOEP BAföG stats (plc0168_h) ---
reported_stats = reported.groupby("syear")["plc0168_h"].agg(
    mean_award_soep = "mean",
    std_award_soep = "std",
).round(2)

# --- Join for comparison ---
comparison = award_stats.join(reported_stats, how="outer")

# Add difference column (model - reported)
comparison["diff_model_vs_soep"] = (comparison["mean_award_model"] - comparison["mean_award_soep"]).round(2)
comparison["percent_error"] = ((comparison["mean_award_model"] - comparison["mean_award_soep"]) / comparison["mean_award_soep"] * 100).round(2)

# ============================================================================
print("Theoretical vs. Reported BAföG Awards")
print("===========================================================================")
print(comparison.fillna("-"))

# ============================================================================
print("\nAdditional Descriptive Statistics (Eligible Only)")
print("===========================================================================")

# Additional stats for eligible students
extra_stats = eligible.groupby("syear")[["monthly_parental_contribution_split", "student_excess_income"]].agg(
    parental_contrib_mean=("monthly_parental_contribution_split", "mean"),
    parental_contrib_std=("monthly_parental_contribution_split", "std"),
).round(2)

print(extra_stats)


# ============================================================================
print("\nStudent Income Statistics (Eligible Only)")
print("===========================================================================")

# Descriptive stats for various student income steps
student_income_stats = eligible.groupby("syear")[[
    "gross_annual_income",
    "income_post_si",
    "net_annual_student_income",
    "student_total_allowance",
    "student_excess_income"
]].agg(
    gross_income_mean=("gross_annual_income", "mean"),
    gross_income_std=("gross_annual_income", "std"),
    post_si_income_mean=("income_post_si", "mean"),
    post_si_income_std=("income_post_si", "std"),
    net_income_mean=("net_annual_student_income", "mean"),
    net_income_std=("net_annual_student_income", "std"),
    total_allowance_mean=("student_total_allowance", "mean"),
    total_allowance_std=("student_total_allowance", "std"),
    excess_income_mean=("student_excess_income", "mean"),
    excess_income_std=("student_excess_income", "std"),
).round(2)

print(student_income_stats)
