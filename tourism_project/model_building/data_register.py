"""Register the Wellness Tourism dataset by validating it and printing a summary."""

import os
import pandas as pd

# Path is relative to the repo root (same in local Jupyter and on the GitHub runner)
RAW_PATH = "tourism_project/data/tourism.csv"

# ---------------------------------------------------------------- Load
# Fail with a clear message if the dataset was not added to the repo
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Add tourism.csv first.")

df = pd.read_csv(RAW_PATH)

# ---------------------------------------------------------------- Validate schema
# The 20 columns listed in the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Missing columns are a hard failure -> the GitHub job turns red and later jobs are skipped
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Extra columns (e.g. a saved index) are only a warning; prep.py drops them
extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
if extra:
    print(f"Note: extra column(s) present and will be handled during preparation: {extra}")

# ---------------------------------------------------------------- Summary
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nColumns:", list(df.columns))

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

# Number of distinct values -> identifies numeric columns that behave as categories
print("\nUnique values per column:")
print(df.nunique().sort_values())

# Class balance of the target (counts and proportions)
print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts(dropna=False))
print(df["ProdTaken"].value_counts(normalize=True).round(4))
