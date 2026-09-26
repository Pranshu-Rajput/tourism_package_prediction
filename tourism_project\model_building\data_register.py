"""Register the Wellness Tourism dataset by validating it and printing a summary."""

import os
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# ---------------------------------------------------------------- Load
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Upload tourism.csv first.")

df = pd.read_csv(RAW_PATH)

# ---------------------------------------------------------------- Validate schema
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

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

print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts(dropna=False))
print(df["ProdTaken"].value_counts(normalize=True).round(4))
