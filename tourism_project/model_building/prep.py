"""Clean the tourism dataset and produce train/test splits for the training job."""

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"

# ---------------------------------------------------------------- Feature groups
# Genuinely continuous measurements -> median imputation
CONTINUOUS_COLS = ["Age", "DurationOfPitch", "NumberOfTrips", "MonthlyIncome"]

# Numeric codes with only a few distinct levels -> treated as categories (mode imputation)
DISCRETE_CAT_COLS = [
    "CityTier", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting",
]

# Text categories -> mode imputation
NOMINAL_CAT_COLS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# ---------------------------------------------------------------- Load
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ---------------------------------------------------------------- Clean
# 1. Drop identifier / index columns: unique per row, so they carry no pattern
#    and a tree could simply memorise them
drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print("Dropped columns:", drop_cols)

# 2. Fix known label inconsistency BEFORE de-duplication,
#    so rows differing only by "Fe Male"/"Female" are caught as duplicates
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3. Remove duplicate records (prevents the same row landing in both train and test)
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {before - df.shape[0]} duplicate row(s).")

# 4. Drop rows with a missing target (cannot learn from unlabelled rows)
df = df[df[TARGET_COL].notna()].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)          # clean 0/1 label

# 5. Impute missing values
#    - continuous columns : median (robust to outliers such as MonthlyIncome)
#    - categorical columns: mode   (the most frequent level is the only valid "centre")
for col in [c for c in CONTINUOUS_COLS if c in df.columns]:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in [c for c in DISCRETE_CAT_COLS + NOMINAL_CAT_COLS if c in df.columns]:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Remaining missing values:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------- Split
X = df.drop(columns=[TARGET_COL])   # 18 features
y = df[TARGET_COL]                  # target

# Stratify keeps the ~81/19 class ratio identical in train and test;
# random_state makes the split reproducible across local and CI runs
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Written to the working-directory root -> uploaded as the "data-splits" artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
print("Data prepared: train/test splits written.")
