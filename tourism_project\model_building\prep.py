"""Clean the tourism dataset and produce train/test splits for the training job."""

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"

# ---------------------------------------------------------------- Load
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ---------------------------------------------------------------- Clean
# 1. Drop identifier / index columns (no predictive value)
drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print("Dropped columns:", drop_cols)

# 2. Remove duplicate records
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {before - df.shape[0]} duplicate row(s).")

# 3. Fix known label inconsistency in Gender ("Fe Male" -> "Female")
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 4. Drop rows where the target is missing (cannot learn from them)
df = df[df[TARGET_COL].notna()].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# 5. Impute missing values: median for numeric, mode for categorical
numeric_cols = df.select_dtypes(include=["number"]).columns.drop(TARGET_COL)
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Remaining missing values:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------- Split
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
print("Data prepared: train/test splits written.")
