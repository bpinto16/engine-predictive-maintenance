
# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os

import json

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/bpinto16/Predictive-Maintenance-HFSpace/engine_data.csv"
HF_REPO_ID = "bpinto16/Predictive-Maintenance-HFSpace"
RANDOM_STATE = 42
TEST_SIZE = 0.20
# Define target variable
TARGET_COL = "Engine Condition"

# Data Load
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Data Cleaning & Validation
print("\n--Data Cleaning.--")

# Initial Shape
print(f"Original Dataset Shape: {df.shape}")
print("Columns:", df.columns.tolist())
df.head(3)

print(f"Shape   : {df.shape}")

# MISSING VALUE HANDLING
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if missing_cols.empty:
    print("No missing values found — no action needed.")
else:
    print("Missing per column:\n", missing_cols.to_string())

df = df.dropna()
print(f"Shape after dropna : {df.shape}")


# DUPLICATE ROW CHECK
dup_count = df.duplicated().sum()
print(f"Duplicates found : {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed. New shape: {df.shape}")


# Outlier Treatment
# 'Coolant temp' has an implausible extreme reading (~195C) -- cap at the 99th
# percentile so it does not distort scaling/model fitting downstream.
cap = df["Coolant temp"].quantile(0.99)
n_clipped = (df["Coolant temp"] > cap).sum()
df["Coolant temp"] = df["Coolant temp"].clip(upper=cap)
print(f"Coolant temp: capped at 99th pct ({cap:.2f}), {n_clipped} rows capped")

print(f"Final cleaned shape: {df.shape}")


# Split into X (features) and y (target)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y,
)
print(f"Xtrain : {Xtrain.shape}   ytrain : {ytrain.shape}")
print(f"Xtest  : {Xtest.shape}    ytest  : {ytest.shape}")

PHYSICAL_LIMITS = {
    "Engine rpm":       {"physical_min": 0.0,  "physical_max": 3000.0},
    "Lub oil pressure": {"physical_min": 0.1,  "physical_max": 10.0},
    "Fuel pressure":    {"physical_min": 0.1,  "physical_max": 25.0},
    "Coolant pressure": {"physical_min": 0.1,  "physical_max": 10.0},
    "lub oil temp":     {"physical_min": 40.0, "physical_max": 130.0},
    "Coolant temp":     {"physical_min": 40.0, "physical_max": 120.0},
}

feature_ranges = {}
for col in Xtrain.columns:
    feature_ranges[col] = {
        **PHYSICAL_LIMITS.get(col, {}),
        "operating_min": float(Xtrain[col].quantile(0.01)),
        "operating_max": float(Xtrain[col].quantile(0.99)),
    }

with open("feature_ranges.json", "w") as f:
    json.dump(feature_ranges, f, indent=2)
print("Saved : feature_ranges.json")

output_files = {
    "Xtrain.csv"          : Xtrain,
    "Xtest.csv"           : Xtest,
    "ytrain.csv"          : ytrain.reset_index(drop=True).to_frame(),
    "ytest.csv"           : ytest.reset_index(drop=True).to_frame(),
}

for filename, data in output_files.items():
    data.to_csv(filename, index=False)
    print(f"  Saved : {filename}  {data.shape}")

for filename in output_files:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print(f"  Uploaded : {filename} to {HF_REPO_ID}")


api.upload_file(
    path_or_fileobj="feature_ranges.json",
    path_in_repo="feature_ranges.json",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
)
print(f"  Uploaded : feature_ranges.json to {HF_REPO_ID}")

print("\n preparation.py completed successfully.")
