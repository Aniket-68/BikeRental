from pathlib import Path
import pandas as pd
from validation import validate_data
from features import prepare_features
from split_data import temporal_split

# Resolve the project folder from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(f"Project root: {PROJECT_ROOT}")
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"

# # Fail clearly if the expected dataset is missing.
if not CSV_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

df=pd.read_csv(CSV_PATH)
# Preview the dataset first 5 rows, shape, columns, summary statistics, and info.
print(df.head())

# print the shape of the dataset (number of rows and columns)
print(df.shape)

# print the column names of the dataset
print(df.columns)

# print the summary statistics of the dataset
print(df.describe())

# print the info of the dataset
print(df.info())

# Missing values in the dataset
missing_values = df.isna().sum()
print("Missing values in the dataset:")
print(missing_values)

validate_data(df)
print("Data validation passed successfully.")

print("Preparing features...")
X, y, timestamps = prepare_features(df)
print("Features prepared successfully.")
print("Feature matrix (X):")
print(X.head())
print(X.shape)
print("Target variable (y):")
print(y.head())
print(y.shape)

print("Timestamps:")
print(timestamps.head())
print(timestamps.shape)

# Check that the lengths of X, y, and timestamps are equal and that timestamps are in chronological order.
assert len(X) == len(y) == len(timestamps)
assert timestamps.is_monotonic_increasing
print("All checks passed: lengths are equal and timestamps are in chronological order.")


# Split the data into train, validation, and test sets.
print("Splitting data into train, validation, and test sets...")
splits = temporal_split(X, y, timestamps)

# Each dictionary entry holds a tuple: (features, target).
X_train, y_train = splits["train"]
X_val, y_val = splits["validation"]
X_test, y_test = splits["test"]

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Every observation must belong to exactly one split.
assert len(X_train) + len(X_val) + len(X_test) == len(X)

# Verify that no row index occurs in two splits.
assert X_train.index.intersection(X_val.index).empty
assert X_train.index.intersection(X_test.index).empty
assert X_val.index.intersection(X_test.index).empty

# Keep validation and test observations out of ALL fitting operations,
# including encoders, scalers, and missing-value imputers.