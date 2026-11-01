from pathlib import Path
import pandas as pd
from validation import validate_data

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