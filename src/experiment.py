from pathlib import Path
import pandas as pd
from validation import validate_data
from features import prepare_features
from split_data import temporal_split
from evaluation import regression_metrics
from baseline import train_and_evaluate
import mlflow
import mlflow.sklearn
from model import build_model
from mlflow import MlflowClient
import numpy as np

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


# Place this AFTER creating X_train, y_train, X_val, and y_val.

# Store experiment metadata in a database inside the project folder.
database_path = (PROJECT_ROOT / "mlflow.db").as_posix()
mlflow.set_tracking_uri(f"sqlite:///{database_path}")

# Group our baseline and future model runs under one experiment.
mlflow.set_experiment("BikeDemand")

with mlflow.start_run(run_name="baseline-mean") as run:
    # Record how this model was configured.
    mlflow.log_params({
        "model": "DummyRegressor",
        "strategy": "mean",
        "split": "chronological",
        "validation_start": "2012-07-01",
        "test_start": "2012-10-01",
    })

    # Train and evaluate a baseline model using the training and validation sets.
    print("Training and evaluating baseline model...")
    baseline_model, baseline_metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val
    )
    print("Baseline model trained and evaluated successfully.")

    print("Baseline model metrics:")
    for metric_name, metric_value in baseline_metrics.items():
        print(f"{metric_name}: {metric_value}")

    print("First five predictions:", baseline_model.predict(X_val.head(1)))

    # Store evaluation results against this run.
    mlflow.log_metrics({
        "validation_mae": baseline_metrics["MAE"],
        "validation_rmse": baseline_metrics["RMSE"],
        "validation_r2": baseline_metrics["R2"],
    })

    # Save the fitted baseline so we can load it later.
    mlflow.sklearn.log_model(
        sk_model=baseline_model,
        name="baseline_model",
        input_example=X_train.head(3),
    )

    print("Run ID:", run.info.run_id)
    print("Validation metrics:", baseline_metrics)



# Mlflow Random Forest Regressor
print("Training and evaluating Random Forest model...")
# Reuse the existing MLflow tracking URI and BikeDemand experiment.
# This is a separate run so we can compare it with baseline-mean.
with mlflow.start_run(run_name="random-forest") as run:
    model = build_model()

    # Log the actual model settings rather than duplicating them manually.
    mlflow.log_params(model.named_steps["regressor"].get_params())

    mlflow.log_params({
        "split": "chronological",
        "validation_start": "2012-07-01",
        "test_start": "2012-10-01",
    })

    # Fit the ENTIRE pipeline using training data only.
    model.fit(X_train, y_train)

    # Evaluate on exactly the same validation rows as the baseline.
    predictions = model.predict(X_val)
    model_metrics = regression_metrics(y_val, predictions)

    mlflow.log_metrics({
        "validation_mae": model_metrics["MAE"],
        "validation_rmse": model_metrics["RMSE"],
        "validation_r2": model_metrics["R2"],
    })

    # Save preprocessing AND the fitted forest together.
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="rental_pipeline",
        input_example=X_train.head(3),

        # Explicitly use skops to save the fitted pipeline.
        serialization_format="skops",

        # Allow the forest's internal tree type.
        # This is appropriate for the model we just trained ourselves.
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("Run ID:", run.info.run_id)
    print("Model URI:", model_info.model_uri)

# pd is already imported in your ingestion script.
comparison = pd.DataFrame({
    "baseline": baseline_metrics,
    "random_forest": model_metrics,
})

print(comparison)

# Lower MAE and RMSE are better.
# An improvement must be measured; do not assume this model wins.
# Keep X_test and y_test untouched.



# Require the trained model to beat the baseline on both error metrics.
# This is our initial model-selection gate, not full production acceptance.
passes_validation = (
    model_metrics["MAE"] < baseline_metrics["MAE"]
    and model_metrics["RMSE"] < baseline_metrics["RMSE"]
)

if not passes_validation:
    raise ValueError("Model failed validation: it must beat the baseline.")

# model_info came from the random forest's log_model(...) call.
# Register that EXACT saved pipeline rather than training another model.
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name="BikeDemand",
)

# Each registration creates a numbered version within the BikeDemand family.
print("Registered model:", registered_model.name)
print("Registered version:", registered_model.version)

# An alias is a readable pointer to a particular version.
# "candidate" means awaiting further checks; it does not mean deployed.
client = MlflowClient()

client.set_registered_model_alias(
    name="BikeDemand",
    alias="candidate",
    version=registered_model.version,
)

print("Candidate URI: models:/BikeDemand@candidate")

# Keep the test set untouched for now.
# Next, we will reload this exact version, verify predictions match,
# and perform the final test evaluation.



print("Loading the candidate model for verification...")
# Pin a numbered version for reproducibility.
# The "candidate" alias may point to a different version later.
candidate_version = str(registered_model.version)
candidate_uri = f"models:/BikeDemand/{candidate_version}"

loaded_model = mlflow.sklearn.load_model(candidate_uri)

# Check that saving and loading preserved the pipeline's predictions.
sample = X_val.head(20)

original_predictions = model.predict(sample)
reloaded_predictions = loaded_model.predict(sample)

# Allow tiny floating-point differences.
# A failed assertion stops execution before final evaluation.
np.testing.assert_allclose(
    original_predictions,
    reloaded_predictions,
    rtol=1e-10,
    atol=1e-10,
)

print("Save/load verification passed.")
print("Verified model:", candidate_uri)

# It is now safe to evaluate the candidate model on the test set.
# Final evaluation: the model has never fitted these observations.
# Do not use these results to repeatedly tune the model.
test_predictions = loaded_model.predict(X_test)
test_metrics = regression_metrics(y_test, test_predictions)

print("Final test metrics:", test_metrics)

# Find the training run associated with this exact registered version.
version_details = client.get_model_version(
    name="BikeDemand",
    version=candidate_version,
)

# Reopen that run and attach clearly labeled TEST metrics.
with mlflow.start_run(run_id=version_details.run_id):
    mlflow.log_metrics({
        "test_mae": test_metrics["MAE"],
        "test_rmse": test_metrics["RMSE"],
        "test_r2": test_metrics["R2"],
    })

    mlflow.set_tags({
        "save_load_check": "passed",
        "final_test_evaluated": "true",
        "evaluated_model_version": candidate_version,
    })

# Registration and successful evaluation do not deploy the model.
# Next, we will separate training from serving and build FastAPI.