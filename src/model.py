from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model():
    # These numbers represent categories rather than continuous measurements.
    categorical_columns = [
        "hr", "mnth", "weekday",
        "workingday", "holiday", "weathersit",
    ]

    numeric_columns = ["temp", "hum", "windspeed"]

    # Apply different transformations to different groups of columns.
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    # Create separate indicator columns for categories.
                    # Unseen categories become all-zero indicators.
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            (
                "numeric",
                # Tree models do not require feature scaling.
                "passthrough",
                numeric_columns,
            ),
        ],
        remainder="drop",
    )

    # A random forest averages predictions from multiple decision trees.
    regressor = RandomForestRegressor(
        n_estimators=200,     # Number of trees.
        max_depth=16,         # Limit each tree's depth.
        min_samples_leaf=2,   # Require at least two training rows per leaf.
        random_state=42,      # Make repeated training reproducible.
        n_jobs=-1,            # Use available CPU cores for offline training.
    )

    # fit(): learn preprocessing, transform inputs, then train the forest.
    # predict(): reuse fitted preprocessing, then generate predictions.
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", regressor),
    ])