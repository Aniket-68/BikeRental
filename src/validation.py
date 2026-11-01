import pandas as pd

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the input DataFrame for missing values and data types.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate.

    Raises:
    ValueError: If there are missing values or incorrect data types.
    """

    if df.empty:
        raise ValueError("The DataFrame is empty.")

    required_columns = [
        "dteday", "hr", "holiday", "weathersit",
        "temp", "hum", "windspeed", "cnt",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} is missing from the DataFrame.")

    
    # A set difference finds required names absent from the actual columns.
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # TODO 1: Reject missing values in the required columns.
    df_required = df[required_columns]
    # Hint: df[required_columns].isna().any().any()
    # First .any() checks each column; second checks across those results.
    if df_required.isna().any().any():
        raise ValueError("DataFrame contains missing values in required columns.")

    # TODO 2: Reject hours outside 0–23.
    df_hours = df["hr"]
    if not df_hours.between(0, 23).all():
        raise ValueError("DataFrame contains hours outside the range 0–23.")
    
    # TODO 3: Reject negative rental counts.
    # Hint: (df["cnt"] >= 0).all()
    if not (df["cnt"] >= 0).all():
        raise ValueError("DataFrame contains negative rental counts.")
    # TODO 4: Reject duplicate observations for the same date and hour.
    # Hint: df.duplicated(subset=["dteday", "hr"]).any()
    duplicates = df.duplicated(subset=["dteday", "hr"])
    if duplicates.any():
        raise ValueError("DataFrame contains duplicate observations for the same date and hour.")

    # For each failed rule, raise ValueError with a useful message.
    # Use explicit exceptions for validation.
    # Python assert statements belong in our tests, not input validation.