import pandas as pd


def prepare_features(df: pd.DataFrame):
    # Copy before modifying so we preserve the caller's DataFrame.
    df = df.copy()

    # Combine the date and hour into one timestamp.
    # Example: 2012-01-01 + 8 hours = 2012-01-01 08:00:00
    df["timestamp"] = (
        pd.to_datetime(df["dteday"], format="%Y-%m-%d", errors="raise")
        + pd.to_timedelta(df["hr"], unit="h")
    )

    # Put observations in chronological order before any train/test split.
    # reset_index(drop=True) gives the sorted rows consecutive index labels.
    df = df.sort_values("timestamp").reset_index(drop=True)

    
    feature_columns = [
        "hr",
        "mnth",
        "weekday",
        "workingday",
        "holiday",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
    ]

    # These features describe the calendar and supplied weather conditions.
    # temp, hum, and windspeed are already normalized in the UCI dataset.
    #
    # EXCLUDE:
    # cnt        -> the answer we want to predict
    # casual     -> part of the answer
    # registered -> part of the answer
    # instant    -> record identifier
    #
    # casual + registered = cnt, so including them would leak the answer.

    # TODO 1: Create X by selecting feature_columns.
    # Hint: df[feature_columns].copy()
    X=df[feature_columns].copy()# store the features in a new DataFrame X

    # TODO 2: Create y by selecting "cnt".
    # Hint: df["cnt"].copy()
    y=df["cnt"].copy()# store the target variable in a new Series y
    # TODO 3: Return X, y, and the timestamp Series.
    return X, y, df["timestamp"]
    # Hint: return X, y, df["timestamp"]

    