import numpy as np
import pandas as pd
import pytest

from src.evaluation import regression_metrics
from src.features import prepare_features
from src.split_data import temporal_split
from src.validation import validate_data


@pytest.fixture
def raw_sample():
    # A fixture supplies fresh test data to each test requesting it.
    # These invented observations sit around our split boundaries.
    return pd.DataFrame({
        "dteday": [
            "2012-06-30",
            "2012-07-01",
            "2012-09-30",
            "2012-10-01",
        ],
        "hr": [23, 0, 23, 0],
        "mnth": [6, 7, 9, 10],
        "weekday": [6, 0, 0, 1],
        "workingday": [0, 0, 0, 1],
        "holiday": [0, 0, 0, 0],
        "weathersit": [1, 1, 1, 1],
        "temp": [0.5, 0.6, 0.4, 0.3],
        "hum": [0.6, 0.7, 0.5, 0.4],
        "windspeed": [0.1, 0.2, 0.1, 0.3],
        "cnt": [30, 50, 90, 70],
        "casual": [10, 20, 30, 40],
        "registered": [20, 30, 60, 30],
        "instant": [1, 2, 3, 4],
    })


def test_valid_data_is_accepted(raw_sample):
    # A valid dataset should pass without raising an exception.
    validate_data(raw_sample)


@pytest.mark.parametrize("column", ["mnth", "weekday", "workingday"])
def test_missing_model_input_is_rejected(raw_sample, column):
    # This decorator runs the same test separately for each column.
    incomplete = raw_sample.drop(columns=[column])

    # The test passes only if validation raises the expected exception.
    with pytest.raises(ValueError):
        validate_data(incomplete)


@pytest.mark.parametrize(
    "column,invalid_value",
    [
        ("hr", 25),
        ("cnt", -1),
        ("temp", np.nan),
    ],
)
def test_invalid_values_are_rejected(raw_sample, column, invalid_value):
    raw_sample.loc[0, column] = invalid_value

    with pytest.raises(ValueError):
        validate_data(raw_sample)


def test_duplicate_observations_are_rejected(raw_sample):
    duplicated = pd.concat(
        [raw_sample, raw_sample.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        validate_data(duplicated)


def test_features_preserve_alignment_and_exclude_leakage(raw_sample):
    shuffled = raw_sample.iloc[[2, 0, 3, 1]].copy()
    before = shuffled.copy(deep=True)

    X, y, timestamps = prepare_features(shuffled)

    # Sorting must preserve which target belongs to each observation.
    assert y.tolist() == [30, 50, 90, 70]
    assert X["hr"].tolist() == [23, 0, 23, 0]
    assert timestamps.is_monotonic_increasing

    # Answers, answer components, and identifiers must not become inputs.
    excluded = {
        "cnt", "casual", "registered",
        "instant", "dteday", "timestamp",
    }
    assert excluded.isdisjoint(X.columns)

    # Feature preparation must not mutate the caller's DataFrame.
    pd.testing.assert_frame_equal(shuffled, before)


def test_temporal_split_boundaries(raw_sample):
    X, y, timestamps = prepare_features(raw_sample)
    splits = temporal_split(X, y, timestamps)

    # July 1 belongs to validation; October 1 belongs to test.
    assert splits["train"][1].tolist() == [30]
    assert splits["validation"][1].tolist() == [50, 90]
    assert splits["test"][1].tolist() == [70]

    # Features and targets must retain matching indexes.
    for X_split, y_split in splits.values():
        assert X_split.index.equals(y_split.index)


def test_metrics_against_hand_calculated_values():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 30.0])

    metrics = regression_metrics(actual, predicted)

    # Absolute errors: 2, 2, 0. Squared errors: 4, 4, 0.
    # approx allows normal floating-point rounding differences.
    assert metrics["MAE"] == pytest.approx(4 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(8 / 3))
    assert metrics["R2"] == pytest.approx(0.96)