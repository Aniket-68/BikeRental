import pandas as pd



def temporal_split(X, y, timestamps):
    # Train: learn patterns from older observations.
    # Validation: compare models and choose settings.
    # Test: evaluate the final chosen model once.
    #
    # Shuffling could let later observations influence a model evaluated
    # on earlier observations, which would misrepresent future performance.

    if not X.index.equals(y.index) or not X.index.equals(timestamps.index):
        raise ValueError("Features, targets, and timestamps must have matching indexes.")

    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be present and sorted.")

    validation_start = pd.Timestamp("2012-07-01")
    test_start = pd.Timestamp("2012-10-01")

    # Each condition produces a Boolean mask: one True/False per observation.
    masks = {
        "train": timestamps < validation_start,

        "validation": (
            (timestamps >= validation_start)
            & (timestamps < test_start)
        ),

        "test": timestamps >= test_start,
    }

    splits = {}

    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"The {name} split is empty.")

        # Apply the SAME mask to X and y to preserve their correspondence.
        # .loc selects rows where the mask is True.
        splits[name] = (
            X.loc[mask].copy(),
            y.loc[mask].copy(),
        )

    return splits