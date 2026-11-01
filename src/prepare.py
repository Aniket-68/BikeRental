from pathlib import Path

import pandas as pd

from features import prepare_features
from split_data import temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Importing this module does not execute preprocessing.
# Running "python src/preprocess.py" executes main().
# it is a good practice to have a main() function in each stage of the pipeline, so that it can be executed independently. 
def main():
    # Consume the previous stage's validated output.
    input_path = PROJECT_ROOT / "data" / "interim" / "validated.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"

    df = pd.read_csv(input_path)

    # Reuse the feature selection and timestamp logic you already wrote.
    X, y, timestamps = prepare_features(df)

    # Reuse the chronological boundaries:
    # train      -> before July 2012
    # validation -> July through September 2012
    # test       -> October 2012 onward
    splits = temporal_split(X, y, timestamps)

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (X_split, y_split) in splits.items():
        split_frame = X_split.copy()

        # Pandas aligns these assignments by index.
        # Features, target, and timestamp must describe the same observations.
        split_frame["cnt"] = y_split
        split_frame["timestamp"] = timestamps.loc[X_split.index]

        output_path = output_dir / f"{name}.csv"
        split_frame.to_csv(output_path, index=False)

        print(f"{name}: {len(split_frame)} rows")
        print(
            "Period:",
            split_frame["timestamp"].min(),
            "to",
            split_frame["timestamp"].max(),
        )

    # We have prepared test data, but have not evaluated any model on it.
    # cnt and timestamp must be excluded from model inputs during training.


if __name__ == "__main__":
    main()