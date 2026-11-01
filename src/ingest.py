from pathlib import Path

import pandas as pd

from validation import validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    input_path = PROJECT_ROOT / "data" / "raw" / "hour.csv"
    output_path = PROJECT_ROOT / "data" / "interim" / "validated.csv"

    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    # A failed check raises an exception and stops the stage.
    # DVC will report failure instead of running downstream stages.
    validate_data(df)

    # Publish the stage output only after validation succeeds.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Validated rows: {len(df)}")
    print(f"Output: {output_path}")


# Importing this module does not execute ingestion.
# Running "python src/ingest.py" executes main().
if __name__ == "__main__":
    main()