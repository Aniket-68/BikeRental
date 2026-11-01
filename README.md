BikeDemand: an end-to-end bike rental prediction service.

The problem statement

A bike rental company wants to estimate how many rentals to expect during a particular hour, given the time, calendar, and weather conditions.

Build a system that:

1. Loads and checks historical rental data.
2. Explores demand patterns.
3. Trains and evaluates a machine learning model.
4. Serves predictions through FastAPI.
5. Records predictions and model versions.
6. Tests, packages, deploys, and monitors the service.
7. Supports replacing a model and rolling back a bad release.
   The first version will estimate demand for supplied conditions. Later, we can extend it to forecasting using weather forecasts.

| Column       | Meaning                        | Role                  |
| ------------ | ------------------------------ | --------------------- |
| `hr`         | Hour, 0–23                     | Input                 |
| `mnth`       | Month                          | Input                 |
| `weekday`    | Day of the week                | Input                 |
| `workingday` | Whether it is a working day    | Input                 |
| `weathersit` | Weather category               | Input                 |
| `temp`       | Normalized temperature         | Input                 |
| `hum`        | Normalized humidity            | Input                 |
| `cnt`        | Total rentals during that hour | **Target to predict** |

| Stage                     | What we build                                     | What you learn                                                      |
| ------------------------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| 1. NumPy foundations      | Calculate demand statistics and select busy hours | Arrays, shapes, indexing, masks, broadcasting, aggregation          |
| 2. pandas exploration     | Load, inspect, clean, and summarize `hour.csv`    | `read_csv`, `head`, `info`, `loc`, missing values, `groupby`, dates |
| 3. Python structure       | Turn exploratory code into reusable functions     | Functions, type hints, exceptions, modules, logging                 |
| 4. First ML model         | Train a baseline, then a regression model         | Features/target, `fit`, `predict`, MAE, RMSE                        |
| 5. Reliable training      | Bundle preprocessing and prediction               | `Pipeline`, `ColumnTransformer`, encoding, validation               |
| 6. Model serving          | Load the saved pipeline and expose endpoints      | FastAPI, Pydantic, HTTP, application lifespan                       |
| 7. Async and persistence  | Store prediction history and retrieve it          | `async def`, `await`, database I/O, concurrency                     |
| 8. Testing                | Check calculations, pipelines, and API behavior   | `assert`, pytest, fixtures, API tests                               |
| 9. MLOps                  | Track experiments and version model releases      | MLflow, dataset fingerprints, reproducibility, promotion, rollback  |
| 10. Deployment and design | Package, load-test, monitor, and scale            | Docker, CI/CD, workers, latency, throughput, failure handling       |

## Progress

| Stage | Status |
| ----- | ------ |
| 1–2. NumPy / pandas exploration | Done (`explore.py`) |
| 3. Python structure | In progress: `src/ingest.py` (loading), `src/validation.py` (`validate_data`) done; `src/features.py` (`prepare_features`) unfinished |
| 4–10 | Not started |

Run `python src/ingest.py` from anywhere to load `data/raw/hour.csv` and validate it. The dataset is git-ignored; download the UCI "Bike Sharing Dataset" into `data/raw/`.
