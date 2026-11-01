# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BikeDemand: a staged, learning-oriented build of an end-to-end bike rental demand prediction service (see `README.md` for the 10-stage plan: NumPy → pandas → structured Python → sklearn model → Pipeline → FastAPI serving → async/persistence → pytest → MLflow → Docker/CI). The target is `cnt` (rentals per hour); the planned model inputs are `hr, mnth, weekday, workingday, weathersit, temp, hum` (`explore.py` also tries `holiday` and `windspeed`).

**Current state:** stages 1–3 are partly done: exploratory pandas scripts (`explore.py`), an ingest script (`src/ingest.py`), a data validator (`src/validation.py`), and a feature-preparation scaffold (`src/features.py`, TODOs unfinished). No model, API, tests, linting, or packaging yet. Update this file as those land.

## Commands

```
pip install -r requirements.txt   # pandas, numpy, fastapi, uvicorn (scikit-learn, pytest etc. not yet listed)
python explore.py                 # run from repo root (uses relative path data/raw/hour.csv)
python src/ingest.py              # path-independent; loads hour.csv, prints a summary, then runs validate_data
```

There is no test suite yet.

## Layout and conventions

- `explore.py` (repo root) is scratch exploration with a relative data path, so it only works from the repo root. `src/ingest.py` is the start of the "reusable structure" stage and resolves `PROJECT_ROOT` from `__file__` — follow that pattern for new modules. Exploratory code is meant to be refactored from `explore.py` into `src/`.
- `src/validation.py` provides `validate_data(df)`, which raises `ValueError` (never `assert`) for: empty frame, missing required columns (`dteday, hr, holiday, weathersit, temp, hum, windspeed, cnt`), NaNs in those columns, `hr` outside 0–23, negative `cnt`, and duplicate `(dteday, hr)` rows. It returns `None`. Note it does not yet check `mnth`, `weekday`, `workingday`, or value ranges for `temp`/`hum`/`windspeed`/`weathersit`.
- `src/features.py` has `prepare_features(df)`, which builds a `timestamp` from `dteday` + `hr` and sorts chronologically (to support a time-ordered split). It is **unfinished**: the `X`/`y` selection and the return statement are still TODO comments, so it currently returns `None`. Planned `feature_columns`: `hr, mnth, weekday, workingday, holiday, weathersit, temp, hum, windspeed`.
- `src/ingest.py` imports `from validation import validate_data` (a bare import), so it only works when run as a script (`python src/ingest.py`), not as `python -m src.ingest` or from another package. `src/__init__.py` exists but is empty.
- `src/__pycache__/validation.cpython-312.pyc` was committed by mistake; `.gitignore` does not cover `__pycache__`.
- `explore.py` has a stray `from pygit2 import features` import (unused, not in requirements.txt); it will fail if pygit2 isn't installed.
- Raw data lives in `data/raw/` (`hour.csv`, `Readme.txt`). `data` is listed in `.gitignore`, so the dataset is **not in the remote repo** and must be obtained separately (UCI "Bike Sharing Dataset", Capital Bikeshare, Washington D.C., 2011–2012). Only `hour.csv` is present locally (no `day.csv`).

## Data gotchas

- `hour.csv`: 17,379 hourly rows, no missing values. `dteday` loads as a string — parse it with `pd.to_datetime` for time work.
- `cnt = casual + registered`; never use `casual`/`registered` as features for `cnt` (leakage).
- `temp`, `atemp`, `hum`, `windspeed` are normalized (divided by 41, 50, 100, 67) — multiply back for real units.
- `yr`: 0 = 2011, 1 = 2012. `season`: 1 spring, 2 summer, 3 fall, 4 winter. `weathersit` 1 (clear) to 4 (heavy rain/snow).
- Anomaly/event example: Hurricane Sandy, 2012-10-30.
- Publications using the dataset must cite Fanaee-T & Gama (2013), "Event labeling combining ensemble detectors and background knowledge" (BibTeX in `data/raw/Readme.txt`).
