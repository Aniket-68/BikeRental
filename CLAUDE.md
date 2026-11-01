# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This repo currently contains **only data** — no source code, build system, tests, or linting config exist yet. If code is added later, update this file with build/test commands and architecture.

## Data

`data/bike+sharing+dataset/` holds the UCI Bike Sharing Dataset (Capital Bikeshare, Washington D.C., 2011–2012), extracted from `data/bike+sharing+dataset.zip`. The extracted folder has its own nested `.git` directory (an artifact of extraction) — don't treat it as the project's repo.

- `hour.csv` — 17,379 hourly records; `day.csv` — 731 daily records. Same columns, except `hr` exists only in `hour.csv`.
- Intended tasks (per `Readme.txt`): regression on rental count (`cnt`) from weather/seasonal features, and event/anomaly detection (e.g. Hurricane Sandy on 2012-10-30).

Column gotchas:
- `cnt = casual + registered`; the latter two are leakage if predicting `cnt`.
- `temp`, `atemp`, `hum`, `windspeed` are **normalized** (divided by 41, 50, 100, 67 respectively) — multiply back for real units.
- `yr`: 0 = 2011, 1 = 2012. `season`: 1 spring, 2 summer, 3 fall, 4 winter. `weathersit` ranges 1 (clear) to 4 (heavy rain/snow).
- `instant` is just a row index; `dteday` is the date.

Publications using the dataset must cite Fanaee-T & Gama (2013), "Event labeling combining ensemble detectors and background knowledge" (see `Readme.txt` for BibTeX).
