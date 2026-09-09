# M4 Daily Forecasting with Conformal Prediction

TiDE-based daily demand forecasting for the M4 competition, with split conformal
prediction for calibrated uncertainty intervals and a newsvendor cost-optimal
ordering policy. Served via FastAPI, tracked and versioned with MLflow.

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API](#api)
- [Testing & CI](#testing--ci)
- [Docker](#docker)
- [Design Decisions](#design-decisions)

## Overview

This project benchmarks nine forecasting approaches on the M4 Daily dataset
(4,211 series after filtering) and productionizes the best-performing model
behind a REST API with calibrated prediction intervals.

Two questions drove the comparison:

1. How do zero-shot time series foundation models (Chronos, TimesFM) compare
   to models trained from scratch on M4 data (gradient boosting, TiDE/NHiTS/
   DLinear, fine-tuned TTM)?
2. Can conformal prediction turn point forecasts into calibrated, cost-aware
   inventory decisions under an asymmetric underage/overage cost structure
   (the newsvendor problem)?

## Results

All models are evaluated on an identical held-out set of 1,000 series, using
a 90-day context window and a 14-day forecast horizon. Calibration (500
series) and test (1,000 series) splits are strictly excluded from the
training pool of any model that requires fitting, to prevent leakage into
the conformal calibration step.

| Rank | Model | sMAPE | MASE |
|---|---|---|---|
| 1 | TiDE | 2.088 | 2.176 |
| 2 | TimesFM-3 (zero-shot) | 2.280 | 2.460 |
| 3 | Chronos-Bolt-Tiny (zero-shot) | 2.286 | 2.465 |
| 4 | TTM (fine-tuned) | 2.312 | 2.533 |
| 5 | DLinear | 2.325 | 2.581 |
| 6 | LightGBM | 2.340 | 2.621 |
| 7 | XGBoost | 2.346 | 2.623 |
| 8 | RandomForest | 2.371 | 2.688 |
| 9 | NHiTS | 2.456 | 2.548 |

Notably, two zero-shot foundation models outperform every model trained
specifically on M4 data except TiDE, with no task-specific fine-tuning.

**TiDE was selected for production** based on:
- Best accuracy (sMAPE, MASE) and lowest newsvendor cost, naive and
  conformal-adjusted, across all nine candidates.
- No licensing restrictions — TimesFM-3, the closest competitor, is released
  under a non-commercial license, which rules it out for production use.
- A genuine need for the training/registry/retraining infrastructure this
  project demonstrates, unlike the zero-shot models, which require no
  training pipeline at all.

## Repository Structure
```
.
├── src/forecasting/ # core library: config, data, models, conformal, evaluation, pipeline
├── scripts/ # train.py / calibrate.py / evaluate.py — MLflow-tracked CLI entrypoints
├── api/ # FastAPI service (main, predictor, schemas)
├── tests/ # pytest unit tests
├── notebooks/ # exploratory research: nine-model benchmark + conformal prediction study
├── Dockerfile, docker-compose.yml
├── pyproject.toml
├── Makefile
└── .github/workflows/ci.yml

```

## Methodology

### Data & splits
- Dataset: M4 Daily (4,227 series; 4,211 after filtering series shorter than
  106 observations).
- Canonical split, identical across every model: the first 500 valid series
  form the calibration set, the next 1,000 form the test set.
- Context window: 90 days. Forecast horizon: 14 days.
- Any trained (non-zero-shot) model is fit on a pool that explicitly
  excludes calibration and test series.

### Conformal prediction
Split conformal prediction with the finite-sample correction
`ceil((n+1)·τ) / n` (Romano et al.) is applied per forecast horizon. Two
quantiles are calibrated on the 500 held-out series:
- An absolute-residual quantile at 90% nominal coverage, used to report
  empirical interval coverage.
- A signed-residual quantile at the newsvendor critical ratio
  `cu / (cu + co)`, added to the point forecast to obtain a cost-optimal
  order quantity.

### Newsvendor cost
Forecast errors are evaluated under an asymmetric cost structure
(`cost_underage=2`, `cost_overage=1`), reflecting a scenario where
stockouts are twice as costly as excess inventory. The conformal-adjusted
forecast reduces average cost by 8-12% relative to the raw point forecast
across all nine models.

## Getting Started

```bash
conda create -n m4-forecasting python=3.11
conda activate m4-forecasting
pip install -e ".[dev]"
```

Place the M4 Daily CSVs at `data/M4/Daily-train.csv` and
`data/M4/Daily-test.csv`.

## Usage

```bash
python scripts/train.py       # trains TiDE on the leakage-free pool, logs an MLflow run
python scripts/calibrate.py   # calibrates conformal quantiles, registers a model+calibration
                               # bundle, promotes it to the "champion" alias
python scripts/evaluate.py    # evaluates the champion bundle on the canonical test split
```

Inspect experiments:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Serve the API locally:
```bash
uvicorn api.main:app --reload
```

## API

**GET /health**
```json
{"status": "ok", "model_version": "3"}
```

**POST /forecast**
```json
{
  "series_id": "D1",
  "history": [1023.4, 1025.1, "... at least 90 most-recent observations, oldest last"]
}
```
Response:
```json
{
  "series_id": "D1",
  "point_forecast": [14 values],
  "conformal_forecast": [14 values],
  "horizon": 14,
  "model_version": "3"
}
```
`conformal_forecast` is the point forecast shifted by the calibrated
newsvendor quantile — the recommended order quantity, not a raw prediction.

## Testing & CI

```bash
pytest
ruff check .
```
GitHub Actions (`.github/workflows/ci.yml`) runs both on every push and pull
request to `main`.

## Docker

```bash
docker compose up --build
```
The container serves the API only. Training and calibration run outside the
container against a local (or remote) MLflow tracking server; the resulting
`mlruns/` directory is mounted read-only.

## Design Decisions

- **Two-stage train/calibrate pipeline, not one.** Retraining TiDE is
  expensive and infrequent; conformal recalibration is cheap and can run
  far more often (e.g. nightly) to track distribution shift without a full
  retrain.
- **Model and calibration are registered as a single MLflow bundle**
  (`ForecasterBundle`), not two separate artifacts. This guarantees a
  served model version and its conformal quantiles can never drift out of
  sync.
- **The `champion` alias is used instead of MLflow's legacy stage system**
  (Staging/Production, now deprecated). The API always resolves
  `models:/<name>@champion`, so promoting a new version requires no code
  change.