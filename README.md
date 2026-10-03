# Retail Demand Forecasting

A reproducible time-series forecasting pipeline with a FastAPI inference service. It compares a weekly seasonal-naive baseline with a gradient-boosted model and only selects the ML model when it improves the holdout MAE.

## What it demonstrates

- Time-aware holdout evaluation rather than a random row split.
- Lag and rolling-window features built only from observations before the target date.
- Baseline comparison using MAE, RMSE and WAPE.
- A deterministic synthetic retail dataset, so the demo has no external data or credentials.
- A saved model artifact, model metadata, input validation and live/readiness endpoints.
- Prometheus request, latency and successful-forecast metrics for the companion deployment project.
- Optional user-provided CSV data using the same explicit schema.

The generated data is synthetic and is intended to demonstrate the engineering workflow. It is not evidence of real-world retail accuracy. The validation setup is a rolling one-step-ahead simulation: each validation prediction may use observed sales from earlier validation dates as its lag history.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m forecasting.train --output-dir artifacts
uvicorn forecasting.api:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The first training run generates the demo dataset in memory and saves `artifacts/model.joblib` and `artifacts/metrics.json`.

The GitHub Actions container workflow trains the model during the image build, then publishes versioned images to `ghcr.io/trima03/demand-forecasting-ml` on pushes to `main` and `v*` tags.

## API example

`POST /v1/forecast` expects the last 28 observed daily sales values, oldest first:

```json
{
  "store_id": 2,
  "product_id": 5,
  "target_date": "2025-01-01",
  "price": 12.5,
  "promotion": false,
  "history": [11, 12, 10, 13, 12, 15, 16, 12, 11, 14, 13, 12, 15, 16, 11, 13, 14, 15, 12, 12, 13, 17, 16, 15, 13, 14, 15, 16]
}
```

Other endpoints: `GET /health/live`, `GET /health/ready`, and `GET /v1/model`.

## Use your own CSV

Required columns: `date,store_id,product_id,price,promotion,sales`. Each store/product series needs at least 120 days; `promotion` is 0 or 1. The final 56 unique dates are reserved as a temporal holdout.

```bash
python -m forecasting.train --data path/to/daily_sales.csv --output-dir artifacts
```

## Project layout

```text
forecasting/
  api.py       FastAPI inference and health endpoints
  data.py      deterministic local demo data
  features.py  shared train/serve feature definitions
  train.py     temporal evaluation, model selection and artifact output
```

## Limitations

This first version is a single-series-at-a-time inference service and has no online retraining, authentication, inventory constraints or external feature store. Production forecasting needs business-specific backtesting, data-quality checks and an agreed retraining policy.
