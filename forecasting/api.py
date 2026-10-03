"""Small inference API with explicit model readiness and input validation."""

from __future__ import annotations

import os
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from forecasting.features import prediction_row

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="Demand Forecast API", version="0.1.0")
forecast_predictions = Counter("forecast_predictions_total", "Successful demand predictions", ["model"])
http_requests = Counter(
    "http_requests_total", "Completed HTTP requests", ["method", "handler", "status"]
)
http_duration = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "handler"]
)
app.mount("/metrics", make_asgi_app(), name="metrics")


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        route = request.scope.get("route")
        handler = getattr(route, "path", "unmatched")
        http_requests.labels(request.method, handler, "500").inc()
        http_duration.labels(request.method, handler).observe(time.perf_counter() - started)
        raise

    route = request.scope.get("route")
    handler = getattr(route, "path", "unmatched")
    http_requests.labels(request.method, handler, str(response.status_code)).inc()
    http_duration.labels(request.method, handler).observe(time.perf_counter() - started)
    return response


class ForecastRequest(BaseModel):
    store_id: int = Field(ge=1)
    product_id: int = Field(ge=1)
    target_date: date
    price: float = Field(gt=0, allow_inf_nan=False)
    promotion: bool = False
    history: list[float] = Field(min_length=28, max_length=365)


class ForecastResponse(BaseModel):
    store_id: int
    product_id: int
    target_date: date
    predicted_units: float
    model: str


@lru_cache(maxsize=1)
def _load_artifact() -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}; train the model first")
    return joblib.load(MODEL_PATH)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    try:
        artifact = _load_artifact()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model": str(artifact["selected_model"])}


@app.get("/v1/model")
def model_info() -> dict[str, object]:
    try:
        artifact = _load_artifact()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {key: artifact[key] for key in ("selected_model", "trained_at", "holdout_start", "holdout_end", "metrics")}


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        artifact = _load_artifact()
        row = prediction_row(**request.model_dump())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503 if isinstance(exc, FileNotFoundError) else 422, detail=str(exc)) from exc

    if artifact["selected_model"] == "weekly_naive":
        prediction = float(row.loc[0, "lag_7"])
    else:
        estimator = artifact["estimator"]
        prediction = float(estimator.predict(row[artifact["feature_names"]])[0])
    forecast_predictions.labels(model=str(artifact["selected_model"])).inc()
    return ForecastResponse(
        store_id=request.store_id,
        product_id=request.product_id,
        target_date=request.target_date,
        predicted_units=round(max(0.0, prediction), 2),
        model=str(artifact["selected_model"]),
    )
