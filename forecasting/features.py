"""Leakage-aware lag features shared by training and online inference."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

FEATURES = [
    "store_id",
    "product_id",
    "price",
    "promotion",
    "day_of_week",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "mean_7",
    "mean_28",
]
LAGS = (1, 7, 14, 28)


def _calendar(target_date: pd.Timestamp) -> dict[str, float]:
    angle = 2 * np.pi * target_date.dayofyear / 365.25
    return {
        "day_of_week": float(target_date.dayofweek),
        "day_of_year_sin": float(np.sin(angle)),
        "day_of_year_cos": float(np.cos(angle)),
    }


def training_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "store_id", "product_id", "price", "promotion", "sales"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    grouped = frame.groupby(["store_id", "product_id"], sort=False)["sales"]
    for lag in LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    frame["mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    frame["mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    angle = 2 * np.pi * frame["date"].dt.dayofyear / 365.25
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype(float)
    frame["day_of_year_sin"] = np.sin(angle)
    frame["day_of_year_cos"] = np.cos(angle)
    return frame.dropna(subset=FEATURES)


def prediction_row(
    *,
    store_id: int,
    product_id: int,
    target_date: date,
    price: float,
    promotion: bool,
    history: Sequence[float],
) -> pd.DataFrame:
    if len(history) < 28:
        raise ValueError("history must contain at least 28 daily sales values")
    sales = np.asarray(history, dtype=float)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("history must contain finite, non-negative sales values")
    if not np.isfinite(price) or price <= 0:
        raise ValueError("price must be a finite value greater than zero")

    row = {
        "store_id": float(store_id),
        "product_id": float(product_id),
        "price": float(price),
        "promotion": float(promotion),
        **_calendar(pd.Timestamp(target_date)),
        "lag_1": float(sales[-1]),
        "lag_7": float(sales[-7]),
        "lag_14": float(sales[-14]),
        "lag_28": float(sales[-28]),
        "mean_7": float(np.mean(sales[-7:])),
        "mean_28": float(np.mean(sales[-28:])),
    }
    return pd.DataFrame([row], columns=FEATURES)
