"""Train, compare and persist a versioned forecasting artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecasting.data import make_demo_data
from forecasting.features import FEATURES, training_frame


def metric_report(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denominator = max(float(np.abs(actual).sum()), 1.0)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "wape": round(float(np.abs(actual - predicted).sum() / denominator), 4),
    }


def train(output_dir: Path, data_path: Path | None = None) -> dict[str, object]:
    raw = pd.read_csv(data_path, parse_dates=["date"]) if data_path else make_demo_data()
    frame = training_frame(raw)
    dates = frame["date"].sort_values().unique()
    if len(dates) < 120:
        raise ValueError("Need at least 120 distinct dates for a 56-day temporal holdout")
    holdout_start = pd.Timestamp(dates[-56])
    train_rows = frame[frame["date"] < holdout_start]
    valid_rows = frame[frame["date"] >= holdout_start]

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_rows[FEATURES], train_rows["sales"])
    actual = valid_rows["sales"].to_numpy(dtype=float)
    model_predictions = np.maximum(model.predict(valid_rows[FEATURES]), 0)
    baseline_predictions = valid_rows["lag_7"].to_numpy(dtype=float)

    model_metrics = metric_report(actual, model_predictions)
    baseline_metrics = metric_report(actual, baseline_predictions)
    selected_name = (
        "hist_gradient_boosting"
        if model_metrics["mae"] < baseline_metrics["mae"]
        else "weekly_naive"
    )
    if selected_name == "hist_gradient_boosting":
        selected_estimator = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42,
        ).fit(frame[FEATURES], frame["sales"])
    else:
        selected_estimator = None
    selected_metrics = model_metrics if selected_estimator is not None else baseline_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    artifact = {
        "estimator": selected_estimator,
        "selected_model": selected_name,
        "feature_names": FEATURES,
        "trained_at": trained_at,
        "holdout_start": holdout_start.date().isoformat(),
        "holdout_end": valid_rows["date"].max().date().isoformat(),
        "metrics": selected_metrics,
        "comparison": {"weekly_naive": baseline_metrics, "hist_gradient_boosting": model_metrics},
    }
    joblib.dump(artifact, output_dir / "model.joblib")
    report = {key: value for key, value in artifact.items() if key != "estimator"}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--data",
        type=Path,
        help="Optional CSV with date/store/product/price/promotion/sales columns",
    )
    args = parser.parse_args()
    print(json.dumps(train(args.output_dir, args.data), indent=2))


if __name__ == "__main__":
    main()
