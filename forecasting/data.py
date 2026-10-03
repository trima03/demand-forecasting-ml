"""Synthetic retail data used for a deterministic, zero-download demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    stores: int = 4,
    products: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Create daily store/product sales with trend, weekly/yearly effects and promotions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    records: list[dict[str, object]] = []

    for store_id in range(1, stores + 1):
        store_effect = rng.uniform(0.75, 1.45)
        for product_id in range(1, products + 1):
            base = rng.uniform(8, 46)
            price = rng.uniform(3, 28)
            product_phase = rng.uniform(0, 2 * np.pi)
            store_noise = rng.normal(0, 1.8, len(dates))
            for index, date in enumerate(dates):
                promotion = int(rng.random() < 0.08)
                annual = 1 + 0.22 * np.sin(2 * np.pi * date.dayofyear / 365.25 + product_phase)
                weekday = 1.22 if date.dayofweek in (4, 5) else 0.92 if date.dayofweek == 0 else 1.0
                trend = 1 + 0.00025 * index
                units = base * store_effect * annual * weekday * trend
                units *= 1.35 if promotion else 1.0
                units *= np.exp(-0.025 * (price - 12))
                sales = max(0, int(round(units + store_noise[index] + rng.normal(0, 2.5))))
                records.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": product_id,
                        "price": round(float(price), 2),
                        "promotion": promotion,
                        "sales": sales,
                    }
                )

    return pd.DataFrame.from_records(records)


def write_demo_data(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data().to_csv(path, index=False)
    return path
