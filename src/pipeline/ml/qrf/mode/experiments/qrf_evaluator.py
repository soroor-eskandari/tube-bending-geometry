from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def coverage_percent(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)

    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(100.0 * inside.mean())


def evaluate_predictions(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_median: np.ndarray,
    y_upper: np.ndarray,
) -> dict[str, float]:
    """Calculate the standard metrics for one QRF run."""
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_median = np.asarray(y_median).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)

    return {
        "coverage_percent": coverage_percent(
            y_true=y_true,
            y_lower=y_lower,
            y_upper=y_upper,
        ),
        "rmse": float(
            mean_squared_error(
                y_true,
                y_median,
                squared=False,
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_median,
            )
        ),
        "mean_interval_width": float(
            np.mean(y_upper - y_lower)
        ),
    }
