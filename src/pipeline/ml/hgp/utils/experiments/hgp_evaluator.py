from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _as_1d(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def _validate_equal_lengths(
    **arrays: np.ndarray,
) -> None:
    lengths = {
        name: len(values)
        for name, values in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Prediction arrays have different lengths: "
            f"{lengths}"
        )


def coverage_percent(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    y_true = _as_1d(y_true, "y_true")
    y_lower = _as_1d(y_lower, "y_lower")
    y_upper = _as_1d(y_upper, "y_upper")

    _validate_equal_lengths(
        y_true=y_true,
        y_lower=y_lower,
        y_upper=y_upper,
    )

    inside = (
        (y_true >= y_lower)
        & (y_true <= y_upper)
    )

    return float(
        100.0 * inside.mean()
    )


def gaussian_negative_log_predictive_density(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    standard_deviation_floor: float = 1e-9,
) -> float:
    y_true = _as_1d(y_true, "y_true")
    y_mean = _as_1d(y_mean, "y_mean")
    y_std = _as_1d(y_std, "y_std")

    _validate_equal_lengths(
        y_true=y_true,
        y_mean=y_mean,
        y_std=y_std,
    )

    safe_std = np.maximum(
        y_std,
        standard_deviation_floor,
    )
    variance = np.square(safe_std)

    nll = (
        0.5 * np.log(
            2.0 * np.pi * variance
        )
        + 0.5
        * np.square(y_true - y_mean)
        / variance
    )

    return float(np.mean(nll))


def interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    confidence_level: float,
) -> float:
    """
    Mean Winkler interval score.

    Lower values are better. It rewards narrow intervals and penalizes
    observations that fall outside the interval.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    y_true = _as_1d(y_true, "y_true")
    y_lower = _as_1d(y_lower, "y_lower")
    y_upper = _as_1d(y_upper, "y_upper")

    _validate_equal_lengths(
        y_true=y_true,
        y_lower=y_lower,
        y_upper=y_upper,
    )

    alpha = 1.0 - confidence_level
    width = y_upper - y_lower

    below_penalty = (
        2.0 / alpha
    ) * (
        y_lower - y_true
    ) * (
        y_true < y_lower
    )

    above_penalty = (
        2.0 / alpha
    ) * (
        y_true - y_upper
    ) * (
        y_true > y_upper
    )

    return float(
        np.mean(
            width
            + below_penalty
            + above_penalty
        )
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_mean: np.ndarray,
    y_upper: np.ndarray,
    y_total_std: np.ndarray | None = None,
    confidence_level: float = 0.90,
) -> dict[str, float]:
    """Calculate point and uncertainty metrics for one HGP run."""
    y_true = _as_1d(y_true, "y_true")
    y_lower = _as_1d(y_lower, "y_lower")
    y_mean = _as_1d(y_mean, "y_mean")
    y_upper = _as_1d(y_upper, "y_upper")

    _validate_equal_lengths(
        y_true=y_true,
        y_lower=y_lower,
        y_mean=y_mean,
        y_upper=y_upper,
    )

    if np.any(y_upper < y_lower):
        raise ValueError(
            "At least one upper bound is below its lower bound."
        )

    metrics = {
        "coverage_percent": coverage_percent(
            y_true=y_true,
            y_lower=y_lower,
            y_upper=y_upper,
        ),
        "rmse": float(
            mean_squared_error(
                y_true,
                y_mean,
            ) ** 0.5
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_mean,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_mean,
            )
        ),
        "mean_interval_width": float(
            np.mean(
                y_upper - y_lower
            )
        ),
        "median_interval_width": float(
            np.median(
                y_upper - y_lower
            )
        ),
        "interval_score": interval_score(
            y_true=y_true,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=confidence_level,
        ),
        "coverage_error_percent": float(
            abs(
                coverage_percent(
                    y_true=y_true,
                    y_lower=y_lower,
                    y_upper=y_upper,
                )
                - confidence_level * 100.0
            )
        ),
    }

    if y_total_std is not None:
        metrics[
            "negative_log_predictive_density"
        ] = (
            gaussian_negative_log_predictive_density(
                y_true=y_true,
                y_mean=y_mean,
                y_std=y_total_std,
            )
        )

    return metrics
