from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor


@dataclass(frozen=True)
class QRFPredictions:
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray


def build_qrf_model(
    n_estimators: int,
    model_config: dict,
) -> RandomForestQuantileRegressor:
    """Build one Quantile Random Forest model."""
    return RandomForestQuantileRegressor(
        n_estimators=int(n_estimators),
        max_depth=model_config.get("max_depth"),
        min_samples_leaf=int(
            model_config.get("min_samples_leaf", 1)
        ),
        min_samples_split=int(
            model_config.get("min_samples_split", 2)
        ),
        max_features=model_config.get("max_features", 1.0),
        bootstrap=bool(model_config.get("bootstrap", True)),
        random_state=int(model_config.get("random_state", 1100)),
        n_jobs=int(model_config.get("n_jobs", -1)),
    )


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_estimators: int,
    model_config: dict,
) -> tuple[RandomForestQuantileRegressor, QRFPredictions]:
    """Train one model and return lower, median and upper predictions."""
    missing_columns = set(
        feature_columns + [target_column]
    ).difference(train_df.columns)

    if missing_columns:
        raise KeyError(
            f"Training data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]

    model = build_qrf_model(
        n_estimators=n_estimators,
        model_config=model_config,
    )
    model.fit(X_train, y_train)

    lower_quantile = float(
        model_config.get("lower_quantile", 0.05)
    )
    upper_quantile = float(
        model_config.get("upper_quantile", 0.95)
    )

    lower = model.predict(
        X_test,
        quantiles=lower_quantile,
    )
    median = model.predict(
        X_test,
        quantiles=0.5,
    )
    upper = model.predict(
        X_test,
        quantiles=upper_quantile,
    )

    predictions = QRFPredictions(
        lower=np.asarray(lower).reshape(-1),
        median=np.asarray(median).reshape(-1),
        upper=np.asarray(upper).reshape(-1),
    )

    return model, predictions
