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


@dataclass(frozen=True)
class GroupWiseTargetNormalizer:
    group_min: dict[int, float]
    group_range: dict[int, float]
    fallback_min: float
    fallback_range: float

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        target_column: str,
        group_column: str = "Experiment_ID",
    ) -> "GroupWiseTargetNormalizer":
        missing_columns = {
            target_column,
            group_column,
        }.difference(frame.columns)

        if missing_columns:
            raise KeyError(
                "Cannot fit group-wise target normalizer. "
                f"Missing columns: {sorted(missing_columns)}"
            )

        target_values = pd.to_numeric(
            frame[target_column],
            errors="raise",
        ).astype(float)

        fallback_min = float(
            target_values.min()
        )
        fallback_range = cls._safe_range(
            minimum=fallback_min,
            maximum=float(target_values.max()),
        )

        group_stats = (
            frame.assign(
                _target_value=target_values
            )
            .groupby(group_column)["_target_value"]
            .agg(["min", "max"])
        )

        group_min = {
            int(group_id): float(row["min"])
            for group_id, row in group_stats.iterrows()
        }
        group_range = {
            int(group_id): cls._safe_range(
                minimum=float(row["min"]),
                maximum=float(row["max"]),
            )
            for group_id, row in group_stats.iterrows()
        }

        return cls(
            group_min=group_min,
            group_range=group_range,
            fallback_min=fallback_min,
            fallback_range=fallback_range,
        )

    def transform_frame(
        self,
        frame: pd.DataFrame,
        target_column: str,
        group_column: str = "Experiment_ID",
    ) -> np.ndarray:
        target_values = pd.to_numeric(
            frame[target_column],
            errors="raise",
        ).astype(float).to_numpy()
        minimums, ranges = self._scales_for_groups(
            frame[group_column]
        )

        return (
            target_values
            - minimums
        ) / ranges

    def inverse_transform_values(
        self,
        values: np.ndarray,
        group_ids: pd.Series | np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)
        minimums, ranges = self._scales_for_groups(
            group_ids
        )

        return (
            values
            * ranges
        ) + minimums

    def _scales_for_groups(
        self,
        group_ids: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        group_ids = np.asarray(
            group_ids
        ).reshape(-1)

        minimums = np.asarray(
            [
                self.group_min.get(
                    int(group_id),
                    self.fallback_min,
                )
                for group_id in group_ids
            ],
            dtype=float,
        )
        ranges = np.asarray(
            [
                self.group_range.get(
                    int(group_id),
                    self.fallback_range,
                )
                for group_id in group_ids
            ],
            dtype=float,
        )

        return minimums, ranges

    @staticmethod
    def _safe_range(
        minimum: float,
        maximum: float,
    ) -> float:
        value_range = float(
            maximum
            - minimum
        )

        if np.isclose(
            value_range,
            0.0,
        ):
            return 1.0

        return value_range


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

    if "Experiment_ID" not in train_df.columns:
        raise KeyError(
            "Training data is missing required column: Experiment_ID"
        )

    if "Experiment_ID" not in test_df.columns:
        raise KeyError(
            "Test data is missing required column: Experiment_ID"
        )

    X_train = train_df[feature_columns]
    target_normalizer = GroupWiseTargetNormalizer.fit(
        frame=train_df,
        target_column=target_column,
    )
    y_train = target_normalizer.transform_frame(
        frame=train_df,
        target_column=target_column,
    )
    X_test = test_df[feature_columns]

    model = build_qrf_model(
        n_estimators=n_estimators,
        model_config=model_config,
    )
    model.fit(X_train, y_train)
    model.group_wise_target_normalizer_ = (
        target_normalizer
    )
    model.target_column_ = target_column
    model.target_normalized_ = True

    lower_quantile = float(
        model_config.get("lower_quantile", 0.05)
    )
    upper_quantile = float(
        model_config.get("upper_quantile", 0.95)
    )

    lower_normalized = model.predict(
        X_test,
        quantiles=lower_quantile,
    )
    median_normalized = model.predict(
        X_test,
        quantiles=0.5,
    )
    upper_normalized = model.predict(
        X_test,
        quantiles=upper_quantile,
    )

    predictions = QRFPredictions(
        lower=target_normalizer.inverse_transform_values(
            lower_normalized,
            test_df["Experiment_ID"],
        ),
        median=target_normalizer.inverse_transform_values(
            median_normalized,
            test_df["Experiment_ID"],
        ),
        upper=target_normalizer.inverse_transform_values(
            upper_normalized,
            test_df["Experiment_ID"],
        ),
    )

    return model, predictions
