import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.logging.log_utils import log_function


class SKModelEvaluator:

    @staticmethod
    @log_function
    def evaluate(
        prediction_df: pd.DataFrame,
        target_name: str,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> dict:

        predictions = prediction_df.copy()

        # =========================
        # VALIDATION
        # =========================
        required_columns = [
            "Angle[degree]",
            "y_true",
            "y_pred_mean",
            "y_pred_lower",
            "y_pred_upper",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in predictions.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns in prediction_df: "
                f"{missing_columns}"
            )

        # =========================
        # GLOBAL METRICS
        # =========================
        y_true = predictions["y_true"].to_numpy()
        y_pred_mean = predictions[
            "y_pred_mean"
        ].to_numpy()
        y_pred_lower = predictions[
            "y_pred_lower"
        ].to_numpy()
        y_pred_upper = predictions[
            "y_pred_upper"
        ].to_numpy()

        mse = mean_squared_error(
            y_true,
            y_pred_mean,
        )

        inside_interval = (
            (y_true >= y_pred_lower)
            & (y_true <= y_pred_upper)
        )

        coverage = np.mean(
            inside_interval
        )

        expected_coverage = (
            upper_quantile
            - lower_quantile
        )

        global_metrics = {
            "target": target_name,
            "r2": r2_score(
                y_true,
                y_pred_mean,
            ),
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(
                y_true,
                y_pred_mean,
            ),
            "mse": mse,
            "coverage": coverage,
            "expected_coverage": expected_coverage,
            "calibration_error": abs(
                coverage
                - expected_coverage
            ),
            "mean_interval_width": np.mean(
                y_pred_upper
                - y_pred_lower
            ),
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "mean_lower_bound": np.mean(
                y_pred_lower
            ),
            "mean_upper_bound": np.mean(
                y_pred_upper
            ),
            "min_lower_bound": np.min(
                y_pred_lower
            ),
            "max_lower_bound": np.max(
                y_pred_lower
            ),
            "min_upper_bound": np.min(
                y_pred_upper
            ),
            "max_upper_bound": np.max(
                y_pred_upper
            ),
        }

        # =========================
        # PER-ANGLE METRICS
        # =========================
        per_angle_results = []

        for angle, group in predictions.groupby(
            "Angle[degree]"
        ):
            group_y_true = group[
                "y_true"
            ].to_numpy()

            group_y_pred_mean = group[
                "y_pred_mean"
            ].to_numpy()

            group_y_pred_lower = group[
                "y_pred_lower"
            ].to_numpy()

            group_y_pred_upper = group[
                "y_pred_upper"
            ].to_numpy()

            group_mse = mean_squared_error(
                group_y_true,
                group_y_pred_mean,
            )

            group_inside_interval = (
                (group_y_true >= group_y_pred_lower)
                & (group_y_true <= group_y_pred_upper)
            )

            group_coverage = np.mean(
                group_inside_interval
            )

            if len(group_y_true) > 1:
                group_r2 = r2_score(
                    group_y_true,
                    group_y_pred_mean,
                )
            else:
                group_r2 = np.nan

            per_angle_results.append(
                {
                    "target": target_name,
                    "angle": angle,
                    "r2": group_r2,
                    "rmse": np.sqrt(group_mse),
                    "mae": mean_absolute_error(
                        group_y_true,
                        group_y_pred_mean,
                    ),
                    "mse": group_mse,
                    "coverage": group_coverage,
                    "expected_coverage": expected_coverage,
                    "calibration_error": abs(
                        group_coverage
                        - expected_coverage
                    ),
                    "mean_interval_width": np.mean(
                        group_y_pred_upper
                        - group_y_pred_lower
                    ),
                    "mean_lower_bound": np.mean(
                        group_y_pred_lower
                    ),
                    "mean_upper_bound": np.mean(
                        group_y_pred_upper
                    ),
                    "min_lower_bound": np.min(
                        group_y_pred_lower
                    ),
                    "max_lower_bound": np.max(
                        group_y_pred_lower
                    ),
                    "min_upper_bound": np.min(
                        group_y_pred_upper
                    ),
                    "max_upper_bound": np.max(
                        group_y_pred_upper
                    ),
                }
            )

        per_angle_df = pd.DataFrame(
            per_angle_results
        )

        # =========================
        # FINAL OUTPUT
        # =========================
        return {
            "global_metrics": global_metrics,
            "per_angle_metrics": per_angle_df,
            "prediction_details": predictions,
        }
