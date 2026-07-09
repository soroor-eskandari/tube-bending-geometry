import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.logging.log_utils import log_function


class QRFModelEvaluator:

    @staticmethod
    @log_function
    def evaluate(
        y_true: np.ndarray,
        y_pred_median: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray,
        target_name: str,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        angle_values: np.ndarray = None,
    ) -> dict:

        # ============================================================
        # GLOBAL METRICS
        # ============================================================

        r2 = r2_score(
            y_true,
            y_pred_median,
        )

        mse = mean_squared_error(
            y_true,
            y_pred_median,
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_true,
            y_pred_median,
        )

        inside_interval = (
            (y_true >= y_pred_lower)
            &
            (y_true <= y_pred_upper)
        )

        coverage = np.mean(
            inside_interval
        )

        interval_width = np.mean(
            y_pred_upper - y_pred_lower
        )

        expected_coverage = (
            upper_quantile
            - lower_quantile
        )

        calibration_error = abs(
            coverage - expected_coverage
        )

        # ============================================================
        # GLOBAL RESULT
        # ============================================================

        global_metrics = {

            "target":
                target_name,

            "r2":
                r2,

            "rmse":
                rmse,

            "mae":
                mae,

            "mse":
                mse,

            "coverage":
                coverage,

            "expected_coverage":
                expected_coverage,

            "calibration_error":
                calibration_error,

            "mean_interval_width":
                interval_width,

            "lower_quantile":
                lower_quantile,

            "upper_quantile":
                upper_quantile,

            # ====================================
            # GLOBAL BOUND STATISTICS
            # ====================================

            "mean_lower_bound":
                np.mean(y_pred_lower),

            "mean_upper_bound":
                np.mean(y_pred_upper),

            "min_lower_bound":
                np.min(y_pred_lower),

            "max_lower_bound":
                np.max(y_pred_lower),

            "min_upper_bound":
                np.min(y_pred_upper),

            "max_upper_bound":
                np.max(y_pred_upper),
        }

        # ============================================================
        # PER-ANGLE METRICS
        # ============================================================

        per_angle_df = pd.DataFrame()

        # ============================================================
        # RAW PREDICTIONS
        # ============================================================

        prediction_df = pd.DataFrame()

        if angle_values is not None:

            prediction_df = pd.DataFrame(
                {
                    "angle":
                        angle_values,

                    "y_true":
                        y_true,

                    "y_pred_median":
                        y_pred_median,

                    "y_pred_lower":
                        y_pred_lower,

                    "y_pred_upper":
                        y_pred_upper,
                }
            )

            per_angle_results = []

            for angle, group in (
                prediction_df.groupby("angle")
            ):

                group_inside_interval = (
                    (
                        group["y_true"]
                        >= group["y_pred_lower"]
                    )
                    &
                    (
                        group["y_true"]
                        <= group["y_pred_upper"]
                    )
                )

                group_mse = (
                    mean_squared_error(
                        group["y_true"],
                        group["y_pred_median"],
                    )
                )

                group_coverage = np.mean(
                    group_inside_interval
                )

                per_angle_results.append(
                    {

                        "target":
                            target_name,

                        "angle":
                            angle,

                        "r2":
                            r2_score(
                                group["y_true"],
                                group["y_pred_median"],
                            ),

                        "rmse":
                            np.sqrt(group_mse),

                        "mae":
                            mean_absolute_error(
                                group["y_true"],
                                group["y_pred_median"],
                            ),

                        "mse":
                            group_mse,

                        "coverage":
                            group_coverage,

                        "expected_coverage":
                            expected_coverage,

                        "calibration_error":
                            abs(
                                group_coverage
                                - expected_coverage
                            ),

                        "mean_interval_width":
                            np.mean(
                                group["y_pred_upper"]
                                - group["y_pred_lower"]
                            ),

                        # ====================================
                        # LOWER / UPPER BOUND STATISTICS
                        # ====================================

                        "mean_lower_bound":
                            np.mean(
                                group["y_pred_lower"]
                            ),

                        "mean_upper_bound":
                            np.mean(
                                group["y_pred_upper"]
                            ),

                        "min_lower_bound":
                            np.min(
                                group["y_pred_lower"]
                            ),

                        "max_lower_bound":
                            np.max(
                                group["y_pred_lower"]
                            ),

                        "min_upper_bound":
                            np.min(
                                group["y_pred_upper"]
                            ),

                        "max_upper_bound":
                            np.max(
                                group["y_pred_upper"]
                            ),
                    }
                )

            per_angle_df = pd.DataFrame(
                per_angle_results
            )

        return {

            "global_metrics":
                global_metrics,

            "per_angle_metrics":
                per_angle_df,

            # ====================================
            # RAW SAMPLE-LEVEL PREDICTIONS
            # ====================================

            "prediction_details":
                prediction_df,
        }
