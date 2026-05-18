from statistics import NormalDist

import numpy as np
import pandas as pd

from src.logging.log_utils import log_function


class SKPredictor:

    @staticmethod
    @log_function
    def predict(
        test_df: pd.DataFrame,
        sk_training_result: dict,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> dict:

        test_data = test_df.copy()

        # =========================
        # COLUMN NAMES
        # =========================
        angle_column = "Angle[degree]ORDistance[mm]"

        feature_columns = [
            "Group_ID",
            angle_column,
        ]

        target_main = "Main-axis [mm]"
        target_secondary = "Secondary-axis [mm]"

        required_columns = (
            [
                "Experiment_ID",
            ]
            + feature_columns
            + [
                target_secondary,
                target_main,
            ]
        )

        missing_columns = [
            col for col in required_columns
            if col not in test_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in test_df: {missing_columns}"
            )

        if not 0 < lower_quantile < upper_quantile < 1:
            raise ValueError(
                "Quantiles must satisfy 0 < lower < upper < 1."
            )

        # =========================
        # MODELS / FEATURES
        # =========================
        X_test = test_data[
            feature_columns
        ].to_numpy()

        main_model = sk_training_result[
            "main_model"
        ]

        secondary_model = sk_training_result[
            "secondary_model"
        ]

        main_noise_model = sk_training_result[
            "main_noise_model"
        ]

        secondary_noise_model = sk_training_result[
            "secondary_noise_model"
        ]

        z_value = NormalDist().inv_cdf(
            upper_quantile
        )

        # =========================
        # PREDICT MAIN
        # =========================
        main_mean, main_surface_std = (
            main_model.predict(
                X_test,
                return_std=True,
            )
        )

        main_log_intrinsic_variance = (
            main_noise_model.predict(
                X_test,
            )
        )

        main_intrinsic_variance = np.exp(
            main_log_intrinsic_variance
        )

        main_total_std = np.sqrt(
            main_surface_std ** 2
            + main_intrinsic_variance
        )

        main_lower = (
            main_mean
            - z_value * main_total_std
        )

        main_upper = (
            main_mean
            + z_value * main_total_std
        )

        # =========================
        # PREDICT SECONDARY
        # =========================
        secondary_mean, secondary_surface_std = (
            secondary_model.predict(
                X_test,
                return_std=True,
            )
        )

        secondary_log_intrinsic_variance = (
            secondary_noise_model.predict(
                X_test,
            )
        )

        secondary_intrinsic_variance = np.exp(
            secondary_log_intrinsic_variance
        )

        secondary_total_std = np.sqrt(
            secondary_surface_std ** 2
            + secondary_intrinsic_variance
        )

        secondary_lower = (
            secondary_mean
            - z_value * secondary_total_std
        )

        secondary_upper = (
            secondary_mean
            + z_value * secondary_total_std
        )

        # =========================
        # OUTPUT DATAFRAMES
        # =========================
        common_columns = {
            "Experiment_ID":
                test_data["Experiment_ID"].values,

            "Group_ID":
                test_data["Group_ID"].values,

            "Angle[degree]":
                test_data[angle_column].values,
        }

        main_prediction_df = pd.DataFrame(
            {
                **common_columns,
                "y_true": test_data[target_main].values,
                "y_pred_mean": main_mean,
                "y_pred_lower": main_lower,
                "y_pred_upper": main_upper,
                "surface_std": main_surface_std,
                "intrinsic_log_variance": (
                    main_log_intrinsic_variance
                ),
                "intrinsic_variance": main_intrinsic_variance,
                "total_std": main_total_std,
            }
        )

        secondary_prediction_df = pd.DataFrame(
            {
                **common_columns,
                "y_true": test_data[target_secondary].values,
                "y_pred_mean": secondary_mean,
                "y_pred_lower": secondary_lower,
                "y_pred_upper": secondary_upper,
                "surface_std": secondary_surface_std,
                "intrinsic_log_variance": (
                    secondary_log_intrinsic_variance
                ),
                "intrinsic_variance": (
                    secondary_intrinsic_variance
                ),
                "total_std": secondary_total_std,
            }
        )

        # =========================
        # FINAL OUTPUT
        # =========================
        return {
            "main": {
                "prediction_df": main_prediction_df,
                "y_true": test_data[target_main].values,
                "y_pred_mean": main_mean,
                "y_pred_lower": main_lower,
                "y_pred_upper": main_upper,
            },
            "secondary": {
                "prediction_df": secondary_prediction_df,
                "y_true": test_data[target_secondary].values,
                "y_pred_mean": secondary_mean,
                "y_pred_lower": secondary_lower,
                "y_pred_upper": secondary_upper,
            },
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "z_value": z_value,
        }
