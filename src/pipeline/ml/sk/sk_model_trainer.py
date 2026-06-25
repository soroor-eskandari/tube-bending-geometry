import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.logging.log_utils import log_function


class SKModelTrainer:

    @staticmethod
    @log_function
    def train(
        sk_dataset: np.ndarray,
        model_dir: str = "artifacts/models/sk",
        paper_name: str = "sk_geometry",
        random_state: int = 1100,
        n_restarts_optimizer: int = 0,
        variance_epsilon: float = 1e-8,
    ) -> dict:

        dataset = np.asarray(
            sk_dataset,
            dtype=float,
        )

        if dataset.ndim != 2:
            raise ValueError(
                "sk_dataset must be a 2D numpy array."
            )

        expected_column_count = 7

        if dataset.shape[1] != expected_column_count:
            raise ValueError(
                "sk_dataset must contain 7 columns in this order: "
                "Group_ID, Angle[degree]ORDistance[mm], "
                "Secondary-axis [mm]_Mean, Main-axis [mm]_Mean, "
                "Secondary-axis [mm]_Local_Variance, "
                "Main-axis [mm]_Local_Variance, Repeat_Count."
            )

        if dataset.shape[0] == 0:
            raise ValueError(
                "sk_dataset must contain at least one row."
            )

        if variance_epsilon <= 0:
            raise ValueError(
                "variance_epsilon must be greater than 0."
            )

        # =========================
        # FEATURES / TARGETS
        # =========================
        feature_columns = [
            "Group_ID",
            "Angle[degree]ORDistance[mm]",
        ]

        target_secondary_mean = (
            "Secondary-axis [mm]_Mean"
        )
        target_main_mean = "Main-axis [mm]_Mean"
        target_secondary_variance = (
            "Secondary-axis [mm]_Local_Variance"
        )
        target_main_variance = (
            "Main-axis [mm]_Local_Variance"
        )

        dataset_df = pd.DataFrame(
            {
                feature_columns[0]: dataset[:, 0],
                feature_columns[1]: dataset[:, 1],
                target_secondary_mean: dataset[:, 2],
                target_main_mean: dataset[:, 3],
                target_secondary_variance: dataset[:, 4],
                target_main_variance: dataset[:, 5],
                "Repeat_Count": dataset[:, 6],
            }
        )

        dataset_df = dataset_df[
            (
                dataset_df[target_secondary_variance]
                >= 0
            )
            & (
                dataset_df[target_main_variance]
                >= 0
            )
            & (
                dataset_df["Repeat_Count"]
                > 0
            )
        ].reset_index(drop=True)

        if dataset_df.empty:
            raise ValueError(
                "sk_dataset contains no valid repeated design points."
            )

        X_train = dataset_df[
            feature_columns
        ].to_numpy()

        y_train_secondary = dataset_df[
            target_secondary_mean
        ].to_numpy()

        y_train_main = dataset_df[
            target_main_mean
        ].to_numpy()

        repeat_count = dataset_df[
            "Repeat_Count"
        ].to_numpy()

        alpha_secondary = (
            dataset_df[target_secondary_variance].to_numpy()
            / repeat_count
            + variance_epsilon
        )

        alpha_main = (
            dataset_df[target_main_variance].to_numpy()
            / repeat_count
            + variance_epsilon
        )

        # =========================
        # MODEL DIRECTORY
        # =========================
        model_dir = Path(model_dir)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =========================
        # GP KERNELS
        # =========================
        def _build_sk_kernel():
            return (
                ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e3),
                )
                * Matern(
                    length_scale=np.ones(
                        X_train.shape[1],
                    ),
                    length_scale_bounds=(1e-3, 1e3),
                    nu=1.5,
                )
            )

        def _build_noise_kernel():
            return (
                ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e3),
                )
                * Matern(
                    length_scale=np.ones(
                        X_train.shape[1],
                    ),
                    length_scale_bounds=(1e-3, 1e3),
                    nu=1.5,
                )
                + WhiteKernel(
                    noise_level=1e-5,
                    noise_level_bounds=(1e-10, 1e1),
                )
            )

        # =========================
        # FIT STOCHASTIC KRIGING MEAN MODELS
        # =========================
        main_model = GaussianProcessRegressor(
            kernel=_build_sk_kernel(),
            alpha=alpha_main,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        secondary_model = GaussianProcessRegressor(
            kernel=_build_sk_kernel(),
            alpha=alpha_secondary,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        main_model.fit(
            X_train,
            y_train_main,
        )

        secondary_model.fit(
            X_train,
            y_train_secondary,
        )

        # =========================
        # FIT INTRINSIC VARIANCE MODELS
        # =========================
        main_noise_model = GaussianProcessRegressor(
            kernel=_build_noise_kernel(),
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        secondary_noise_model = GaussianProcessRegressor(
            kernel=_build_noise_kernel(),
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        y_train_main_log_variance = np.log(
            dataset_df[target_main_variance].to_numpy()
            + variance_epsilon
        )

        y_train_secondary_log_variance = np.log(
            dataset_df[target_secondary_variance].to_numpy()
            + variance_epsilon
        )

        main_noise_model.fit(
            X_train,
            y_train_main_log_variance,
        )

        secondary_noise_model.fit(
            X_train,
            y_train_secondary_log_variance,
        )

        # =========================
        # TRAINING EVALUATION VALUES
        # =========================
        main_pred_mean, main_surface_std = (
            main_model.predict(
                X_train,
                return_std=True,
            )
        )

        secondary_pred_mean, secondary_surface_std = (
            secondary_model.predict(
                X_train,
                return_std=True,
            )
        )

        main_log_noise_pred = (
            main_noise_model.predict(
                X_train,
            )
        )

        secondary_log_noise_pred = (
            secondary_noise_model.predict(
                X_train,
            )
        )

        evaluation_df = pd.DataFrame(
            {
                feature_columns[0]: dataset_df[feature_columns[0]],
                feature_columns[1]: dataset_df[feature_columns[1]],
                "Repeat_Count": repeat_count,
                target_secondary_mean: y_train_secondary,
                "Secondary-axis [mm]_SK_Mean": secondary_pred_mean,
                "Secondary-axis [mm]_SK_Surface_Std": (
                    secondary_surface_std
                ),
                "Secondary-axis [mm]_Alpha": alpha_secondary,
                target_secondary_variance: dataset_df[
                    target_secondary_variance
                ],
                "Secondary-axis [mm]_Predicted_Intrinsic_Variance": (
                    np.exp(secondary_log_noise_pred)
                ),
                "Secondary-axis [mm]_Residual": (
                    y_train_secondary
                    - secondary_pred_mean
                ),
                target_main_mean: y_train_main,
                "Main-axis [mm]_SK_Mean": main_pred_mean,
                "Main-axis [mm]_SK_Surface_Std": main_surface_std,
                "Main-axis [mm]_Alpha": alpha_main,
                target_main_variance: dataset_df[
                    target_main_variance
                ],
                "Main-axis [mm]_Predicted_Intrinsic_Variance": (
                    np.exp(main_log_noise_pred)
                ),
                "Main-axis [mm]_Residual": (
                    y_train_main
                    - main_pred_mean
                ),
            }
        )

        # =========================
        # SAVE MODELS / OUTPUTS
        # =========================
        main_model_path = (
            model_dir
            / f"sk_main_axis_{paper_name}.pkl"
        )

        secondary_model_path = (
            model_dir
            / f"sk_secondary_axis_{paper_name}.pkl"
        )

        main_noise_model_path = (
            model_dir
            / f"sk_noise_main_axis_{paper_name}.pkl"
        )

        secondary_noise_model_path = (
            model_dir
            / f"sk_noise_secondary_axis_{paper_name}.pkl"
        )

        evaluation_path = (
            model_dir
            / f"sk_training_evaluation_{paper_name}.csv"
        )

        config_path = (
            model_dir
            / f"sk_config_{paper_name}.json"
        )

        joblib.dump(
            main_model,
            main_model_path,
        )

        joblib.dump(
            secondary_model,
            secondary_model_path,
        )

        joblib.dump(
            main_noise_model,
            main_noise_model_path,
        )

        joblib.dump(
            secondary_noise_model,
            secondary_noise_model_path,
        )

        evaluation_df.to_csv(
            evaluation_path,
            index=False,
        )

        config = {
            "paper_name": paper_name,
            "feature_columns": feature_columns,
            "target_main": "Main-axis [mm]",
            "target_secondary": "Secondary-axis [mm]",
            "target_main_mean": target_main_mean,
            "target_secondary_mean": target_secondary_mean,
            "target_main_variance": target_main_variance,
            "target_secondary_variance": target_secondary_variance,
            "random_state": random_state,
            "n_restarts_optimizer": n_restarts_optimizer,
            "variance_epsilon": variance_epsilon,
            "training_rows": int(dataset_df.shape[0]),
            "main_kernel": str(main_model.kernel_),
            "secondary_kernel": str(secondary_model.kernel_),
            "main_noise_kernel": str(main_noise_model.kernel_),
            "secondary_noise_kernel": str(
                secondary_noise_model.kernel_
            ),
        }

        with open(
            config_path,
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=4,
            )

        # =========================
        # FINAL OUTPUT
        # =========================
        return {
            "main_model": main_model,
            "secondary_model": secondary_model,
            "main_noise_model": main_noise_model,
            "secondary_noise_model": secondary_noise_model,
            "feature_columns": feature_columns,
            "target_main": "Main-axis [mm]",
            "target_secondary": "Secondary-axis [mm]",
            "evaluation_df": evaluation_df,
            "X_train": X_train,
            "y_train_main": y_train_main,
            "y_train_secondary": y_train_secondary,
            "alpha_main": alpha_main,
            "alpha_secondary": alpha_secondary,
            "main_model_path": main_model_path,
            "secondary_model_path": secondary_model_path,
            "main_noise_model_path": main_noise_model_path,
            "secondary_noise_model_path": secondary_noise_model_path,
            "evaluation_path": evaluation_path,
            "config_path": config_path,
            "variance_epsilon": variance_epsilon,
        }
