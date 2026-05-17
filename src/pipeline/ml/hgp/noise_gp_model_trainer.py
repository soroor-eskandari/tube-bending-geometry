import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from src.logging.log_utils import log_function


class NoiseGPModelTrainer:

    @staticmethod
    @log_function
    def train(
        noise_gp_dataset: np.ndarray,
        model_dir: str = "artifacts/models/hgp",
        paper_name: str = "hgp_noise_geometry",
        random_state: int = 1100,
        n_restarts_optimizer: int = 0,
        variance_epsilon: float = 1e-8,
    ) -> dict:

        dataset = np.asarray(
            noise_gp_dataset,
            dtype=float,
        )

        if dataset.ndim != 2:
            raise ValueError(
                "noise_gp_dataset must be a 2D numpy array."
            )

        expected_column_count = 4

        if dataset.shape[1] != expected_column_count:
            raise ValueError(
                "noise_gp_dataset must contain 4 columns in this order: "
                "Group_ID, Angle[degree]ORDistance[mm], "
                "Secondary-axis [mm]_Local_Variance, "
                "Main-axis [mm]_Local_Variance."
            )

        if dataset.shape[0] == 0:
            raise ValueError(
                "noise_gp_dataset must contain at least one row."
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

        target_secondary = (
            "Secondary-axis [mm]_Local_Variance"
        )

        target_main = (
            "Main-axis [mm]_Local_Variance"
        )

        dataset_df = pd.DataFrame(
            {
                feature_columns[0]: dataset[:, 0],
                feature_columns[1]: dataset[:, 1],
                target_secondary: dataset[:, 2],
                target_main: dataset[:, 3],
            }
        )

        dataset_df = dataset_df[
            (
                dataset_df[target_secondary] >= 0
            )
            & (
                dataset_df[target_main] >= 0
            )
        ].reset_index(drop=True)

        if dataset_df.empty:
            raise ValueError(
                "noise_gp_dataset contains no non-negative variance rows."
            )

        X_train = dataset_df[
            feature_columns
        ].to_numpy()

        y_train_secondary = np.log(
            dataset_df[target_secondary].to_numpy()
            + variance_epsilon
        )

        y_train_main = np.log(
            dataset_df[target_main].to_numpy()
            + variance_epsilon
        )

        # =========================
        # CREATE MODEL DIRECTORY
        # =========================
        model_dir = Path(model_dir)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =========================
        # GP KERNEL
        # =========================
        def _build_kernel():
            return (
                ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e3),
                )
                * RBF(
                    length_scale=np.ones(
                        X_train.shape[1],
                    ),
                    length_scale_bounds=(1e-3, 1e3),
                )
                + WhiteKernel(
                    noise_level=1e-5,
                    noise_level_bounds=(1e-10, 1e1),
                )
            )

        # =========================
        # FIT MODELS
        # =========================
        main_model = GaussianProcessRegressor(
            kernel=_build_kernel(),
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        secondary_model = GaussianProcessRegressor(
            kernel=_build_kernel(),
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
        # TRAINING EVALUATION VALUES
        # =========================
        main_log_pred_mean, main_log_pred_std = (
            main_model.predict(
                X_train,
                return_std=True,
            )
        )

        secondary_log_pred_mean, secondary_log_pred_std = (
            secondary_model.predict(
                X_train,
                return_std=True,
            )
        )

        main_pred_variance = np.exp(
            main_log_pred_mean
        )

        secondary_pred_variance = np.exp(
            secondary_log_pred_mean
        )

        evaluation_df = pd.DataFrame(
            {
                feature_columns[0]: dataset_df[
                    feature_columns[0]
                ],
                feature_columns[1]: dataset_df[
                    feature_columns[1]
                ],
                target_secondary: dataset_df[
                    target_secondary
                ],
                "Secondary-axis [mm]_Log_Local_Variance": (
                    y_train_secondary
                ),
                "Secondary-axis [mm]_GP_Log_Variance_Mean": (
                    secondary_log_pred_mean
                ),
                "Secondary-axis [mm]_GP_Log_Variance_Std": (
                    secondary_log_pred_std
                ),
                "Secondary-axis [mm]_GP_Variance": (
                    secondary_pred_variance
                ),
                "Secondary-axis [mm]_Variance_Residual": (
                    dataset_df[target_secondary]
                    - secondary_pred_variance
                ),
                target_main: dataset_df[target_main],
                "Main-axis [mm]_Log_Local_Variance": (
                    y_train_main
                ),
                "Main-axis [mm]_GP_Log_Variance_Mean": (
                    main_log_pred_mean
                ),
                "Main-axis [mm]_GP_Log_Variance_Std": (
                    main_log_pred_std
                ),
                "Main-axis [mm]_GP_Variance": (
                    main_pred_variance
                ),
                "Main-axis [mm]_Variance_Residual": (
                    dataset_df[target_main]
                    - main_pred_variance
                ),
            }
        )

        # =========================
        # SAVE MODELS / OUTPUTS
        # =========================
        main_model_path = (
            model_dir
            / f"noise_gp_main_axis_{paper_name}.pkl"
        )

        secondary_model_path = (
            model_dir
            / f"noise_gp_secondary_axis_{paper_name}.pkl"
        )

        evaluation_path = (
            model_dir
            / f"noise_gp_training_evaluation_{paper_name}.csv"
        )

        config_path = (
            model_dir
            / f"noise_gp_config_{paper_name}.json"
        )

        joblib.dump(
            main_model,
            main_model_path,
        )

        joblib.dump(
            secondary_model,
            secondary_model_path,
        )

        evaluation_df.to_csv(
            evaluation_path,
            index=False,
        )

        config = {
            "paper_name": paper_name,
            "feature_columns": feature_columns,
            "target_main": target_main,
            "target_secondary": target_secondary,
            "random_state": random_state,
            "n_restarts_optimizer": n_restarts_optimizer,
            "variance_epsilon": variance_epsilon,
            "training_rows": int(dataset_df.shape[0]),
            "main_kernel": str(main_model.kernel_),
            "secondary_kernel": str(secondary_model.kernel_),
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
            "feature_columns": feature_columns,
            "target_main": target_main,
            "target_secondary": target_secondary,
            "evaluation_df": evaluation_df,
            "X_train": X_train,
            "y_train_main": y_train_main,
            "y_train_secondary": y_train_secondary,
            "main_model_path": main_model_path,
            "secondary_model_path": secondary_model_path,
            "evaluation_path": evaluation_path,
            "config_path": config_path,
            "variance_epsilon": variance_epsilon,
        }
