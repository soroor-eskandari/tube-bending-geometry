import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from src.logging.log_utils import log_function


class MeanGPModelTrainer:

    @staticmethod
    @log_function
    def train(
        gp_dataset: np.ndarray,
        model_dir: str = "artifacts/models/hgp",
        paper_name: str = "hgp_mean_geometry",
        random_state: int = 1100,
        n_restarts_optimizer: int = 0,
    ) -> dict:

        dataset = np.asarray(
            gp_dataset,
            dtype=float,
        )

        if dataset.ndim != 2:
            raise ValueError(
                "gp_dataset must be a 2D numpy array."
            )

        expected_column_count = 5

        if dataset.shape[1] != expected_column_count:
            raise ValueError(
                "gp_dataset must contain 5 columns in this order: "
                "Experiment_ID, Group_ID, Angle[degree]ORDistance[mm], "
                "Secondary-axis [mm]_Mean, Main-axis [mm]_Mean."
            )

        if dataset.shape[0] == 0:
            raise ValueError(
                "gp_dataset must contain at least one row."
            )

        # =========================
        # FEATURES / TARGETS
        # =========================
        id_columns = [
            "Experiment_ID",
            "Group_ID",
            "Angle[degree]ORDistance[mm]",
        ]

        feature_columns = [
            "Group_ID",
            "Angle[degree]ORDistance[mm]",
        ]

        target_secondary = "Secondary-axis [mm]_Mean"
        target_main = "Main-axis [mm]_Mean"

        dataset_df = pd.DataFrame(
            {
                id_columns[0]: dataset[:, 0],
                id_columns[1]: dataset[:, 1],
                id_columns[2]: dataset[:, 2],
                target_secondary: dataset[:, 3],
                target_main: dataset[:, 4],
            }
        )

        fit_dataset_df = (
            dataset_df
            .groupby(
                feature_columns,
                as_index=False,
                sort=True,
            )
            .agg(
                **{
                    id_columns[0]: (
                        id_columns[0],
                        "first",
                    ),
                    target_secondary: (
                        target_secondary,
                        "mean",
                    ),
                    target_main: (
                        target_main,
                        "mean",
                    ),
                }
            )
        )

        fit_dataset_df = fit_dataset_df[
            [
                id_columns[0],
                id_columns[1],
                id_columns[2],
                target_secondary,
                target_main,
            ]
        ]

        X_train = fit_dataset_df[
            feature_columns
        ].to_numpy()

        y_train_secondary = fit_dataset_df[
            target_secondary
        ].to_numpy()

        y_train_main = fit_dataset_df[
            target_main
        ].to_numpy()

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
        X_evaluation = dataset_df[
            feature_columns
        ].to_numpy()

        main_pred_mean, main_pred_std = (
            main_model.predict(
                X_evaluation,
                return_std=True,
            )
        )

        secondary_pred_mean, secondary_pred_std = (
            secondary_model.predict(
                X_evaluation,
                return_std=True,
            )
        )

        evaluation_df = pd.DataFrame(
            {
                id_columns[0]: dataset_df[id_columns[0]],
                id_columns[1]: dataset_df[id_columns[1]],
                id_columns[2]: dataset_df[id_columns[2]],
                target_secondary: dataset_df[target_secondary],
                "Secondary-axis [mm]_GP_Mean": (
                    secondary_pred_mean
                ),
                "Secondary-axis [mm]_GP_Std": (
                    secondary_pred_std
                ),
                "Secondary-axis [mm]_Residual": (
                    dataset_df[target_secondary]
                    - secondary_pred_mean
                ),
                target_main: dataset_df[target_main],
                "Main-axis [mm]_GP_Mean": main_pred_mean,
                "Main-axis [mm]_GP_Std": main_pred_std,
                "Main-axis [mm]_Residual": (
                    dataset_df[target_main]
                    - main_pred_mean
                ),
            }
        )

        # =========================
        # SAVE MODELS / OUTPUTS
        # =========================
        main_model_path = (
            model_dir
            / f"mean_gp_main_axis_{paper_name}.pkl"
        )

        secondary_model_path = (
            model_dir
            / f"mean_gp_secondary_axis_{paper_name}.pkl"
        )

        evaluation_path = (
            model_dir
            / f"mean_gp_training_evaluation_{paper_name}.csv"
        )

        config_path = (
            model_dir
            / f"mean_gp_config_{paper_name}.json"
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
            "id_columns": id_columns,
            "feature_columns": feature_columns,
            "target_main": target_main,
            "target_secondary": target_secondary,
            "random_state": random_state,
            "n_restarts_optimizer": n_restarts_optimizer,
            "raw_training_rows": int(dataset.shape[0]),
            "unique_fit_rows": int(fit_dataset_df.shape[0]),
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
            "fit_dataset_df": fit_dataset_df,
            "X_train": X_train,
            "y_train_main": y_train_main,
            "y_train_secondary": y_train_secondary,
            "main_model_path": main_model_path,
            "secondary_model_path": secondary_model_path,
            "evaluation_path": evaluation_path,
            "config_path": config_path,
        }
