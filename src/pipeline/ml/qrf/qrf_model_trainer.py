import logging
import json
from pathlib import Path
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except Exception:
    mlflow = None


class QRFModelTrainer:

    @staticmethod
    @log_function
    def train(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_dir: str = "artifacts/models/qrf",
        paper_name: str = "qrf_geometry",
        n_estimators: int = 300,
        max_depth: int = 20,
        random_state: int = 1100,
        lower_quantile: float = 0.05,
        median_quantile: float = 0.50,
        upper_quantile: float = 0.95,
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = None,
        mlflow_experiment: str = "QRF_Geometry_Model",
        mlflow_run_name: str = None,
        mlflow_tags: dict = None,
    ) -> dict:

        # =====================================================
        # COPY DATA
        # =====================================================
        train_data = train_df.copy()

        test_data = test_df.copy()

        # =====================================================
        # FEATURES / TARGETS
        # =====================================================
        angle_column = (
            "Angle[degree]ORDistance[mm]"
        )

        feature_columns = [
            angle_column,
        ]

        target_main = "Main-axis [mm]"

        target_secondary = (
            "Secondary-axis [mm]"
        )

        X_train = train_data[
            feature_columns
        ]

        X_test = test_data[
            feature_columns
        ]

        y_train_main = train_data[
            target_main
        ]

        y_test_main = test_data[
            target_main
        ]

        y_train_secondary = train_data[
            target_secondary
        ]

        y_test_secondary = test_data[
            target_secondary
        ]

        # =====================================================
        # CREATE MODEL DIRECTORY
        # =====================================================
        model_dir = Path(model_dir)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # MLFLOW SETUP
        # =====================================================
        if use_mlflow:

            if mlflow is None:
                raise ImportError(
                    "mlflow is not installed."
                )

            if mlflow_tracking_uri:

                mlflow.set_tracking_uri(
                    mlflow_tracking_uri
                )

            mlflow.set_experiment(
                mlflow_experiment
            )

            run_name = (
                mlflow_run_name
                or f"QRF_{paper_name}"
            )

            run_ctx = mlflow.start_run(
                run_name=run_name
            )

        else:

            run_ctx = nullcontext()

        # =====================================================
        # MAIN EXECUTION
        # =====================================================
        def _run():

            logger.info(
                "Training QRF models..."
            )

            # -------------------------------------------------
            # MAIN MODEL
            # -------------------------------------------------
            model_main = (
                RandomForestQuantileRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            )

            # -------------------------------------------------
            # SECONDARY MODEL
            # -------------------------------------------------
            model_secondary = (
                RandomForestQuantileRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            )

            # -------------------------------------------------
            # FIT
            # -------------------------------------------------
            model_main.fit(
                X_train,
                y_train_main,
            )

            model_secondary.fit(
                X_train,
                y_train_secondary,
            )

            # -------------------------------------------------
            # MAIN PREDICTIONS
            # -------------------------------------------------
            main_lower = model_main.predict(
                X_test,
                quantiles=lower_quantile,
            )

            main_median = model_main.predict(
                X_test,
                quantiles=median_quantile,
            )

            main_upper = model_main.predict(
                X_test,
                quantiles=upper_quantile,
            )

            # -------------------------------------------------
            # SECONDARY PREDICTIONS
            # -------------------------------------------------
            secondary_lower = (
                model_secondary.predict(
                    X_test,
                    quantiles=lower_quantile,
                )
            )

            secondary_median = (
                model_secondary.predict(
                    X_test,
                    quantiles=median_quantile,
                )
            )

            secondary_upper = (
                model_secondary.predict(
                    X_test,
                    quantiles=upper_quantile,
                )
            )

            # =================================================
            # BUILD PREDICTION DATAFRAMES
            # =================================================
            main_prediction_df = (
                pd.DataFrame(
                    {
                        "Angle[degree]":
                            X_test[
                                angle_column
                            ].values,

                        "y_true":
                            y_test_main.values,

                        "y_pred_lower":
                            main_lower,

                        "y_pred_mean":
                            main_median,

                        "y_pred_upper":
                            main_upper,
                    }
                )
            )

            secondary_prediction_df = (
                pd.DataFrame(
                    {
                        "Angle[degree]":
                            X_test[
                                angle_column
                            ].values,

                        "y_true":
                            y_test_secondary.values,

                        "y_pred_lower":
                            secondary_lower,

                        "y_pred_mean":
                            secondary_median,

                        "y_pred_upper":
                            secondary_upper,
                    }
                )
            )

            # =================================================
            # OPTIONAL:
            # PER-ANGLE AGGREGATION
            # =================================================
            main_per_angle_summary = (
                main_prediction_df
                .groupby(
                    "Angle[degree]"
                )
                .agg(
                    y_true_mean=(
                        "y_true",
                        "mean",
                    ),

                    y_pred_lower_mean=(
                        "y_pred_lower",
                        "mean",
                    ),

                    y_pred_mean_mean=(
                        "y_pred_mean",
                        "mean",
                    ),

                    y_pred_upper_mean=(
                        "y_pred_upper",
                        "mean",
                    ),
                )
                .reset_index()
            )

            secondary_per_angle_summary = (
                secondary_prediction_df
                .groupby(
                    "Angle[degree]"
                )
                .agg(
                    y_true_mean=(
                        "y_true",
                        "mean",
                    ),

                    y_pred_lower_mean=(
                        "y_pred_lower",
                        "mean",
                    ),

                    y_pred_mean_mean=(
                        "y_pred_mean",
                        "mean",
                    ),

                    y_pred_upper_mean=(
                        "y_pred_upper",
                        "mean",
                    ),
                )
                .reset_index()
            )

            # =================================================
            # SAVE MODELS
            # =================================================
            main_model_path = (
                model_dir
                / f"qrf_main_axis_{paper_name}.pkl"
            )

            secondary_model_path = (
                model_dir
                / f"qrf_secondary_axis_{paper_name}.pkl"
            )

            joblib.dump(
                model_main,
                main_model_path,
            )

            joblib.dump(
                model_secondary,
                secondary_model_path,
            )

            logger.info(
                f"[SAVED] MAIN → {main_model_path}"
            )

            logger.info(
                f"[SAVED] SECONDARY → {secondary_model_path}"
            )

            # =================================================
            # SAVE CONFIG
            # =================================================
            config = {
                "paper_name": paper_name,
                "feature_columns": (
                    feature_columns
                ),
                "lower_quantile": (
                    lower_quantile
                ),
                "median_quantile": (
                    median_quantile
                ),
                "upper_quantile": (
                    upper_quantile
                ),
                "n_estimators": (
                    n_estimators
                ),
                "max_depth": max_depth,
                "random_state": (
                    random_state
                ),
            }

            config_path = (
                model_dir
                / f"qrf_config_{paper_name}.json"
            )

            with open(
                config_path,
                "w",
            ) as f:

                json.dump(
                    config,
                    f,
                    indent=4,
                )

            # =================================================
            # LOG TO MLFLOW
            # =================================================
            if use_mlflow:

                mlflow.log_params(config)

                mlflow.log_artifact(
                    str(config_path)
                )

                mlflow.sklearn.log_model(
                    sk_model=model_main,
                    artifact_path=(
                        "qrf_main_axis_model"
                    ),
                    input_example=(
                        X_train[:3]
                    ),
                    signature=(
                        infer_signature(
                            X_train[:10],
                            main_median[:10],
                        )
                    ),
                )

                mlflow.sklearn.log_model(
                    sk_model=model_secondary,
                    artifact_path=(
                        "qrf_secondary_axis_model"
                    ),
                    input_example=(
                        X_train[:3]
                    ),
                    signature=(
                        infer_signature(
                            X_train[:10],
                            secondary_median[:10],
                        )
                    ),
                )

                if mlflow_tags:

                    mlflow.set_tags(
                        mlflow_tags
                    )

            # =================================================
            # RETURN
            # =================================================
            return {

                # MODELS
                "model_main":
                    model_main,

                "model_secondary":
                    model_secondary,

                # MAIN
                "main": {

                    "prediction_df":
                        main_prediction_df,

                    "per_angle_summary":
                        main_per_angle_summary,

                    "y_true":
                        y_test_main.values,

                    "y_pred_lower":
                        main_lower,

                    "y_pred_mean":
                        main_median,

                    "y_pred_upper":
                        main_upper,
                },

                # SECONDARY
                "secondary": {

                    "prediction_df":
                        secondary_prediction_df,

                    "per_angle_summary":
                        (
                            secondary_per_angle_summary
                        ),

                    "y_true":
                        (
                            y_test_secondary.values
                        ),

                    "y_pred_lower":
                        secondary_lower,

                    "y_pred_mean":
                        secondary_median,

                    "y_pred_upper":
                        secondary_upper,
                },

                # PATHS
                "model_main_path":
                    main_model_path,

                "model_secondary_path":
                    secondary_model_path,

                "config_path":
                    config_path,
            }

        with run_ctx:
            return _run()