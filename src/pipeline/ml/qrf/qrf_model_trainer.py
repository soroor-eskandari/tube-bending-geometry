import logging
import json
from pathlib import Path
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    def _interval_metrics(
        y_true,
        y_pred_median,
        y_pred_lower,
        y_pred_upper,
        lower_quantile: float,
        upper_quantile: float,
    ) -> dict[str, float]:
        y_true = np.asarray(y_true)
        y_pred_median = np.asarray(y_pred_median)
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)

        coverage = np.mean(
            (y_true >= y_pred_lower)
            & (y_true <= y_pred_upper)
        )
        expected_coverage = upper_quantile - lower_quantile
        mse = mean_squared_error(
            y_true,
            y_pred_median,
        )

        return {
            "r2": r2_score(
                y_true,
                y_pred_median,
            ),
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(
                y_true,
                y_pred_median,
            ),
            "bias": float(np.mean(y_pred_median - y_true)),
            "mse": mse,
            "coverage": coverage,
            "expected_coverage": expected_coverage,
            "calibration_error": abs(
                coverage - expected_coverage
            ),
            "mean_interval_width": np.mean(
                y_pred_upper - y_pred_lower
            ),
        }

    @staticmethod
    def _group_coverage_metrics(
        prediction_df: pd.DataFrame,
    ) -> dict[str, float]:
        prediction_df = prediction_df.copy()
        prediction_df["inside_interval"] = (
            (prediction_df["y_true"] >= prediction_df["y_pred_lower"])
            & (prediction_df["y_true"] <= prediction_df["y_pred_upper"])
        )
        group_coverage = (
            prediction_df
            .groupby("Group_ID")["inside_interval"]
            .mean()
        )

        return {
            "mean_group_coverage": group_coverage.mean(),
            "min_group_coverage": group_coverage.min(),
            "std_group_coverage": group_coverage.std(ddof=0),
            "n_groups": float(group_coverage.shape[0]),
        }

    @staticmethod
    @log_function
    def train(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_dir: str = "artifacts/models/qrf",
        paper_name: str = "qrf_geometry",
        n_estimators: int = 300,
        max_depth: int = 20,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: str | float = 1.0,
        bootstrap: bool = True,
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

        group_column = "Group_ID"

        feature_columns = [
            group_column,
            angle_column,
        ]

        missing_feature_columns = [
            column for column in feature_columns
            if column not in train_data.columns
        ]

        if missing_feature_columns:
            raise ValueError(
                "Missing required QRF feature columns in train_df: "
                f"{missing_feature_columns}"
            )

        missing_test_feature_columns = [
            column for column in feature_columns
            if column not in test_data.columns
        ]

        if missing_test_feature_columns:
            raise ValueError(
                "Missing required QRF feature columns in test_df: "
                f"{missing_test_feature_columns}"
            )

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
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    bootstrap=bootstrap,
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
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    bootstrap=bootstrap,
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

                        "Group_ID":
                            X_test[
                                group_column
                            ].values,

                        "y_true":
                            y_test_main.values,

                        "y_pred_lower":
                            main_lower,

                        "y_pred_median":
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

                        "Group_ID":
                            X_test[
                                group_column
                            ].values,

                        "y_true":
                            y_test_secondary.values,

                        "y_pred_lower":
                            secondary_lower,

                        "y_pred_median":
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
                    y_true_median=(
                        "y_true",
                        "median",
                    ),

                    y_pred_lower_median=(
                        "y_pred_lower",
                        "median",
                    ),

                    y_pred_median=(
                        "y_pred_median",
                        "median",
                    ),

                    y_pred_upper_median=(
                        "y_pred_upper",
                        "median",
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
                    y_true_median=(
                        "y_true",
                        "median",
                    ),

                    y_pred_lower_median=(
                        "y_pred_lower",
                        "median",
                    ),

                    y_pred_median=(
                        "y_pred_median",
                        "median",
                    ),

                    y_pred_upper_median=(
                        "y_pred_upper",
                        "median",
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
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
                "bootstrap": bootstrap,
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

                main_metrics = QRFModelTrainer._interval_metrics(
                    y_true=y_test_main.values,
                    y_pred_median=main_median,
                    y_pred_lower=main_lower,
                    y_pred_upper=main_upper,
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                )
                secondary_metrics = QRFModelTrainer._interval_metrics(
                    y_true=y_test_secondary.values,
                    y_pred_median=secondary_median,
                    y_pred_lower=secondary_lower,
                    y_pred_upper=secondary_upper,
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                )
                main_group_metrics = QRFModelTrainer._group_coverage_metrics(
                    main_prediction_df
                )
                secondary_group_metrics = QRFModelTrainer._group_coverage_metrics(
                    secondary_prediction_df
                )
                mean_group_coverage = np.mean(
                    [
                        main_group_metrics["mean_group_coverage"],
                        secondary_group_metrics["mean_group_coverage"],
                    ]
                )

                mlflow.log_metrics(
                    {
                        **{
                            f"main_{key}": value
                            for key, value in main_metrics.items()
                        },
                        **{
                            f"main_{key}": value
                            for key, value in main_group_metrics.items()
                        },
                        **{
                            f"secondary_{key}": value
                            for key, value in secondary_metrics.items()
                        },
                        **{
                            f"secondary_{key}": value
                            for key, value in secondary_group_metrics.items()
                        },
                        "mean_calibration_error": np.mean(
                            [
                                main_metrics["calibration_error"],
                                secondary_metrics["calibration_error"],
                            ]
                        ),
                        "mean_interval_width": np.mean(
                            [
                                main_metrics["mean_interval_width"],
                                secondary_metrics["mean_interval_width"],
                            ]
                        ),
                        "mean_rmse": np.mean(
                            [
                                main_metrics["rmse"],
                                secondary_metrics["rmse"],
                            ]
                        ),
                        "mean_group_coverage": np.mean(
                            [
                                main_group_metrics["mean_group_coverage"],
                                secondary_group_metrics["mean_group_coverage"],
                            ]
                        ),
                        "min_group_coverage": np.min(
                            [
                                main_group_metrics["min_group_coverage"],
                                secondary_group_metrics["min_group_coverage"],
                            ]
                        ),
                        "std_group_coverage": np.mean(
                            [
                                main_group_metrics["std_group_coverage"],
                                secondary_group_metrics["std_group_coverage"],
                            ]
                        ),
                        "group_calibration_error": abs(
                            mean_group_coverage
                            - (upper_quantile - lower_quantile)
                        ),
                    }
                )

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

                    "y_pred_median":
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

                    "y_pred_median":
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
