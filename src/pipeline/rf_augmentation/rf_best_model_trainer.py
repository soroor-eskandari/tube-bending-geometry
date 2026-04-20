import logging
import json
from pathlib import Path
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except Exception:
    mlflow = None


class RFModelTrainer:

    @staticmethod
    @log_function
    def train(
        X_main,
        X_secondary,
        y_main,
        y_secondary,
        model_dir: Path,
        paper_name: str = "paper_model",
        n_estimators: int = 200,
        random_state: int = 42,
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = None,
        mlflow_experiment: str = None,
        mlflow_run_name: str = None,
        mlflow_tags: dict = None,
        feature_names_main: list = None,
        feature_names_secondary: list = None,
    ):

        model_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # MLflow setup
        # -------------------------
        if use_mlflow:
            if mlflow is None:
                raise ImportError("mlflow is not installed.")

            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)

            mlflow.set_experiment(mlflow_experiment or "rf_final_models")

            run_name = mlflow_run_name or f"FINAL_{paper_name}"
            run_ctx = mlflow.start_run(run_name=run_name)
        else:
            run_ctx = nullcontext()

        # -------------------------
        # Metric function
        # -------------------------
        def compute_metrics(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return mse, rmse, mae, r2

        # -------------------------
        # MAIN EXECUTION
        # -------------------------
        def _run():

            logger.info("Training FINAL models on full dataset")

            # -------------------------
            # Train models
            # -------------------------
            model_main = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )

            model_sec = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )

            model_main.fit(X_main, y_main)
            model_sec.fit(X_secondary, y_secondary)

            # -------------------------
            # Evaluate on training data
            # -------------------------
            y_main_pred = model_main.predict(X_main)
            y_sec_pred = model_sec.predict(X_secondary)

            main_metrics = compute_metrics(y_main, y_main_pred)
            sec_metrics = compute_metrics(y_secondary, y_sec_pred)

            metrics = {
                "r2_main": main_metrics[3],
                "r2_secondary": sec_metrics[3],
                "mse_main": main_metrics[0],
                "mse_secondary": sec_metrics[0],
                "rmse_main": main_metrics[1],
                "rmse_secondary": sec_metrics[1],
                "mae_main": main_metrics[2],
                "mae_secondary": sec_metrics[2],
            }

            logger.info(
                f"FINAL TRAIN METRICS | "
                f"MAIN R2={metrics['r2_main']:.6f} | "
                f"SEC R2={metrics['r2_secondary']:.6f}"
            )

            # -------------------------
            # Save locally
            # -------------------------
            main_path = model_dir / f"rf_main_{paper_name}.pkl"
            sec_path = model_dir / f"rf_secondary_{paper_name}.pkl"

            joblib.dump(model_main, main_path)
            joblib.dump(model_sec, sec_path)

            logger.info(
                f"[SAVED] MAIN → {main_path} | SECONDARY → {sec_path}"
            )

            # -------------------------
            # Save summary CSV
            # -------------------------
            summary = {
                "paper_name": paper_name,
                "n_estimators": n_estimators,
                "random_state": random_state,
                **metrics,
                "feature_names_main": json.dumps(feature_names_main) if feature_names_main else "",
                "feature_names_secondary": json.dumps(feature_names_secondary) if feature_names_secondary else "",
            }

            summary_df = pd.DataFrame([summary])
            summary_path = model_dir / f"rf_summary_{paper_name}.csv"
            summary_df.to_csv(summary_path, index=False)

            # -------------------------
            # MLflow logging
            # -------------------------
            if use_mlflow:

                mlflow.log_params({
                    "paper_name": paper_name,
                    "n_estimators": n_estimators,
                    "random_state": random_state,
                    "n_features_main": X_main.shape[1],
                    "n_features_secondary": X_secondary.shape[1],
                })

                mlflow.log_metrics(metrics)

                mlflow.log_artifact(str(summary_path))

                # Log models
                mlflow.sklearn.log_model(
                    model_main,
                    artifact_path="model_main",
                    input_example=X_main[:3],
                    signature=infer_signature(X_main[:10], model_main.predict(X_main[:10])),
                )

                mlflow.sklearn.log_model(
                    model_sec,
                    artifact_path="model_secondary",
                    input_example=X_secondary[:3],
                    signature=infer_signature(X_secondary[:10], model_sec.predict(X_secondary[:10])),
                )

                if mlflow_tags:
                    mlflow.set_tags(mlflow_tags)

            return {
                "model_main": model_main,
                "model_secondary": model_sec,
                "metrics": metrics,
                "model_main_path": main_path,
                "model_secondary_path": sec_path,
                "summary_path": summary_path,
            }

        with run_ctx:
            return _run()