import logging
import json
from pathlib import Path
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
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
        subset_main_size: int,
        subset_secondary_size: int,
        subset_tag: str = "subset",
        n_splits: int = 5,
        n_estimators: int = 200,
        random_state: int = 42,
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = None,
        mlflow_experiment: str = None,
        mlflow_run_name: str = None,
        mlflow_tags: dict = None,
        artifact_paths: list[Path] | None = None,
        subset_indices_main: list = None,
        subset_indices_secondary: list = None,
        feature_names_main: list = None,
        feature_names_secondary: list = None,
        log_fold_models_to_mlflow: bool = False,
        log_best_fold_models_to_mlflow: bool = False,
        log_final_models_to_mlflow: bool = True,
        save_local_models: bool = True,

        # NEW: running bests + hard thresholds for MLflow model storage
        current_best_mlflow_main_r2: float = 0.90,
        current_best_mlflow_secondary_r2: float = 0.80,
        min_main_r2_to_log: float = 0.93,
        min_secondary_r2_to_log: float = 0.85,
    ):

        model_dir.mkdir(parents=True, exist_ok=True)

        if use_mlflow:
            if mlflow is None:
                raise ImportError("mlflow is not installed.")

            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)

            mlflow.set_experiment(mlflow_experiment or "rf_augmentation")

            run_name = mlflow_run_name or f"k{subset_main_size}_{subset_tag}"
            run_ctx = mlflow.start_run(run_name=run_name)
        else:
            run_ctx = nullcontext()

        def compute_metrics(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return mse, rmse, mae, r2

        def _run():
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            results = []

            best_main_fold_model = None
            best_sec_fold_model = None
            best_main_r2 = -np.inf
            best_sec_r2 = -np.inf
            best_main_fold = None
            best_sec_fold = None

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_main), 1):
                logger.info(f"[Fold {fold}/{n_splits}]")

                X_main_train, X_main_val = X_main[train_idx], X_main[val_idx]
                X_sec_train, X_sec_val = X_secondary[train_idx], X_secondary[val_idx]

                y_main_train, y_main_val = y_main[train_idx], y_main[val_idx]
                y_sec_train, y_sec_val = y_secondary[train_idx], y_secondary[val_idx]

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

                model_main.fit(X_main_train, y_main_train)
                model_sec.fit(X_sec_train, y_sec_train)

                y_main_pred = model_main.predict(X_main_val)
                y_sec_pred = model_sec.predict(X_sec_val)

                main_metrics = compute_metrics(y_main_val, y_main_pred)
                sec_metrics = compute_metrics(y_sec_val, y_sec_pred)

                results.append({
                    "fold": fold,
                    "main_r2": main_metrics[3],
                    "sec_r2": sec_metrics[3],
                    "main_mse": main_metrics[0],
                    "sec_mse": sec_metrics[0],
                    "main_rmse": main_metrics[1],
                    "sec_rmse": sec_metrics[1],
                    "main_mae": main_metrics[2],
                    "sec_mae": sec_metrics[2],
                })

                if main_metrics[3] > best_main_r2:
                    best_main_r2 = main_metrics[3]
                    best_main_fold = fold
                    best_main_fold_model = model_main

                if sec_metrics[3] > best_sec_r2:
                    best_sec_r2 = sec_metrics[3]
                    best_sec_fold = fold
                    best_sec_fold_model = model_sec

                if use_mlflow and log_fold_models_to_mlflow:
                    mlflow.log_metrics({
                        f"fold_{fold}_main_r2": main_metrics[3],
                        f"fold_{fold}_sec_r2": sec_metrics[3],
                        f"fold_{fold}_main_mse": main_metrics[0],
                        f"fold_{fold}_sec_mse": sec_metrics[0],
                        f"fold_{fold}_main_rmse": main_metrics[1],
                        f"fold_{fold}_sec_rmse": main_metrics[1],
                        f"fold_{fold}_main_mae": main_metrics[2],
                        f"fold_{fold}_sec_mae": sec_metrics[2],
                    })

                    mlflow.sklearn.log_model(
                        model_main,
                        artifact_path=f"cv_models/fold_{fold}_model_main",
                        input_example=X_main_train[:3],
                    )
                    mlflow.sklearn.log_model(
                        model_sec,
                        artifact_path=f"cv_models/fold_{fold}_model_secondary",
                        input_example=X_sec_train[:3],
                    )

            df_folds = pd.DataFrame(results)

            mean_metrics = {
                "r2_main_mean": df_folds["main_r2"].mean(),
                "r2_secondary_mean": df_folds["sec_r2"].mean(),
                "mse_main_mean": df_folds["main_mse"].mean(),
                "mse_secondary_mean": df_folds["sec_mse"].mean(),
                "rmse_main_mean": df_folds["main_rmse"].mean(),
                "rmse_secondary_mean": df_folds["sec_rmse"].mean(),
                "mae_main_mean": df_folds["main_mae"].mean(),
                "mae_secondary_mean": df_folds["sec_mae"].mean(),
            }

            best_metrics = {
                "r2_main_best": df_folds["main_r2"].max(),
                "r2_secondary_best": df_folds["sec_r2"].max(),
                "mse_main_best": df_folds.loc[df_folds["main_r2"].idxmax(), "main_mse"],
                "mse_secondary_best": df_folds.loc[df_folds["sec_r2"].idxmax(), "sec_mse"],
                "rmse_main_best": df_folds.loc[df_folds["main_r2"].idxmax(), "main_rmse"],
                "rmse_secondary_best": df_folds.loc[df_folds["sec_r2"].idxmax(), "sec_rmse"],
                "mae_main_best": df_folds.loc[df_folds["main_r2"].idxmax(), "main_mae"],
                "mae_secondary_best": df_folds.loc[df_folds["sec_r2"].idxmax(), "sec_mae"],
                "best_main_fold": best_main_fold,
                "best_secondary_fold": best_sec_fold,
            }

            logger.info(
                f"Best main fold: {best_main_fold} | best main R2: {best_metrics['r2_main_best']:.6f}"
            )
            logger.info(
                f"Best secondary fold: {best_sec_fold} | best secondary R2: {best_metrics['r2_secondary_best']:.6f}"
            )

            folds_metrics_path = (
                model_dir
                / f"rf_fold_metrics_{subset_tag}_main{subset_main_size}_sec{subset_secondary_size}.csv"
            )
            df_folds.to_csv(folds_metrics_path, index=False)

            final_main = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
            final_sec = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )

            final_main.fit(X_main, y_main)
            final_sec.fit(X_secondary, y_secondary)

            main_path = model_dir / f"rf_main_{subset_tag}_k{subset_main_size}.pkl"
            sec_path = model_dir / f"rf_secondary_{subset_tag}_k{subset_secondary_size}.pkl"

            if save_local_models:
                joblib.dump(final_main, main_path)
                joblib.dump(final_sec, sec_path)
                logger.info(f"Saved local models → {main_path}, {sec_path}")

            summary_row = {
                "subset_tag": subset_tag,
                "subset_main_size": subset_main_size,
                "subset_secondary_size": subset_secondary_size,
                "n_splits": n_splits,
                "n_estimators": n_estimators,
                "random_state": random_state,
                **mean_metrics,
                **best_metrics,
                "subset_indices_main": json.dumps(subset_indices_main) if subset_indices_main is not None else None,
                "subset_indices_secondary": json.dumps(subset_indices_secondary) if subset_indices_secondary is not None else None,
                "feature_names_main": json.dumps(feature_names_main) if feature_names_main is not None else None,
                "feature_names_secondary": json.dumps(feature_names_secondary) if feature_names_secondary is not None else None,
            }

            summary_df = pd.DataFrame([summary_row])
            summary_path = (
                model_dir
                / f"rf_summary_{subset_tag}_main{subset_main_size}_sec{subset_secondary_size}.csv"
            )
            summary_df.to_csv(summary_path, index=False)

            logger.info(f"Saved fold metrics → {folds_metrics_path}")
            logger.info(f"Saved summary metrics/features → {summary_path}")

            # ------------------------------------------------------------
            # NEW: decide whether best-fold models should be stored in MLflow
            # ------------------------------------------------------------
            should_log_main_best_fold = (
                log_best_fold_models_to_mlflow
                and best_main_fold_model is not None
                and best_metrics["r2_main_best"] > min_main_r2_to_log
                and best_metrics["r2_main_best"] > current_best_mlflow_main_r2
            )

            should_log_secondary_best_fold = (
                log_best_fold_models_to_mlflow
                and best_sec_fold_model is not None
                and best_metrics["r2_secondary_best"] > min_secondary_r2_to_log
                and best_metrics["r2_secondary_best"] > current_best_mlflow_secondary_r2
            )

            updated_best_mlflow_main_r2 = current_best_mlflow_main_r2
            updated_best_mlflow_secondary_r2 = current_best_mlflow_secondary_r2

            if should_log_main_best_fold:
                updated_best_mlflow_main_r2 = best_metrics["r2_main_best"]
                logger.info(
                    f"Logging MAIN best-fold model to MLflow | "
                    f"R2={best_metrics['r2_main_best']:.6f} "
                    f"(previous best={current_best_mlflow_main_r2:.6f}, "
                    f"threshold={min_main_r2_to_log:.6f})"
                )
            else:
                logger.info(
                    f"Skipping MAIN best-fold MLflow model logging | "
                    f"R2={best_metrics['r2_main_best']:.6f} "
                    f"(current best={current_best_mlflow_main_r2:.6f}, "
                    f"threshold={min_main_r2_to_log:.6f})"
                )

            if should_log_secondary_best_fold:
                updated_best_mlflow_secondary_r2 = best_metrics["r2_secondary_best"]
                logger.info(
                    f"Logging SECONDARY best-fold model to MLflow | "
                    f"R2={best_metrics['r2_secondary_best']:.6f} "
                    f"(previous best={current_best_mlflow_secondary_r2:.6f}, "
                    f"threshold={min_secondary_r2_to_log:.6f})"
                )
            else:
                logger.info(
                    f"Skipping SECONDARY best-fold MLflow model logging | "
                    f"R2={best_metrics['r2_secondary_best']:.6f} "
                    f"(current best={current_best_mlflow_secondary_r2:.6f}, "
                    f"threshold={min_secondary_r2_to_log:.6f})"
                )

            if use_mlflow:
                mlflow.log_params({
                    "subset_main_size": subset_main_size,
                    "subset_secondary_size": subset_secondary_size,
                    "subset_tag": subset_tag,
                    "n_estimators": n_estimators,
                    "n_splits": n_splits,
                    "random_state": random_state,
                    "main_indices": json.dumps(subset_indices_main) if subset_indices_main is not None else "",
                    "sec_indices": json.dumps(subset_indices_secondary) if subset_indices_secondary is not None else "",
                    "feature_names_main": json.dumps(feature_names_main) if feature_names_main is not None else "",
                    "feature_names_secondary": json.dumps(feature_names_secondary) if feature_names_secondary is not None else "",
                    "log_fold_models_to_mlflow": log_fold_models_to_mlflow,
                    "log_best_fold_models_to_mlflow": log_best_fold_models_to_mlflow,
                    "log_final_models_to_mlflow": log_final_models_to_mlflow,
                    "save_local_models": save_local_models,
                    "current_best_mlflow_main_r2_in": current_best_mlflow_main_r2,
                    "current_best_mlflow_secondary_r2_in": current_best_mlflow_secondary_r2,
                    "min_main_r2_to_log": min_main_r2_to_log,
                    "min_secondary_r2_to_log": min_secondary_r2_to_log,
                    "should_log_main_best_fold": should_log_main_best_fold,
                    "should_log_secondary_best_fold": should_log_secondary_best_fold,
                })

                mlflow.log_metrics({
                    **mean_metrics,
                    **best_metrics,
                    "updated_best_mlflow_main_r2": updated_best_mlflow_main_r2,
                    "updated_best_mlflow_secondary_r2": updated_best_mlflow_secondary_r2,
                })

                mlflow.log_artifact(str(folds_metrics_path))
                mlflow.log_artifact(str(summary_path))

                if artifact_paths:
                    for path in artifact_paths:
                        if Path(path).exists():
                            mlflow.log_artifact(str(path))

                # Usually keep this False in your exhaustive search to save storage
                if log_final_models_to_mlflow:
                    mlflow.sklearn.log_model(
                        final_main,
                        artifact_path="model_main_final",
                        input_example=X_main[:3],
                        signature=infer_signature(X_main[:10], final_main.predict(X_main[:10])),
                    )
                    mlflow.sklearn.log_model(
                        final_sec,
                        artifact_path="model_secondary_final",
                        input_example=X_secondary[:3],
                        signature=infer_signature(X_secondary[:10], final_sec.predict(X_secondary[:10])),
                    )

                if should_log_main_best_fold:
                    mlflow.sklearn.log_model(
                        best_main_fold_model,
                        artifact_path="model_main_best_fold",
                        input_example=X_main[:3],
                    )

                if should_log_secondary_best_fold:
                    mlflow.sklearn.log_model(
                        best_sec_fold_model,
                        artifact_path="model_secondary_best_fold",
                        input_example=X_secondary[:3],
                    )

                if mlflow_tags:
                    mlflow.set_tags(mlflow_tags)

            return {
                "model_main": final_main,
                "model_secondary": final_sec,
                "best_fold_model_main": best_main_fold_model,
                "best_fold_model_secondary": best_sec_fold_model,
                "metrics_mean": mean_metrics,
                "metrics_best": best_metrics,
                "feature_names_main": feature_names_main,
                "feature_names_secondary": feature_names_secondary,
                "subset_indices_main": subset_indices_main,
                "subset_indices_secondary": subset_indices_secondary,
                "fold_metrics_path": folds_metrics_path,
                "summary_path": summary_path,

                # NEW: return updated tracking values to pipeline
                "should_log_main_best_fold": should_log_main_best_fold,
                "should_log_secondary_best_fold": should_log_secondary_best_fold,
                "updated_best_mlflow_main_r2": updated_best_mlflow_main_r2,
                "updated_best_mlflow_secondary_r2": updated_best_mlflow_secondary_r2,
            }

        with run_ctx:
            return _run()