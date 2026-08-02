from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold

from mode.experiments.data_splittor import (
    load_split_metadata,
    make_train_test_split,
)
from mode.experiments.geometry_data_preprocessor import (
    load_bending_setups,
    load_selected_geometry_source,
)
from mode.experiments.qrf_model_trainer import (
    GroupWiseTargetNormalizer,
)


TARGET_COLUMNS_BY_AXIS = {
    "main": "Main-axis [mm]",
    "secondary": "Secondary-axis [mm]",
}


DEFAULT_PARAM_GRID = {
    # 200 is a stable baseline; 100 and 400 test convergence around it.
    "n_estimators": [200, 100, 400],
    "max_depth": [6, 8, 10],
    "min_samples_leaf": [16, 24, 32],
    "min_samples_split": [20,40],
    "max_features": [0.3, 0.5, "sqrt"],
    "bootstrap": [True],
}


DEFAULT_QUANTILE_GRID = [
    (0.01, 0.99),
    (0.025, 0.975),
    (0.05, 0.95),
    (0.10, 0.90),
]


@dataclass
class HyperParameterTuning:
    """
    Tune QRF hyperparameters for the best ranked split of each axis.

    Only a parquet summary is persisted. Models, predictions and run
    directories are intentionally not stored.
    """

    config: dict[str, Any]
    param_grid: dict[str, list[Any]] | None = None
    quantile_grid: list[tuple[float, float]] | None = None
    cv_splits: int = 5
    use_early_stopping: bool = True
    greedy_patience: int = 2
    max_coordinate_passes: int = 4
    curve_order_column: str | None = None
    curve_distance_tolerance: float = 0.01
    trend_tolerance: float = 0.01
    coverage_target: float = 85.0
    coverage_tolerance: float = 1.0
    pinaw_tolerance: float = 0.01
    oob_curve_distance_tolerance: float = 0.01
    oob_trend_tolerance: float = 0.01
    rank_value: int = 1
    output_filename: str = (
        "qrf_hyperparameter_tuning_results.parquet"
    )

    def run(self) -> pd.DataFrame:
        project_root = Path(
            self.config["_project_root"]
        ).resolve()
        qrf_data_dir = (
            project_root
            / "src"
            / "pipeline"
            / "ml"
            / "qrf"
            / "data"
        )
        qrf_data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        geometry_df, geometry_source, geometry_path = (
            self._load_geometry(project_root)
        )
        split_metadata_df = load_split_metadata(
            self._resolve_path(
                project_root,
                self.config["paths"]["split_metadata"],
            )
        )

        feature_columns = list(
            self.config["features"]["input_columns"]
        )
        target_columns = dict(
            self.config["features"].get(
                "target_columns",
                TARGET_COLUMNS_BY_AXIS,
            )
        )

        rows = []

        for axis in ("main", "secondary"):
            target_column = target_columns.get(axis)

            if target_column is None:
                raise KeyError(
                    f"No target column configured for axis={axis!r}."
                )

            split_index = self._best_split_index(
                split_metadata_df=split_metadata_df,
                axis=axis,
            )

            train_df, test_df, split_row = make_train_test_split(
                geometry_df=geometry_df,
                split_metadata_df=split_metadata_df,
                split_index=split_index,
            )

            fit_search = (
                self._fit_early_stopping_search
                if self.use_early_stopping
                else self._fit_exhaustive_search
            )

            tuning_result = fit_search(
                train_df=train_df,
                feature_columns=feature_columns,
                target_column=target_column,
            )

            metrics = self._evaluate_on_test(
                estimator=tuning_result["best_estimator"],
                target_normalizer=tuning_result[
                    "target_normalizer"
                ],
                test_df=test_df,
                target_column=target_column,
                lower_quantile=tuning_result["lower_quantile"],
                upper_quantile=tuning_result["upper_quantile"],
            )

            row = {
                "target_axis": axis,
                "is_main": axis == "main",
                "is_secondary": axis == "secondary",
                "split_index": int(split_index),
                "split_name": str(
                    split_row.get(
                        "split_name",
                        f"split_{split_index}",
                    )
                ),
                "split_rank": int(self.rank_value),
                "geometry_source": geometry_source,
                "geometry_path": str(geometry_path),
                "target_column": target_column,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_groups": int(
                    train_df["Group_ID"].nunique()
                ),
                "test_groups": int(
                    test_df["Group_ID"].nunique()
                ),
                "std_train_group": (
                    self._mean_group_target_std(
                        train_df,
                        target_column,
                    )
                ),
                "std_test_group": (
                    self._mean_group_target_std(
                        test_df,
                        target_column,
                    )
                ),
                "cv_best_score": float(
                    tuning_result["best_score"]
                ),
                "cv_curve_distance": float(
                    tuning_result["best_objective"]["curve_distance"]
                ),
                "cv_trend_error": float(
                    tuning_result["best_objective"]["trend_error"]
                ),
                "cv_coverage": float(
                    tuning_result["best_objective"]["coverage"]
                ),
                "cv_pinaw": float(
                    tuning_result["best_objective"]["pinaw"]
                ),
                "cv_oob_curve_distance": float(
                    tuning_result["best_objective"]["oob_curve_distance"]
                ),
                "cv_oob_trend_error": float(
                    tuning_result["best_objective"]["oob_trend_error"]
                ),
                "cv_oob_coverage": float(
                    tuning_result["best_objective"]["oob_coverage"]
                ),
                "cv_oob_pinaw": float(
                    tuning_result["best_objective"]["oob_pinaw"]
                ),
                "cv_oob_valid_fraction": float(
                    tuning_result["best_objective"]["oob_valid_fraction"]
                ),
                "greedy_trial_count": int(
                    tuning_result["trial_count"]
                ),
                "tuning_trial_count": int(
                    tuning_result["trial_count"]
                ),
                "coordinate_pass_count": int(
                    tuning_result.get("completed_passes", 1)
                ),
                "tuning_search_strategy": (
                    "iterative_coordinate_search"
                    if self.use_early_stopping
                    else "exhaustive"
                ),
                "best_lower_quantile": float(
                    tuning_result["lower_quantile"]
                ),
                "best_upper_quantile": float(
                    tuning_result["upper_quantile"]
                ),
                **self._flatten_best_params(
                    tuning_result["best_params"]
                ),
                **metrics,
            }
            rows.append(row)

            print(
                f"Finished {axis}: split={split_index}, "
                f"coverage={metrics['coverage_test']:.2f}, "
                f"mae={metrics['mae_test']:.4f}, "
                f"pinaw={metrics['pinaw_test']:.4f}, "
                f"rmse={metrics['rmse_test']:.4f}"
            )

        results_df = self._add_final_scores(
            pd.DataFrame(rows)
        )

        output_path = qrf_data_dir / self.output_filename
        results_df.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"Saved tuning summary to {output_path}"
        )

        return results_df

    def _load_geometry(
        self,
        project_root: Path,
    ) -> tuple[pd.DataFrame, str, Path]:
        paths = self.config["paths"]

        bending_setups_df = load_bending_setups(
            path=self._resolve_path(
                project_root,
                paths["bending_setups"],
            ),
            excluded_experiments=self.config["data"].get(
                "excluded_experiments",
                [],
            ),
        )

        return load_selected_geometry_source(
            project_root=project_root,
            paths_config=paths,
            data_config=self.config["data"],
            bending_setups_df=bending_setups_df,
        )

    def _fit_early_stopping_search(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> dict[str, Any]:
        """Iterative coordinate search with pass-level early stopping.

        Every candidate value of a parameter is evaluated. The search does
        not stop after the first worse value. After one full pass, it repeats
        all parameters using the newly selected configuration. Search stops
        only after ``greedy_patience`` complete passes without improvement.
        """
        self._validate_training_columns(
            train_df, feature_columns, target_column
        )
        model_config = dict(self.config.get("model", {}))
        X_train = train_df[feature_columns]
        y_train = train_df[target_column].to_numpy()
        groups = train_df["Group_ID"].to_numpy()
        experiment_ids = train_df["Experiment_ID"].to_numpy()

        param_grid = self._parameter_grid()
        best_params = self._initial_params(model_config, param_grid)
        lower_quantile, upper_quantile = self._quantile_grid()[0]
        best_objective = self._cross_validated_score(
            X_train, y_train, groups, experiment_ids,
            best_params, lower_quantile, upper_quantile,
        )
        trial_count = 1
        stale_passes = 0
        completed_passes = 0

        search_dimensions = [
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "n_estimators",
            "bootstrap",
            "quantile_interval",
        ]

        for pass_index in range(max(1, int(self.max_coordinate_passes))):
            pass_improved = False

            for param_name in search_dimensions:
                values = (
                    self._quantile_grid()
                    if param_name == "quantile_interval"
                    else param_grid[param_name]
                )
                dimension_best_objective = best_objective
                dimension_best_params = dict(best_params)
                dimension_best_lower = lower_quantile
                dimension_best_upper = upper_quantile

                for value in values:
                    candidate_params = dict(best_params)
                    candidate_lower = lower_quantile
                    candidate_upper = upper_quantile

                    if param_name == "quantile_interval":
                        candidate_lower, candidate_upper = value
                        if (
                            np.isclose(candidate_lower, lower_quantile)
                            and np.isclose(candidate_upper, upper_quantile)
                        ):
                            continue
                    else:
                        if value == best_params.get(param_name):
                            continue
                        candidate_params[param_name] = value

                    candidate_objective = self._cross_validated_score(
                        X_train, y_train, groups, experiment_ids,
                        candidate_params, candidate_lower, candidate_upper,
                    )
                    trial_count += 1

                    if self._objective_is_better(
                        candidate_objective,
                        dimension_best_objective,
                    ):
                        dimension_best_objective = candidate_objective
                        dimension_best_params = candidate_params
                        dimension_best_lower = candidate_lower
                        dimension_best_upper = candidate_upper

                if self._objective_is_better(
                    dimension_best_objective, best_objective
                ):
                    best_objective = dimension_best_objective
                    best_params = dimension_best_params
                    lower_quantile = dimension_best_lower
                    upper_quantile = dimension_best_upper
                    pass_improved = True

            completed_passes = pass_index + 1
            if pass_improved:
                stale_passes = 0
            else:
                stale_passes += 1
                if stale_passes >= max(1, int(self.greedy_patience)):
                    break

        return self._finalize_search_result(
            train_df=train_df,
            X_train=X_train,
            target_column=target_column,
            model_config=model_config,
            best_params=best_params,
            best_objective=best_objective,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            trial_count=trial_count,
            completed_passes=completed_passes,
        )
    def _fit_exhaustive_search(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> dict[str, Any]:
        self._validate_training_columns(
            train_df, feature_columns, target_column
        )
        model_config = dict(self.config.get("model", {}))
        X_train = train_df[feature_columns]
        y_train = train_df[target_column].to_numpy()
        groups = train_df["Group_ID"].to_numpy()
        experiment_ids = train_df["Experiment_ID"].to_numpy()

        best_objective = None
        best_params = None
        lower_quantile = None
        upper_quantile = None
        trial_count = 0

        for candidate_params in self._parameter_combinations(
            self._parameter_grid()
        ):
            for candidate_lower, candidate_upper in self._quantile_grid():
                candidate_objective = self._cross_validated_score(
                    X_train, y_train, groups, experiment_ids,
                    candidate_params, candidate_lower, candidate_upper,
                )
                trial_count += 1
                if (
                    best_objective is None
                    or self._objective_is_better(
                        candidate_objective, best_objective
                    )
                ):
                    best_objective = candidate_objective
                    best_params = dict(candidate_params)
                    lower_quantile = candidate_lower
                    upper_quantile = candidate_upper

        return self._finalize_search_result(
            train_df=train_df,
            X_train=X_train,
            target_column=target_column,
            model_config=model_config,
            best_params=best_params,
            best_objective=best_objective,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            trial_count=trial_count,
            completed_passes=1,
        )
    def _finalize_search_result(
        self,
        train_df: pd.DataFrame,
        X_train: pd.DataFrame,
        target_column: str,
        model_config: dict[str, Any],
        best_params: dict[str, Any],
        best_objective: dict[str, float],
        lower_quantile: float,
        upper_quantile: float,
        trial_count: int,
        completed_passes: int,
    ) -> dict[str, Any]:
        best_estimator = self._build_estimator(best_params, model_config)
        target_normalizer = GroupWiseTargetNormalizer.fit(
            frame=train_df,
            target_column=target_column,
        )
        best_estimator.fit(
            X_train,
            target_normalizer.transform_frame(
                frame=train_df,
                target_column=target_column,
            ),
        )
        return {
            "best_estimator": best_estimator,
            "target_normalizer": target_normalizer,
            "best_score": -float(best_objective["curve_distance"]),
            "best_objective": best_objective,
            "best_params": best_params,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "trial_count": trial_count,
            "completed_passes": completed_passes,
        }
    def _validate_training_columns(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        required = feature_columns + [
            target_column,
            "Group_ID",
            "Experiment_ID",
        ]
        missing_columns = set(required).difference(train_df.columns)
        if missing_columns:
            raise KeyError(
                "Training data is missing columns: "
                f"{sorted(missing_columns)}"
            )

    def _resolve_curve_order_values(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if self.curve_order_column is not None:
            if self.curve_order_column not in X.columns:
                raise KeyError(
                    "curve_order_column is not present in the input "
                    f"features: {self.curve_order_column!r}"
                )
            return X[self.curve_order_column].to_numpy()

        candidates = (
            "Angle [degree]",
            "Angle [deg]",
            "Angle",
            "angle",
            "Target-angle",
        )
        for column in candidates:
            if column in X.columns:
                return X[column].to_numpy()

        # Preserve row order when no explicit angle feature exists.
        return np.arange(len(X), dtype=float)

    def _objective_is_better(
        self,
        candidate: dict[str, float],
        incumbent: dict[str, float],
    ) -> bool:
        """Compare candidates using validation first and OOB second.

        Selection order
        ---------------
        1. Minimise GroupKFold validation curve distance.
        2. Within tolerance, minimise OOB curve distance.
        3. Minimise validation trend error.
        4. Within tolerance, minimise OOB trend error.
        5. Reach the requested validation coverage.
        6. Minimise validation PINAW.
        7. Prefer higher validation coverage as the final tie-breaker.

        OOB is not used instead of grouped validation. It acts as an
        internal anti-overfitting tie-breaker computed only from samples
        that were not used to grow the corresponding trees.
        """
        curve_delta = (
            candidate["curve_distance"]
            - incumbent["curve_distance"]
        )
        if curve_delta < -float(self.curve_distance_tolerance):
            return True
        if curve_delta > float(self.curve_distance_tolerance):
            return False

        oob_curve_delta = (
            candidate["oob_curve_distance"]
            - incumbent["oob_curve_distance"]
        )
        if oob_curve_delta < -float(self.oob_curve_distance_tolerance):
            return True
        if oob_curve_delta > float(self.oob_curve_distance_tolerance):
            return False

        trend_delta = (
            candidate["trend_error"]
            - incumbent["trend_error"]
        )
        if trend_delta < -float(self.trend_tolerance):
            return True
        if trend_delta > float(self.trend_tolerance):
            return False

        oob_trend_delta = (
            candidate["oob_trend_error"]
            - incumbent["oob_trend_error"]
        )
        if oob_trend_delta < -float(self.oob_trend_tolerance):
            return True
        if oob_trend_delta > float(self.oob_trend_tolerance):
            return False

        candidate_shortfall = max(
            0.0,
            float(self.coverage_target) - candidate["coverage"],
        )
        incumbent_shortfall = max(
            0.0,
            float(self.coverage_target) - incumbent["coverage"],
        )
        if candidate_shortfall < (
            incumbent_shortfall - float(self.coverage_tolerance)
        ):
            return True
        if candidate_shortfall > (
            incumbent_shortfall + float(self.coverage_tolerance)
        ):
            return False

        pinaw_delta = candidate["pinaw"] - incumbent["pinaw"]
        if pinaw_delta < -float(self.pinaw_tolerance):
            return True
        if pinaw_delta > float(self.pinaw_tolerance):
            return False

        return candidate["coverage"] > (
            incumbent["coverage"] + float(self.coverage_tolerance)
        )

    def _cross_validated_score(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        experiment_ids: np.ndarray,
        params: dict[str, Any],
        lower_quantile: float,
        upper_quantile: float,
    ) -> dict[str, float]:
        model_config = dict(self.config.get("model", {}))
        fold_objectives: list[dict[str, float]] = []

        if not bool(params.get("bootstrap", True)):
            raise ValueError(
                "OOB tuning requires bootstrap=True for every candidate."
            )

        for train_index, validation_index in self._make_cv(groups).split(
            X, y, groups
        ):
            X_fold_train = X.iloc[train_index]
            X_fold_validation = X.iloc[validation_index]

            fold_train_frame = X_fold_train.copy()
            fold_train_frame["Experiment_ID"] = (
                experiment_ids[train_index]
            )
            fold_train_frame["_target"] = y[train_index]

            target_normalizer = GroupWiseTargetNormalizer.fit(
                frame=fold_train_frame,
                target_column="_target",
            )
            y_fold_train_normalized = (
                target_normalizer.transform_frame(
                    frame=fold_train_frame,
                    target_column="_target",
                )
            )

            estimator = self._build_estimator(params, model_config)
            estimator.fit(
                X_fold_train,
                y_fold_train_normalized,
            )

            validation_objective = self._score_denormalized_predictions(
                estimator=estimator,
                X_validation=X_fold_validation,
                y_validation=y[validation_index],
                validation_group_ids=groups[validation_index],
                validation_experiment_ids=(
                    experiment_ids[validation_index]
                ),
                target_normalizer=target_normalizer,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                oob_score=False,
            )

            oob_objective = self._score_denormalized_predictions(
                estimator=estimator,
                X_validation=X_fold_train,
                y_validation=y[train_index],
                validation_group_ids=groups[train_index],
                validation_experiment_ids=(
                    experiment_ids[train_index]
                ),
                target_normalizer=target_normalizer,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                oob_score=True,
            )

            fold_objectives.append(
                {
                    **validation_objective,
                    "oob_curve_distance": oob_objective[
                        "curve_distance"
                    ],
                    "oob_trend_error": oob_objective[
                        "trend_error"
                    ],
                    "oob_coverage": oob_objective["coverage"],
                    "oob_pinaw": oob_objective["pinaw"],
                    "oob_valid_fraction": oob_objective[
                        "valid_fraction"
                    ],
                }
            )

        return {
            key: float(
                np.mean([fold[key] for fold in fold_objectives])
            )
            for key in fold_objectives[0]
        }

    def _score_denormalized_predictions(
        self,
        estimator: RandomForestQuantileRegressor,
        X_validation: pd.DataFrame,
        y_validation: np.ndarray,
        validation_group_ids: np.ndarray,
        validation_experiment_ids: np.ndarray,
        target_normalizer: GroupWiseTargetNormalizer,
        lower_quantile: float,
        upper_quantile: float,
        *,
        oob_score: bool = False,
    ) -> dict[str, float]:
        predictions: dict[str, np.ndarray] = {}

        for name, quantile in (
            ("lower", lower_quantile),
            ("median", 0.5),
            ("upper", upper_quantile),
        ):
            normalized = np.asarray(
                estimator.predict(
                    X_validation,
                    quantiles=quantile,
                    oob_score=oob_score,
                )
            ).reshape(-1)

            predictions[name] = (
                target_normalizer.inverse_transform_values(
                    normalized,
                    validation_experiment_ids,
                )
            )

        finite_mask = (
            np.isfinite(np.asarray(y_validation).reshape(-1))
            & np.isfinite(predictions["lower"])
            & np.isfinite(predictions["median"])
            & np.isfinite(predictions["upper"])
        )

        valid_fraction = float(finite_mask.mean())
        if not finite_mask.any():
            raise RuntimeError(
                "No finite OOB predictions were produced. Increase "
                "n_estimators or verify that bootstrap=True."
            )

        y_valid = np.asarray(y_validation).reshape(-1)[finite_mask]
        lower_valid = predictions["lower"][finite_mask]
        median_valid = predictions["median"][finite_mask]
        upper_valid = predictions["upper"][finite_mask]
        group_valid = np.asarray(validation_group_ids).reshape(-1)[
            finite_mask
        ]
        order_valid = self._resolve_curve_order_values(
            X_validation
        )[finite_mask]

        group_metrics = self._group_metric_values(
            y_true=y_valid,
            y_lower=lower_valid,
            y_median=median_valid,
            y_upper=upper_valid,
            group_ids=group_valid,
            order_values=order_valid,
        )

        return {
            "curve_distance": group_metrics[
                "median_curve_distance_norm"
            ],
            "trend_error": group_metrics[
                "median_trend_shape_loss"
            ],
            "coverage": group_metrics[
                "median_group_coverage"
            ],
            "pinaw": group_metrics[
                "median_group_pinaw"
            ],
            "valid_fraction": valid_fraction,
        }

    def _group_metric_values(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_median: np.ndarray,
        y_upper: np.ndarray,
        group_ids: np.ndarray,
        order_values: np.ndarray | None = None,
    ) -> dict[str, float]:
        if order_values is None:
            order_values = np.arange(len(y_true), dtype=float)

        frame = pd.DataFrame({
            "group_id": np.asarray(group_ids).reshape(-1),
            "order_value": np.asarray(order_values).reshape(-1),
            "y_true": np.asarray(y_true).reshape(-1),
            "y_lower": np.asarray(y_lower).reshape(-1),
            "y_median": np.asarray(y_median).reshape(-1),
            "y_upper": np.asarray(y_upper).reshape(-1),
        })
        global_range = self._target_range(frame["y_true"].to_numpy())
        rows = []

        for _, group in frame.groupby("group_id", sort=False):
            group = group.sort_values("order_value", kind="stable")
            true = group["y_true"].to_numpy()
            lower = group["y_lower"].to_numpy()
            median = group["y_median"].to_numpy()
            upper = group["y_upper"].to_numpy()
            group_range = self._target_range(true)
            if np.ptp(true) == 0.0:
                group_range = global_range

            curve_distance = float(np.mean(np.abs(true - median)))
            curve_distance_norm = curve_distance / group_range

            if len(true) >= 2:
                true_diff = np.diff(true)
                pred_diff = np.diff(median)
                slope_error = float(
                    np.mean(np.abs(true_diff - pred_diff)) / group_range
                )
                direction_error = float(
                    np.mean(np.sign(true_diff) != np.sign(pred_diff))
                )
                trend_shape_loss = 0.5 * slope_error + 0.5 * direction_error
            else:
                slope_error = 0.0
                direction_error = 0.0
                trend_shape_loss = 0.0

            coverage = float(
                100.0 * ((true >= lower) & (true <= upper)).mean()
            )
            pinaw = float(np.mean(upper - lower) / group_range)
            mae = float(mean_absolute_error(true, median))
            rmse = float(mean_squared_error(true, median) ** 0.5)

            rows.append({
                "mae": mae,
                "rmse": rmse,
                "mae_norm": mae / group_range,
                "rmse_norm": rmse / group_range,
                "coverage": coverage,
                "pinaw": pinaw,
                "curve_distance": curve_distance,
                "curve_distance_norm": curve_distance_norm,
                "slope_error_norm": slope_error,
                "direction_error": direction_error,
                "trend_shape_loss": trend_shape_loss,
            })

        metric_df = pd.DataFrame(rows)
        result = {}
        for name in (
            "mae", "rmse", "mae_norm", "rmse_norm", "coverage",
            "pinaw", "curve_distance", "curve_distance_norm",
            "slope_error_norm", "direction_error", "trend_shape_loss",
        ):
            result[f"median_group_{name}"] = float(metric_df[name].median())
            result[f"p90_group_{name}"] = float(metric_df[name].quantile(0.90))

        # Backward-compatible names used by existing notebooks.
        result.update({
            "median_group_mae": result["median_group_mae"],
            "p90_group_mae": result["p90_group_mae"],
            "median_group_rmse": result["median_group_rmse"],
            "p90_group_rmse": result["p90_group_rmse"],
            "median_group_mae_norm": result["median_group_mae_norm"],
            "p90_group_mae_norm": result["p90_group_mae_norm"],
            "median_group_rmse_norm": result["median_group_rmse_norm"],
            "p90_group_rmse_norm": result["p90_group_rmse_norm"],
            "median_group_coverage": result["median_group_coverage"],
            "p10_group_coverage": float(metric_df["coverage"].quantile(0.10)),
            "p90_group_coverage": result["p90_group_coverage"],
            "median_group_pinaw": result["median_group_pinaw"],
            "p90_group_pinaw": result["p90_group_pinaw"],
            "median_curve_distance": result["median_group_curve_distance"],
            "p90_curve_distance": result["p90_group_curve_distance"],
            "median_curve_distance_norm": result["median_group_curve_distance_norm"],
            "p90_curve_distance_norm": result["p90_group_curve_distance_norm"],
            "median_trend_shape_loss": result["median_group_trend_shape_loss"],
            "p90_trend_shape_loss": result["p90_group_trend_shape_loss"],
            "median_distribution_coverage": result["median_group_coverage"],
            "p10_distribution_coverage": float(metric_df["coverage"].quantile(0.10)),
            "p90_distribution_coverage": result["p90_group_coverage"],
            "median_quantile_shape_error": result["median_group_curve_distance"],
            "p90_quantile_shape_error": result["p90_group_curve_distance"],
            "median_quantile_shape_error_norm": result["median_group_curve_distance_norm"],
            "p90_quantile_shape_error_norm": result["p90_group_curve_distance_norm"],
        })
        return result
    def _median_group_loss(
        self,
        group_metrics: dict[str, float],
    ) -> float:
        return float(group_metrics["median_curve_distance_norm"])
    def _interval_group_loss(
        self,
        group_metrics: dict[str, float],
    ) -> float:
        # Kept only for backward-compatible output. Search selection uses
        # _objective_is_better and therefore has no weighted score.
        return float(group_metrics["median_curve_distance_norm"])
    def _coverage_range_penalty(
        self,
        coverage_percent: float,
    ) -> float:
        coverage_percent = float(
            coverage_percent
        )

        if (
            self.coverage_lower_bound
            <= coverage_percent
            <= self.coverage_upper_bound
        ):
            return 0.0

        if coverage_percent < self.coverage_lower_bound:
            return (
                self.coverage_lower_bound
                - coverage_percent
            ) / max(
                self.coverage_lower_bound,
                1.0,
            )

        return (
            coverage_percent
            - self.coverage_upper_bound
        ) / max(
            self.coverage_upper_bound,
            1.0,
        )

    @staticmethod
    def _build_estimator(
        params: dict[str, Any],
        model_config: dict[str, Any],
    ) -> RandomForestQuantileRegressor:
        return RandomForestQuantileRegressor(
            n_estimators=int(
                params["n_estimators"]
            ),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(
                params["min_samples_leaf"]
            ),
            min_samples_split=int(
                params["min_samples_split"]
            ),
            max_features=params["max_features"],
            bootstrap=bool(
                params["bootstrap"]
            ),
            oob_score=bool(
                params["bootstrap"]
            ),
            max_samples=params.get(
                "max_samples",
                model_config.get("max_samples"),
            ),
            default_quantiles=0.5,
            random_state=int(
                model_config.get("random_state", 1100)
            ),
            n_jobs=int(
                model_config.get("n_jobs", -1)
            ),
        )

    def _evaluate_on_test(
        self,
        estimator,
        target_normalizer: GroupWiseTargetNormalizer,
        test_df: pd.DataFrame,
        target_column: str,
        lower_quantile: float,
        upper_quantile: float,
    ) -> dict[str, float]:
        feature_columns = list(
            self.config["features"]["input_columns"]
        )

        if "Experiment_ID" not in test_df.columns:
            raise KeyError(
                "Test data is missing required column: Experiment_ID"
            )

        y_true = test_df[target_column].to_numpy()
        X_test = test_df[feature_columns]

        y_lower_normalized = np.asarray(
            estimator.predict(
                X_test,
                quantiles=lower_quantile,
            )
        ).reshape(-1)
        y_median_normalized = np.asarray(
            estimator.predict(
                X_test,
                quantiles=0.5,
            )
        ).reshape(-1)
        y_upper_normalized = np.asarray(
            estimator.predict(
                X_test,
                quantiles=upper_quantile,
            )
        ).reshape(-1)

        y_lower = target_normalizer.inverse_transform_values(
            y_lower_normalized,
            test_df["Experiment_ID"],
        )
        y_median = target_normalizer.inverse_transform_values(
            y_median_normalized,
            test_df["Experiment_ID"],
        )
        y_upper = target_normalizer.inverse_transform_values(
            y_upper_normalized,
            test_df["Experiment_ID"],
        )

        metrics = self._metric_values(
            y_true=y_true,
            y_lower=y_lower,
            y_median=y_median,
            y_upper=y_upper,
        )
        group_metrics = self._group_metric_values(
            y_true=y_true,
            y_lower=y_lower,
            y_median=y_median,
            y_upper=y_upper,
            group_ids=test_df["Group_ID"].to_numpy(),
            order_values=self._resolve_curve_order_values(X_test),
        )

        return {
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "coverage_test": metrics["coverage"],
            "rmse_test": metrics["rmse"],
            "mae_test": metrics["mae"],
            "pinaw_test": metrics["pinaw"],
            **{
                f"{key}_test": value
                for key, value in group_metrics.items()
            },
            "r2_test": float(
                r2_score(
                    y_true,
                    y_median,
                )
            ),
        }

    def _metric_values(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_median: np.ndarray,
        y_upper: np.ndarray,
    ) -> dict[str, float]:
        y_true = np.asarray(
            y_true
        ).reshape(-1)
        y_lower = np.asarray(
            y_lower
        ).reshape(-1)
        y_median = np.asarray(
            y_median
        ).reshape(-1)
        y_upper = np.asarray(
            y_upper
        ).reshape(-1)

        coverage = float(
            100.0
            * (
                (y_true >= y_lower)
                & (y_true <= y_upper)
            ).mean()
        )
        rmse = float(
            mean_squared_error(
                y_true,
                y_median,
            )
            ** 0.5
        )
        mae = float(
            mean_absolute_error(
                y_true,
                y_median,
            )
        )
        pinaw = float(
            np.mean(y_upper - y_lower)
            / self._target_range(y_true)
        )

        return {
            "coverage": coverage,
            "rmse": rmse,
            "mae": mae,
            "pinaw": pinaw,
        }

    def _add_final_scores(
        self,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        results_df = results_df.copy()
        target_coverage = (
            results_df["upper_quantile"]
            - results_df["lower_quantile"]
        ) * 100.0
        results_df["coverage_error_test"] = (
            results_df["coverage_test"] - target_coverage
        ).abs()
        results_df["coverage_shortfall_test"] = (
            float(self.coverage_target)
            - results_df["median_group_coverage_test"]
        ).clip(lower=0.0)

        # Lower is better. This is a display column only. Model selection was
        # hierarchical and did not use a weighted sum.
        results_df["selection_score"] = results_df[
            "median_curve_distance_norm_test"
        ]

        preferred = [
            "target_axis", "is_main", "is_secondary",
            "selection_score", "coverage_test", "rmse_test",
            "mae_test", "pinaw_test", "r2_test",
            "median_curve_distance_norm_test",
            "p90_curve_distance_norm_test",
            "median_trend_shape_loss_test",
            "p90_trend_shape_loss_test",
            "median_group_coverage_test",
            "p10_group_coverage_test",
            "median_group_pinaw_test",
            "p90_group_pinaw_test",
            "coverage_shortfall_test",
            "std_test_group", "std_train_group",
            "coverage_error_test", "lower_quantile", "upper_quantile",
            "best_lower_quantile", "best_upper_quantile",
            "split_index", "split_name", "split_rank",
            "train_rows", "test_rows", "train_groups", "test_groups",
            "cv_best_score", "cv_curve_distance", "cv_trend_error",
            "cv_coverage", "cv_pinaw",
            "cv_oob_curve_distance", "cv_oob_trend_error",
            "cv_oob_coverage", "cv_oob_pinaw",
            "cv_oob_valid_fraction", "tuning_trial_count",
            "greedy_trial_count",
        ]
        ordered = [column for column in preferred if column in results_df.columns]
        remaining = [column for column in results_df.columns if column not in ordered]
        return results_df[ordered + remaining].sort_values(
            ["target_axis", "selection_score"]
        )
    def _best_split_index(
        self,
        split_metadata_df: pd.DataFrame,
        axis: str,
    ) -> int:
        rank_column = f"qrf_rank_{axis}"

        if rank_column not in split_metadata_df.columns:
            raise KeyError(
                "Split metadata is missing rank column "
                f"{rank_column!r}. Run split ranking first."
            )

        ranked_df = split_metadata_df[
            [
                "split_index",
                rank_column,
            ]
        ].dropna()

        ranked_df[rank_column] = pd.to_numeric(
            ranked_df[rank_column],
            errors="raise",
        ).astype(int)
        ranked_df["split_index"] = pd.to_numeric(
            ranked_df["split_index"],
            errors="raise",
        ).astype(int)

        matches = ranked_df[
            ranked_df[rank_column].eq(
                int(self.rank_value)
            )
        ]

        if matches.empty:
            raise ValueError(
                f"No split found with {rank_column}={self.rank_value}."
            )

        return int(
            matches.sort_values(
                "split_index"
            ).iloc[0]["split_index"]
        )

    def _make_cv(
        self,
        groups: np.ndarray,
    ):
        unique_groups = np.unique(groups)
        n_splits = min(
            int(self.cv_splits),
            len(unique_groups),
        )

        if n_splits >= 2:
            return GroupKFold(
                n_splits=n_splits
            )

        return KFold(
            n_splits=2,
            shuffle=True,
            random_state=int(
                self.config.get(
                    "model",
                    {},
                ).get("random_state", 1100)
            ),
        )

    def _normalised_weights(self) -> dict[str, float]:
        raise RuntimeError(
            "Weighted scoring was removed. Selection now uses a "
            "hierarchical curve/coverage/PINAW objective."
        )
    def _parameter_grid(
        self,
    ) -> dict[str, list[Any]]:
        param_grid = (
            self.param_grid
            if self.param_grid is not None
            else DEFAULT_PARAM_GRID
        )

        if not param_grid:
            raise ValueError(
                "param_grid cannot be empty."
            )

        validated_grid = {}

        for param_name, values in param_grid.items():
            values = list(values)

            if not values:
                raise ValueError(
                    f"param_grid[{param_name!r}] cannot be empty."
                )

            validated_grid[param_name] = values

        required_params = {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "bootstrap",
        }
        missing_params = required_params.difference(
            validated_grid
        )

        if missing_params:
            raise KeyError(
                "param_grid is missing required parameters: "
                f"{sorted(missing_params)}"
            )

        return validated_grid

    @staticmethod
    def _parameter_combinations(
        param_grid: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        parameter_names = list(
            param_grid.keys()
        )

        return [
            dict(
                zip(
                    parameter_names,
                    values,
                )
            )
            for values in product(
                *[
                    param_grid[
                        parameter_name
                    ]
                    for parameter_name in parameter_names
                ]
            )
        ]

    def _initial_params(
        self,
        model_config: dict[str, Any],
        param_grid: dict[str, list[Any]],
    ) -> dict[str, Any]:
        initial = {}
        for param_name, values in param_grid.items():
            configured = model_config.get(param_name)
            initial[param_name] = (
                configured if configured in values else values[0]
            )
        return initial
    def _quantile_grid(
        self,
    ) -> list[tuple[float, float]]:
        quantile_grid = (
            self.quantile_grid
            if self.quantile_grid is not None
            else DEFAULT_QUANTILE_GRID
        )

        if not quantile_grid:
            raise ValueError(
                "quantile_grid cannot be empty."
            )

        validated_grid = []

        for lower_quantile, upper_quantile in quantile_grid:
            lower_quantile = float(
                lower_quantile
            )
            upper_quantile = float(
                upper_quantile
            )

            if not (
                0.0
                <= lower_quantile
                < upper_quantile
                <= 1.0
            ):
                raise ValueError(
                    "Each quantile pair must satisfy "
                    "0 <= lower < upper <= 1. "
                    f"Received: "
                    f"({lower_quantile}, {upper_quantile})"
                )

            validated_grid.append(
                (
                    lower_quantile,
                    upper_quantile,
                )
            )

        return validated_grid

    @staticmethod
    def _mean_group_target_std(
        frame: pd.DataFrame,
        target_column: str,
    ) -> float:
        group_std = (
            frame.groupby("Group_ID")[target_column]
            .std()
            .fillna(0.0)
        )

        return float(
            group_std.mean()
        )

    @staticmethod
    def _target_range(
        y_true: np.ndarray,
    ) -> float:
        value_range = float(
            np.max(y_true)
            - np.min(y_true)
        )

        if np.isclose(
            value_range,
            0.0,
        ):
            return 1.0

        return value_range

    @staticmethod
    def _min_max(
        series: pd.Series,
    ) -> pd.Series:
        values = pd.to_numeric(
            series,
            errors="raise",
        ).astype(float)
        value_range = float(
            values.max()
            - values.min()
        )

        if np.isclose(
            value_range,
            0.0,
        ):
            return pd.Series(
                0.0,
                index=values.index,
                dtype=float,
            )

        return (
            values
            - float(values.min())
        ) / value_range

    @staticmethod
    def _flatten_best_params(
        best_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            f"best_{key}": value
            for key, value in best_params.items()
        }

    @staticmethod
    def _resolve_path(
        project_root: Path,
        value: str | Path,
    ) -> Path:
        path = Path(value)

        if path.is_absolute():
            return path

        return project_root / path
