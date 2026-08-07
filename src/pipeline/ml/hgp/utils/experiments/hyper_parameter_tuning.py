from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.pipeline.ml.hgp.utils.experiments.geometry_data_preprocessor import (
    load_bending_setups,
    load_selected_geometry_source,
)
from src.pipeline.ml.hgp.utils.experiments.hgp_model_trainer import (
    HGPPredictions,
    train_and_predict,
)


TARGET_COLUMNS_BY_AXIS = {
    "main": "Main-axis [mm]",
    "secondary": "Secondary-axis [mm]",
}

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "mean_gp_alpha": [1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
    "mean_kernel": ["matern_1.5", "matern_2.5", "rbf"],
    "initial_length_scale": [0.7, 1.5, 3.0, 6.0],
    "constant_value": [0.3, 1.0, 3.0],
    "noise_kernel": ["rbf", "matern_2.5"],
    "noise_initial_length_scale": [3.0, 6.0, 10.0],
    "noise_gp_alpha": [1e-3, 1e-2, 5e-2, 1e-1],
    "noise_variance_floor": [1e-3, 1e-2, 5e-2],
    "interval_scale": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
}


@dataclass
class HyperParameterTuning:
    """Tune HGP hyperparameters for the best ranked split of each axis.

    The class mirrors the QRF tuner interface:

        tuner = HyperParameterTuning(config=config, ...)
        results_df = tuner.run()

    Hyperparameters are selected only with GroupKFold on the training part of
    each selected split. The test part is evaluated after selection.

    The output parquet uses one stable path and is overwritten on every run.
    """

    config: dict[str, Any]
    param_grid: dict[str, list[Any]] | None = None
    cv_splits: int = 5
    greedy_patience: int = 2
    max_coordinate_passes: int = 4
    coverage_target: float | None = None
    minimum_coverage: float = 75.0
    curve_distance_tolerance: float = 0.01
    trend_tolerance: float = 0.01
    coverage_tolerance: float = 1.0
    pinaw_tolerance: float = 0.01
    rmse_tolerance: float = 0.01
    rank_value: int = 1
    group_column: str = "Group_ID"
    curve_order_column: str = "Angle[degree]ORDistance[mm]"
    output_filename: str = "hgp_hyperparameter_tuning_results.parquet"

    def run(self) -> pd.DataFrame:
        project_root = Path(self.config["_project_root"]).resolve()

        if not 0.0 <= float(self.minimum_coverage) <= 100.0:
            raise ValueError(
                "minimum_coverage must be between 0 and 100."
            )

        geometry_df, geometry_source, geometry_path = self._load_geometry(
            project_root
        )
        split_metadata_df = pd.read_parquet(
            self._resolve_path(
                project_root,
                self.config["paths"]["split_metadata"],
            )
        )

        feature_columns = list(
            self.config["features"]["input_columns"]
        )
        if self.group_column in feature_columns:
            raise ValueError(
                f"{self.group_column!r} must not be included in "
                "config['features']['input_columns']."
            )

        target_columns = dict(
            self.config["features"].get(
                "target_columns",
                TARGET_COLUMNS_BY_AXIS,
            )
        )
        aggregation_columns = list(
            self.config["features"].get(
                "aggregation_columns",
                [self.group_column, self.curve_order_column],
            )
        )

        rows: list[dict[str, Any]] = []

        for axis in ("main", "secondary"):
            target_column = target_columns.get(axis)
            if target_column is None:
                raise KeyError(
                    f"No target column configured for axis={axis!r}."
                )

            split_row = self._best_split_row(
                split_metadata_df=split_metadata_df,
                axis=axis,
            )
            train_df, test_df = self._split_geometry(
                geometry_df=geometry_df,
                split_row=split_row,
            )

            axis_model_config = self._axis_model_config(
                geometry_source=geometry_source,
                axis=axis,
            )

            tuning_result = self._tune_axis(
                axis=axis,
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                aggregation_columns=aggregation_columns,
                target_column=target_column,
                base_model_config=axis_model_config,
            )

            best_params = tuning_result["best_params"]
            best_config = tuning_result["best_config"]
            cv_objective = tuning_result["best_objective"]
            test_objective = tuning_result["test_objective"]

            rows.append(
                {
                    "geometry_source": geometry_source,
                    "geometry_path": str(geometry_path),
                    "target_axis": axis,
                    "target_column": target_column,
                    "split_index": int(split_row["split_index"]),
                    "split_name": str(
                        split_row.get(
                            "split_name",
                            f"split_{int(split_row['split_index'])}",
                        )
                    ),
                    "split_rank": int(split_row[f"qrf_rank_{axis}"]),
                    "source_rank_column": f"qrf_rank_{axis}",
                    "source_score": self._optional_float(
                        split_row.get(f"qrf_score_{axis}")
                    ),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "train_groups": int(
                        train_df[self.group_column].nunique()
                    ),
                    "test_groups": int(
                        test_df[self.group_column].nunique()
                    ),
                    "train_experiments": int(
                        train_df["Experiment_ID"].nunique()
                    ),
                    "test_experiments": int(
                        test_df["Experiment_ID"].nunique()
                    ),
                    "tuning_trial_count": int(
                        tuning_result["trial_count"]
                    ),
                    "coordinate_pass_count": int(
                        tuning_result["completed_passes"]
                    ),
                    **{
                        f"best_{key}": value
                        for key, value in best_params.items()
                    },
                    **{
                        f"cv_{key}": value
                        for key, value in cv_objective.items()
                    },
                    **{
                        f"test_{key}": value
                        for key, value in test_objective.items()
                    },
                    "minimum_coverage": float(self.minimum_coverage),
                    "cv_coverage_is_valid": bool(
                        cv_objective["coverage"]
                        >= float(self.minimum_coverage)
                    ),
                    "best_model_config": best_config,
                }
            )

            print(
                "Finished HGP tuning | "
                f"axis={axis} | "
                f"split={int(split_row['split_index'])} | "
                f"kernel={best_params['mean_kernel']} | "
                f"length_scale={best_params['initial_length_scale']} | "
                f"alpha={best_params['mean_gp_alpha']} | "
                f"cv_curve={cv_objective['curve_distance_norm']:.5f} | "
                f"cv_coverage={cv_objective['coverage']:.2f}% | "
                f"test_curve={test_objective['curve_distance_norm']:.5f} | "
                f"test_coverage={test_objective['coverage']:.2f}%"
            )

        results_df = self._order_columns(pd.DataFrame(rows))

        output_path = (
            project_root
            / "src"
            / "pipeline"
            / "ml"
            / "hgp"
            / "data"
            / self.output_filename
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stable path: pandas overwrites the previous parquet file.
        results_df.to_parquet(output_path, index=False)

        print(f"Saved HGP tuning summary to {output_path}")
        return results_df

    def _load_geometry(
        self,
        project_root: Path,
    ) -> tuple[pd.DataFrame, str, Path]:
        paths = self.config["paths"]
        data_config = dict(self.config["data"])

        bending_setups_df = load_bending_setups(
            path=self._resolve_path(
                project_root,
                paths["bending_setups"],
            ),
            excluded_experiments=data_config.get(
                "excluded_experiments",
                [],
            ),
        )

        return load_selected_geometry_source(
            project_root=project_root,
            paths_config=paths,
            data_config=data_config,
            bending_setups_df=bending_setups_df,
        )

    def _axis_model_config(
        self,
        *,
        geometry_source: str,
        axis: str,
    ) -> dict[str, Any]:
        model_section = dict(self.config.get("model", {}))

        merged = {
            key: value
            for key, value in model_section.items()
            if key not in {"main", "secondary", geometry_source}
        }

        source_config = model_section.get(geometry_source)
        if isinstance(source_config, dict):
            source_shared = {
                key: value
                for key, value in source_config.items()
                if key not in {"main", "secondary"}
            }
            merged = self._deep_merge_dicts(merged, source_shared)

            source_axis = source_config.get(axis)
            if isinstance(source_axis, dict):
                merged = self._deep_merge_dicts(merged, source_axis)

        axis_config = model_section.get(axis)
        if isinstance(axis_config, dict):
            merged = self._deep_merge_dicts(merged, axis_config)

        return merged

    def _best_split_row(
        self,
        *,
        split_metadata_df: pd.DataFrame,
        axis: str,
    ) -> pd.Series:
        rank_column = f"qrf_rank_{axis}"

        required_columns = {
            "split_index",
            "train_experiment_ids",
            "test_experiment_ids",
            rank_column,
        }
        missing = required_columns.difference(split_metadata_df.columns)
        if missing:
            raise KeyError(
                "Split metadata is missing columns: "
                f"{sorted(missing)}"
            )

        ranked_df = split_metadata_df.dropna(
            subset=["split_index", rank_column]
        ).copy()
        ranked_df[rank_column] = pd.to_numeric(
            ranked_df[rank_column],
            errors="raise",
        ).astype(int)
        ranked_df["split_index"] = pd.to_numeric(
            ranked_df["split_index"],
            errors="raise",
        ).astype(int)

        matches = ranked_df[
            ranked_df[rank_column].eq(int(self.rank_value))
        ].sort_values("split_index")

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one split with {rank_column}="
                f"{self.rank_value}, found {len(matches)}."
            )

        return matches.iloc[0]

    def _split_geometry(
        self,
        *,
        geometry_df: pd.DataFrame,
        split_row: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_ids = set(
            self._decode_id_list(split_row["train_experiment_ids"])
        )
        test_ids = set(
            self._decode_id_list(split_row["test_experiment_ids"])
        )

        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(
                "Train/test experiment leakage detected: "
                f"{sorted(overlap)}"
            )

        experiment_values = pd.to_numeric(
            geometry_df["Experiment_ID"],
            errors="raise",
        ).astype(int)

        train_df = geometry_df[
            experiment_values.isin(train_ids)
        ].copy()
        test_df = geometry_df[
            experiment_values.isin(test_ids)
        ].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                "The selected split produced an empty train or test frame."
            )

        train_groups = set(train_df[self.group_column].unique())
        test_groups = set(test_df[self.group_column].unique())
        group_overlap = train_groups.intersection(test_groups)
        if group_overlap:
            raise ValueError(
                "Train/test Group_ID leakage detected: "
                f"{sorted(group_overlap)}"
            )

        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _tune_axis(
        self,
        *,
        axis: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str],
        aggregation_columns: list[str],
        target_column: str,
        base_model_config: dict[str, Any],
    ) -> dict[str, Any]:
        grid = self._validated_param_grid(axis=axis)
        best_params = self._initial_params(
            grid=grid,
            base_model_config=base_model_config,
        )
        best_objective = self._cross_validated_objective(
            train_df=train_df,
            feature_columns=feature_columns,
            aggregation_columns=aggregation_columns,
            target_column=target_column,
            params=best_params,
            base_model_config=base_model_config,
        )

        trial_count = 1
        stale_passes = 0
        completed_passes = 0
        preferred_search_order = [
            "mean_gp_alpha",
            "mean_kernel",
            "initial_length_scale",
            "constant_value",
            "noise_kernel",
            "noise_initial_length_scale",
            "noise_gp_alpha",
            "noise_variance_floor",
            "interval_scale",
        ]
        search_dimensions = [
            name for name in preferred_search_order
            if name in grid
        ]

        for pass_index in range(max(1, int(self.max_coordinate_passes))):
            pass_improved = False

            for parameter_name in search_dimensions:
                dimension_best_params = dict(best_params)
                dimension_best_objective = dict(best_objective)

                for value in grid[parameter_name]:
                    if value == best_params[parameter_name]:
                        continue

                    candidate_params = dict(best_params)
                    candidate_params[parameter_name] = value
                    candidate_objective = self._cross_validated_objective(
                        train_df=train_df,
                        feature_columns=feature_columns,
                        aggregation_columns=aggregation_columns,
                        target_column=target_column,
                        params=candidate_params,
                        base_model_config=base_model_config,
                    )
                    trial_count += 1

                    if self._objective_is_better(
                        candidate_objective,
                        dimension_best_objective,
                    ):
                        dimension_best_params = candidate_params
                        dimension_best_objective = candidate_objective

                if self._objective_is_better(
                    dimension_best_objective,
                    best_objective,
                ):
                    best_params = dimension_best_params
                    best_objective = dimension_best_objective
                    pass_improved = True

            completed_passes = pass_index + 1
            if pass_improved:
                stale_passes = 0
            else:
                stale_passes += 1
                if stale_passes >= max(1, int(self.greedy_patience)):
                    break

        best_config = self._candidate_config(
            params=best_params,
            base_model_config=base_model_config,
        )
        _, test_predictions = train_and_predict(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_config=best_config,
            group_column=self.group_column,
            aggregation_columns=aggregation_columns,
        )
        test_objective = self._score_predictions(
            frame=test_df,
            target_column=target_column,
            predictions=test_predictions,
            base_model_config=base_model_config,
        )

        return {
            "best_params": best_params,
            "best_config": best_config,
            "best_objective": best_objective,
            "test_objective": test_objective,
            "trial_count": trial_count,
            "completed_passes": completed_passes,
        }

    def _cross_validated_objective(
        self,
        *,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        aggregation_columns: list[str],
        target_column: str,
        params: dict[str, Any],
        base_model_config: dict[str, Any],
    ) -> dict[str, float]:
        groups = train_df[self.group_column].to_numpy()
        n_splits = min(
            int(self.cv_splits),
            len(np.unique(groups)),
        )
        if n_splits < 2:
            raise ValueError(
                "HGP tuning requires at least two unique groups."
            )

        splitter = GroupKFold(n_splits=n_splits)
        candidate_config = self._candidate_config(
            params=params,
            base_model_config=base_model_config,
        )
        fold_metrics: list[dict[str, float]] = []

        for train_index, validation_index in splitter.split(
            train_df,
            groups=groups,
        ):
            fold_train_df = train_df.iloc[train_index].reset_index(drop=True)
            fold_validation_df = train_df.iloc[
                validation_index
            ].reset_index(drop=True)

            _, predictions = train_and_predict(
                train_df=fold_train_df,
                test_df=fold_validation_df,
                feature_columns=feature_columns,
                target_column=target_column,
                model_config=candidate_config,
                group_column=self.group_column,
                aggregation_columns=aggregation_columns,
            )
            fold_metrics.append(
                self._score_predictions(
                    frame=fold_validation_df,
                    target_column=target_column,
                    predictions=predictions,
                    base_model_config=base_model_config,
                )
            )

        return {
            key: float(np.mean([fold[key] for fold in fold_metrics]))
            for key in fold_metrics[0]
        }

    def _score_predictions(
        self,
        *,
        frame: pd.DataFrame,
        target_column: str,
        predictions: HGPPredictions,
        base_model_config: dict[str, Any],
    ) -> dict[str, float]:
        scoring_df = pd.DataFrame(
            {
                "group_id": frame[self.group_column].to_numpy(),
                "order": pd.to_numeric(
                    frame[self.curve_order_column],
                    errors="raise",
                ).to_numpy(dtype=float),
                "y_true": pd.to_numeric(
                    frame[target_column],
                    errors="raise",
                ).to_numpy(dtype=float),
                "y_mean": np.asarray(predictions.mean, dtype=float),
                "y_lower": np.asarray(predictions.lower, dtype=float),
                "y_upper": np.asarray(predictions.upper, dtype=float),
            }
        )

        global_range = self._safe_range(
            scoring_df["y_true"].to_numpy()
        )
        group_rows: list[dict[str, float]] = []

        for _, group in scoring_df.groupby("group_id", sort=False):
            group = group.sort_values("order", kind="stable")
            y_true = group["y_true"].to_numpy()
            y_mean = group["y_mean"].to_numpy()
            y_lower = group["y_lower"].to_numpy()
            y_upper = group["y_upper"].to_numpy()

            group_range = self._safe_range(y_true)
            if np.isclose(np.ptp(y_true), 0.0):
                group_range = global_range

            curve_distance_norm = float(
                np.mean(np.abs(y_true - y_mean)) / group_range
            )
            rmse_norm = float(
                np.sqrt(np.mean(np.square(y_true - y_mean)))
                / group_range
            )

            if len(y_true) >= 2:
                true_diff = np.diff(y_true)
                predicted_diff = np.diff(y_mean)
                slope_error = float(
                    np.mean(np.abs(true_diff - predicted_diff))
                    / group_range
                )
                direction_error = float(
                    np.mean(
                        np.sign(true_diff)
                        != np.sign(predicted_diff)
                    )
                )
                trend_shape_loss = (
                    0.5 * slope_error
                    + 0.5 * direction_error
                )
            else:
                trend_shape_loss = 0.0

            coverage = float(
                100.0
                * np.mean(
                    (y_true >= y_lower)
                    & (y_true <= y_upper)
                )
            )
            pinaw = float(
                np.mean(y_upper - y_lower)
                / group_range
            )

            group_rows.append(
                {
                    "curve_distance_norm": curve_distance_norm,
                    "trend_shape_loss": trend_shape_loss,
                    "coverage": coverage,
                    "pinaw": pinaw,
                    "rmse_norm": rmse_norm,
                }
            )

        metrics_df = pd.DataFrame(group_rows)
        coverage_target = self._coverage_target(
            base_model_config
        )
        median_coverage = float(
            metrics_df["coverage"].median()
        )

        return {
            "curve_distance_norm": float(
                metrics_df["curve_distance_norm"].median()
            ),
            "trend_shape_loss": float(
                metrics_df["trend_shape_loss"].median()
            ),
            "coverage": median_coverage,
            "coverage_shortfall": max(
                0.0,
                coverage_target - median_coverage,
            ),
            "coverage_error": abs(
                coverage_target - median_coverage
            ),
            "pinaw": float(metrics_df["pinaw"].median()),
            "rmse_norm": float(
                metrics_df["rmse_norm"].median()
            ),
            "p90_curve_distance_norm": float(
                metrics_df["curve_distance_norm"].quantile(0.90)
            ),
            "p10_coverage": float(
                metrics_df["coverage"].quantile(0.10)
            ),
            "p90_pinaw": float(
                metrics_df["pinaw"].quantile(0.90)
            ),
        }

    def _objective_is_better(
        self,
        candidate: dict[str, float],
        incumbent: dict[str, float],
    ) -> bool:
        """
        Compare candidates in this strict order:

        1. Median curve distance.
        2. Median trend-shape loss.
        3. Coverage, with preference for reaching minimum_coverage and then
           for being closer to coverage_target.
        4. PINAW.
        5. Normalized RMSE.

        A wider interval cannot compensate for a clearly worse median curve.
        """
        curve_delta = (
            candidate["curve_distance_norm"]
            - incumbent["curve_distance_norm"]
        )
        if curve_delta < -float(self.curve_distance_tolerance):
            return True
        if curve_delta > float(self.curve_distance_tolerance):
            return False

        trend_delta = (
            candidate["trend_shape_loss"]
            - incumbent["trend_shape_loss"]
        )
        if trend_delta < -float(self.trend_tolerance):
            return True
        if trend_delta > float(self.trend_tolerance):
            return False

        minimum_coverage = float(self.minimum_coverage)
        candidate_valid = candidate["coverage"] >= minimum_coverage
        incumbent_valid = incumbent["coverage"] >= minimum_coverage

        if candidate_valid and not incumbent_valid:
            return True
        if incumbent_valid and not candidate_valid:
            return False

        coverage_delta = (
            candidate["coverage_error"]
            - incumbent["coverage_error"]
        )
        if coverage_delta < -float(self.coverage_tolerance):
            return True
        if coverage_delta > float(self.coverage_tolerance):
            return False

        pinaw_delta = candidate["pinaw"] - incumbent["pinaw"]
        if pinaw_delta < -float(self.pinaw_tolerance):
            return True
        if pinaw_delta > float(self.pinaw_tolerance):
            return False

        rmse_delta = candidate["rmse_norm"] - incumbent["rmse_norm"]
        return rmse_delta < -float(self.rmse_tolerance)

    def _candidate_config(
        self,
        *,
        params: dict[str, Any],
        base_model_config: dict[str, Any],
    ) -> dict[str, Any]:
        config = self._deep_merge_dicts({}, base_model_config)

        mean_params = dict(config.get("mean_kernel_params", {}))
        mean_params["initial_length_scale"] = float(
            params["initial_length_scale"]
        )
        mean_params["constant_value"] = float(
            params["constant_value"]
        )

        noise_params = dict(config.get("noise_kernel_params", {}))
        if "noise_initial_length_scale" in params:
            noise_params["initial_length_scale"] = float(
                params["noise_initial_length_scale"]
            )

        config["mean_kernel"] = str(params["mean_kernel"])
        config["mean_gp_alpha"] = float(params["mean_gp_alpha"])
        config["mean_kernel_params"] = mean_params

        if "noise_kernel" in params:
            config["noise_kernel"] = str(params["noise_kernel"])
        if "noise_gp_alpha" in params:
            config["noise_gp_alpha"] = float(params["noise_gp_alpha"])
        config["noise_kernel_params"] = noise_params

        if "noise_variance_floor" in params:
            config["noise_variance_floor"] = float(
                params["noise_variance_floor"]
            )
        if "interval_scale" in params:
            config["interval_scale"] = float(params["interval_scale"])

        config["optimizer"] = None
        return config

    def _validated_param_grid(
        self,
        *,
        axis: str,
    ) -> dict[str, list[Any]]:
        if self.param_grid is not None:
            raw_grid = self.param_grid
        else:
            tuning = dict(self.config.get("tuning", {}))
            raw_grid = tuning.get(
                f"{axis}_param_grid",
                tuning.get("param_grid", DEFAULT_PARAM_GRID),
            )

        validated = {
            key: list(values)
            for key, values in raw_grid.items()
        }

        required = {
            "mean_gp_alpha",
            "mean_kernel",
            "initial_length_scale",
            "constant_value",
        }
        missing = required.difference(validated)
        if missing:
            raise KeyError(
                f"Tuning grid is missing: {sorted(missing)}"
            )

        for key, values in validated.items():
            if not values:
                raise ValueError(
                    f"Tuning parameter {key!r} cannot be empty."
                )

        return validated

    @staticmethod
    def _initial_params(
        *,
        grid: dict[str, list[Any]],
        base_model_config: dict[str, Any],
    ) -> dict[str, Any]:
        mean_params = dict(
            base_model_config.get("mean_kernel_params", {})
        )
        noise_params = dict(
            base_model_config.get("noise_kernel_params", {})
        )

        configured = {
            "mean_gp_alpha": base_model_config.get("mean_gp_alpha"),
            "mean_kernel": base_model_config.get("mean_kernel"),
            "initial_length_scale": mean_params.get(
                "initial_length_scale"
            ),
            "constant_value": mean_params.get("constant_value"),
            "noise_kernel": base_model_config.get("noise_kernel"),
            "noise_initial_length_scale": noise_params.get(
                "initial_length_scale"
            ),
            "noise_gp_alpha": base_model_config.get("noise_gp_alpha"),
            "noise_variance_floor": base_model_config.get(
                "noise_variance_floor"
            ),
            "interval_scale": base_model_config.get(
                "interval_scale",
                1.0,
            ),
        }

        return {
            key: (
                configured.get(key)
                if configured.get(key) in values
                else values[0]
            )
            for key, values in grid.items()
        }

    @staticmethod
    def _deep_merge_dicts(
        base: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in update.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = HyperParameterTuning._deep_merge_dicts(
                    merged[key],
                    value,
                )
            else:
                merged[key] = value
        return merged

    def _coverage_target(
        self,
        base_model_config: dict[str, Any],
    ) -> float:
        if self.coverage_target is not None:
            return float(self.coverage_target)

        return 100.0 * float(
            base_model_config.get("confidence_level", 0.90)
        )

    @staticmethod
    def _decode_id_list(value: Any) -> list[int]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple, set)):
            raise TypeError(
                "Experiment IDs must be list-like, received "
                f"{type(value).__name__}."
            )
        return sorted({int(item) for item in value})

    @staticmethod
    def _safe_range(values: np.ndarray) -> float:
        value_range = float(np.max(values) - np.min(values))
        return 1.0 if np.isclose(value_range, 0.0) else value_range

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _resolve_path(
        project_root: Path,
        value: str | Path,
    ) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    @staticmethod
    def _order_columns(results_df: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "geometry_source",
            "target_axis",
            "target_column",
            "best_mean_kernel",
            "best_mean_gp_alpha",
            "best_initial_length_scale",
            "best_constant_value",
            "best_noise_kernel",
            "best_noise_initial_length_scale",
            "best_noise_gp_alpha",
            "best_noise_variance_floor",
            "best_interval_scale",
            "cv_curve_distance_norm",
            "cv_trend_shape_loss",
            "minimum_coverage",
            "cv_coverage",
            "cv_coverage_is_valid",
            "cv_coverage_shortfall",
            "cv_pinaw",
            "cv_rmse_norm",
            "test_curve_distance_norm",
            "test_trend_shape_loss",
            "test_coverage",
            "test_pinaw",
            "test_rmse_norm",
            "tuning_trial_count",
            "coordinate_pass_count",
            "split_index",
            "split_name",
            "split_rank",
            "source_score",
            "train_rows",
            "test_rows",
            "train_groups",
            "test_groups",
            "train_experiments",
            "test_experiments",
            "geometry_path",
            "best_model_config",
        ]
        ordered = [
            column
            for column in preferred
            if column in results_df.columns
        ]
        remaining = [
            column
            for column in results_df.columns
            if column not in ordered
        ]
        return results_df[ordered + remaining]


# More explicit alias, while keeping the QRF-like class name above.
HGPHyperParameterTuning = HyperParameterTuning
