from __future__ import annotations

import ast
import itertools
import math
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


MEAN_ARD_PARAM_BY_FEATURE = {
    "Angle[degree]ORDistance[mm]": "mean_ls_angle",
    "Collet boost": "mean_ls_collet_boost",
    "Pressure-die distance": "mean_ls_pressure_die_distance",
    "Mandrel retraction timing": "mean_ls_mandrel_retraction_timing",
    "Pressure-die boost": "mean_ls_pressure_die_boost",
    "Clamp-die lateral position": "mean_ls_clamp_die_lateral_position",
    "Mandrel position": "mean_ls_mandrel_position",
}


@dataclass
class MainHyperParameterTuning:
    """
    Shared HGP hyperparameter-tuning engine.

    This class does not decide which axis should be tuned.
    MainMainHyperParameterTuning and SecondaryMainHyperParameterTuning
    call run_axis() explicitly.
    """

    config: dict[str, Any]
    param_grid: dict[str, list[Any]] | None = None

    cv_splits: int = 5

    max_candidates_main: int = 220
    max_candidates_secondary: int = 220

    search_random_state: int = 1100

    coverage_target: float | None = None
    minimum_coverage: float = 75.0

    curve_distance_tolerance: float = 0.01
    trend_tolerance: float = 0.01
    interval_shape_tolerance: float = 0.01
    coverage_tolerance: float = 1.0
    p10_coverage_tolerance: float = 1.0
    pinaw_tolerance: float = 0.01
    rmse_tolerance: float = 0.01

    rank_value: int = 1

    group_column: str = "Group_ID"

    curve_order_column: str = (
        "Angle[degree]ORDistance[mm]"
    )

    def __post_init__(self) -> None:

        tuning = dict(
            self.config.get("tuning", {})
        )

        configurable_names = (
            "cv_splits",
            "max_candidates_main",
            "max_candidates_secondary",
            "search_random_state",
            "coverage_target",
            "minimum_coverage",
            "curve_distance_tolerance",
            "trend_tolerance",
            "interval_shape_tolerance",
            "coverage_tolerance",
            "p10_coverage_tolerance",
            "pinaw_tolerance",
            "rmse_tolerance",
        )

        for name in configurable_names:
            if name in tuning:
                setattr(
                    self,
                    name,
                    tuning[name],
                )

    # =========================================================
    # Public axis-specific run
    # =========================================================

    def run_axis(
        self,
        axis: str,
    ) -> pd.DataFrame:

        if axis not in {
            "main",
            "secondary",
        }:
            raise ValueError(
                "axis must be 'main' or 'secondary'. "
                f"Received: {axis!r}"
            )

        project_root = Path(
            self.config["_project_root"]
        ).resolve()

        if not (
            0.0
            <= float(self.minimum_coverage)
            <= 100.0
        ):
            raise ValueError(
                "minimum_coverage must be "
                "between 0 and 100."
            )

        (
            geometry_df,
            geometry_source,
            geometry_path,
        ) = self._load_geometry(
            project_root
        )

        # Tuning requested only for real geometry.
        if geometry_source != "real":
            raise ValueError(
                "HGP tuning is restricted to "
                "geometry_source='real'. "
                f"Received {geometry_source!r}."
            )

        split_metadata_df = pd.read_parquet(
            self._resolve_path(
                project_root,
                self.config[
                    "paths"
                ]["split_metadata"],
            )
        )

        feature_columns = list(
            self.config[
                "features"
            ]["input_columns"]
        )

        if self.group_column in feature_columns:
            raise ValueError(
                f"{self.group_column!r} must not "
                "be included in input_columns."
            )

        if (
            self.curve_order_column
            not in feature_columns
        ):
            raise ValueError(
                "Angle-dependent HGP requires "
                f"{self.curve_order_column!r} "
                "inside input_columns."
            )

        target_columns = dict(
            self.config[
                "features"
            ].get(
                "target_columns",
                TARGET_COLUMNS_BY_AXIS,
            )
        )

        target_column = target_columns[
            axis
        ]

        aggregation_columns = list(
            self.config[
                "features"
            ].get(
                "aggregation_columns",
                [
                    self.group_column,
                    self.curve_order_column,
                ],
            )
        )

        objective_orders = (
            self._objective_orders(
                axis=axis
            )
        )

        split_row = self._best_split_row(
            split_metadata_df=(
                split_metadata_df
            ),
            axis=axis,
        )

        train_df, test_df = (
            self._split_geometry(
                geometry_df=geometry_df,
                split_row=split_row,
            )
        )

        base_model_config = (
            self._axis_model_config(
                geometry_source=(
                    geometry_source
                ),
                axis=axis,
            )
        )

        tuning_results = (
            self._tune_axis(
                axis=axis,
                train_df=train_df,
                test_df=test_df,
                feature_columns=(
                    feature_columns
                ),
                aggregation_columns=(
                    aggregation_columns
                ),
                target_column=target_column,
                base_model_config=(
                    base_model_config
                ),
                objective_orders=(
                    objective_orders
                ),
            )
        )

        rows = []

        for result in tuning_results:

            best_params = result[
                "best_params"
            ]

            cv_objective = result[
                "best_objective"
            ]

            test_objective = result[
                "test_objective"
            ]

            row = {
                "geometry_source": (
                    geometry_source
                ),
                "geometry_path": str(
                    geometry_path
                ),
                "target_axis": axis,
                "target_column": (
                    target_column
                ),
                "objective_strategy": (
                    result[
                        "objective_strategy"
                    ]
                ),
                "objective_order": (
                    result[
                        "objective_order"
                    ]
                ),
                "candidate_search": (
                    "joint_stratified_random"
                ),
                "candidate_pool_size": int(
                    result[
                        "candidate_pool_size"
                    ]
                ),
                "unique_test_winner_count": int(
                    result[
                        "unique_test_winner_count"
                    ]
                ),
                "split_index": int(
                    split_row[
                        "split_index"
                    ]
                ),
                "split_name": str(
                    split_row.get(
                        "split_name",
                        (
                            "split_"
                            f"{int(split_row['split_index'])}"
                        ),
                    )
                ),
                "split_rank": int(
                    split_row[
                        f"qrf_rank_{axis}"
                    ]
                ),
                "source_rank_column": (
                    f"qrf_rank_{axis}"
                ),
                "source_score": (
                    self._optional_float(
                        split_row.get(
                            f"qrf_score_{axis}"
                        )
                    )
                ),
                "train_rows": int(
                    len(train_df)
                ),
                "test_rows": int(
                    len(test_df)
                ),
                "train_groups": int(
                    train_df[
                        self.group_column
                    ].nunique()
                ),
                "test_groups": int(
                    test_df[
                        self.group_column
                    ].nunique()
                ),
                "train_experiments": int(
                    train_df[
                        "Experiment_ID"
                    ].nunique()
                ),
                "test_experiments": int(
                    test_df[
                        "Experiment_ID"
                    ].nunique()
                ),
                **{
                    f"best_{key}": value
                    for key, value
                    in best_params.items()
                },
                **{
                    f"cv_{key}": value
                    for key, value
                    in cv_objective.items()
                },
                **{
                    f"test_{key}": value
                    for key, value
                    in test_objective.items()
                },
                "minimum_coverage": float(
                    self.minimum_coverage
                ),
                "cv_coverage_is_valid": bool(
                    cv_objective[
                        "coverage"
                    ]
                    >= float(
                        self.minimum_coverage
                    )
                ),
                "best_model_config": (
                    result[
                        "best_config"
                    ]
                ),
            }

            rows.append(row)

        results_df = pd.DataFrame(
            rows
        )

        return self._order_columns(
            results_df
        )

    # =========================================================
    # Data
    # =========================================================

    def _load_geometry(
        self,
        project_root: Path,
    ) -> tuple[
        pd.DataFrame,
        str,
        Path,
    ]:

        paths = self.config[
            "paths"
        ]

        data_config = dict(
            self.config["data"]
        )

        bending_setups_df = (
            load_bending_setups(
                path=self._resolve_path(
                    project_root,
                    paths[
                        "bending_setups"
                    ],
                ),
                excluded_experiments=(
                    data_config.get(
                        "excluded_experiments",
                        [],
                    )
                ),
            )
        )

        return (
            load_selected_geometry_source(
                project_root=(
                    project_root
                ),
                paths_config=paths,
                data_config=(
                    data_config
                ),
                bending_setups_df=(
                    bending_setups_df
                ),
            )
        )

    # =========================================================
    # Model configuration
    # =========================================================

    def _axis_model_config(
        self,
        *,
        geometry_source: str,
        axis: str,
    ) -> dict[str, Any]:

        model_section = dict(
            self.config.get(
                "model",
                {},
            )
        )

        merged = {
            key: value
            for key, value
            in model_section.items()
            if key
            not in {
                "main",
                "secondary",
                geometry_source,
            }
        }

        source_config = (
            model_section.get(
                geometry_source
            )
        )

        if isinstance(
            source_config,
            dict,
        ):

            source_shared = {
                key: value
                for key, value
                in source_config.items()
                if key
                not in {
                    "main",
                    "secondary",
                }
            }

            merged = (
                self._deep_merge_dicts(
                    merged,
                    source_shared,
                )
            )

            source_axis = (
                source_config.get(
                    axis
                )
            )

            if isinstance(
                source_axis,
                dict,
            ):
                merged = (
                    self._deep_merge_dicts(
                        merged,
                        source_axis,
                    )
                )

        axis_config = (
            model_section.get(
                axis
            )
        )

        if isinstance(
            axis_config,
            dict,
        ):
            merged = (
                self._deep_merge_dicts(
                    merged,
                    axis_config,
                )
            )

        return merged

    # =========================================================
    # Split selection
    # =========================================================

    def _best_split_row(
        self,
        *,
        split_metadata_df: pd.DataFrame,
        axis: str,
    ) -> pd.Series:

        rank_column = (
            f"qrf_rank_{axis}"
        )

        required_columns = {
            "split_index",
            "train_experiment_ids",
            "test_experiment_ids",
            rank_column,
        }

        missing = (
            required_columns.difference(
                split_metadata_df.columns
            )
        )

        if missing:
            raise KeyError(
                "Split metadata is missing: "
                f"{sorted(missing)}"
            )

        ranked_df = (
            split_metadata_df
            .dropna(
                subset=[
                    "split_index",
                    rank_column,
                ]
            )
            .copy()
        )

        ranked_df[
            rank_column
        ] = pd.to_numeric(
            ranked_df[
                rank_column
            ],
            errors="raise",
        ).astype(int)

        ranked_df[
            "split_index"
        ] = pd.to_numeric(
            ranked_df[
                "split_index"
            ],
            errors="raise",
        ).astype(int)

        matches = ranked_df[
            ranked_df[
                rank_column
            ].eq(
                int(
                    self.rank_value
                )
            )
        ].sort_values(
            "split_index"
        )

        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one "
                f"{rank_column}="
                f"{self.rank_value}, "
                f"found {len(matches)}."
            )

        return matches.iloc[0]

    def _split_geometry(
        self,
        *,
        geometry_df: pd.DataFrame,
        split_row: pd.Series,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        train_ids = set(
            self._decode_id_list(
                split_row[
                    "train_experiment_ids"
                ]
            )
        )

        test_ids = set(
            self._decode_id_list(
                split_row[
                    "test_experiment_ids"
                ]
            )
        )

        overlap = (
            train_ids.intersection(
                test_ids
            )
        )

        if overlap:
            raise ValueError(
                "Train/test experiment "
                "leakage detected: "
                f"{sorted(overlap)}"
            )

        experiment_values = (
            pd.to_numeric(
                geometry_df[
                    "Experiment_ID"
                ],
                errors="raise",
            )
            .astype(int)
        )

        train_df = geometry_df[
            experiment_values.isin(
                train_ids
            )
        ].copy()

        test_df = geometry_df[
            experiment_values.isin(
                test_ids
            )
        ].copy()

        if (
            train_df.empty
            or test_df.empty
        ):
            raise ValueError(
                "Selected split produced "
                "an empty train/test frame."
            )

        train_groups = set(
            train_df[
                self.group_column
            ].unique()
        )

        test_groups = set(
            test_df[
                self.group_column
            ].unique()
        )

        group_overlap = (
            train_groups.intersection(
                test_groups
            )
        )

        if group_overlap:
            raise ValueError(
                "Train/test Group_ID "
                "leakage detected: "
                f"{sorted(group_overlap)}"
            )

        return (
            train_df.reset_index(
                drop=True
            ),
            test_df.reset_index(
                drop=True
            ),
        )

    # =========================================================
    # Joint candidate tuning
    # =========================================================

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
        objective_orders: dict[
            str,
            list[str],
        ],
    ) -> list[dict[str, Any]]:

        grid = (
            self._validated_param_grid(
                axis=axis
            )
        )

        initial_params = (
            self._initial_params(
                grid=grid,
                base_model_config=(
                    base_model_config
                ),
            )
        )

        max_candidates = (
            int(
                self.max_candidates_main
            )
            if axis == "main"
            else int(
                self.max_candidates_secondary
            )
        )

        random_state = (
            int(
                self.search_random_state
            )
            + (
                0
                if axis == "main"
                else 10000
            )
        )

        candidates = (
            self._generate_joint_candidates(
                grid=grid,
                initial_params=(
                    initial_params
                ),
                max_candidates=(
                    max_candidates
                ),
                random_state=(
                    random_state
                ),
            )
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Tuning HGP {axis.upper()}"
        )

        print(
            f"Candidates: "
            f"{len(candidates)}"
        )

        print(
            f"Objective strategies: "
            f"{list(objective_orders)}"
        )

        print(
            "=" * 80
        )

        evaluated = []

        for (
            candidate_index,
            params,
        ) in enumerate(
            candidates,
            start=1,
        ):

            objective = (
                self._cross_validated_objective(
                    train_df=train_df,
                    feature_columns=(
                        feature_columns
                    ),
                    aggregation_columns=(
                        aggregation_columns
                    ),
                    target_column=(
                        target_column
                    ),
                    params=params,
                    base_model_config=(
                        base_model_config
                    ),
                )
            )

            evaluated.append(
                {
                    "params": params,
                    "objective": objective,
                }
            )

            if (
                candidate_index == 1
                or candidate_index % 10 == 0
                or candidate_index
                == len(candidates)
            ):
                print(
                    f"{axis}: "
                    f"{candidate_index}/"
                    f"{len(candidates)}"
                )

        # -----------------------------------------------------
        # Same evaluated candidate pool, different objective
        # orders.
        # -----------------------------------------------------

        winners = {}

        for (
            strategy_name,
            order,
        ) in objective_orders.items():

            best = evaluated[0]

            for candidate in (
                evaluated[1:]
            ):

                if (
                    self
                    ._objective_is_better_by_order(
                        candidate=(
                            candidate[
                                "objective"
                            ]
                        ),
                        incumbent=(
                            best[
                                "objective"
                            ]
                        ),
                        order=order,
                    )
                ):
                    best = candidate

            winners[
                strategy_name
            ] = best

        # -----------------------------------------------------
        # Evaluate only unique winners on the held-out test.
        # -----------------------------------------------------

        test_cache = {}
        config_cache = {}

        for winner in (
            winners.values()
        ):

            key = self._params_key(
                winner["params"]
            )

            if key in test_cache:
                continue

            best_config = (
                self._candidate_config(
                    params=(
                        winner["params"]
                    ),
                    base_model_config=(
                        base_model_config
                    ),
                )
            )

            _, test_predictions = (
                train_and_predict(
                    train_df=train_df,
                    test_df=test_df,
                    feature_columns=(
                        feature_columns
                    ),
                    target_column=(
                        target_column
                    ),
                    model_config=(
                        best_config
                    ),
                    group_column=(
                        self.group_column
                    ),
                    aggregation_columns=(
                        aggregation_columns
                    ),
                )
            )

            test_cache[key] = (
                self._score_predictions(
                    frame=test_df,
                    target_column=(
                        target_column
                    ),
                    predictions=(
                        test_predictions
                    ),
                    base_model_config=(
                        base_model_config
                    ),
                )
            )

            config_cache[
                key
            ] = best_config

        results = []

        for (
            strategy_name,
            order,
        ) in objective_orders.items():

            winner = winners[
                strategy_name
            ]

            key = self._params_key(
                winner["params"]
            )

            results.append(
                {
                    "objective_strategy": (
                        strategy_name
                    ),
                    "objective_order": (
                        " > ".join(
                            order
                        )
                    ),
                    "best_params": (
                        winner[
                            "params"
                        ]
                    ),
                    "best_config": (
                        config_cache[key]
                    ),
                    "best_objective": (
                        winner[
                            "objective"
                        ]
                    ),
                    "test_objective": (
                        test_cache[key]
                    ),
                    "candidate_pool_size": (
                        len(candidates)
                    ),
                    "unique_test_winner_count": (
                        len(test_cache)
                    ),
                }
            )

            print(
                "\n"
                f"Winner [{strategy_name}]"
            )

            print(
                "CV curve:",
                winner[
                    "objective"
                ][
                    "curve_distance_norm"
                ],
            )

            print(
                "CV trend:",
                winner[
                    "objective"
                ][
                    "trend_shape_loss"
                ],
            )

            print(
                "CV coverage:",
                winner[
                    "objective"
                ]["coverage"],
            )

            print(
                "TEST curve:",
                test_cache[
                    key
                ][
                    "curve_distance_norm"
                ],
            )

            print(
                "TEST coverage:",
                test_cache[
                    key
                ]["coverage"],
            )

        return results

    # =========================================================
    # Candidate generation
    # =========================================================

    @staticmethod
    def _params_key(
        params: dict[str, Any],
    ) -> tuple[
        tuple[str, str],
        ...
    ]:

        return tuple(
            sorted(
                (
                    str(key),
                    repr(value),
                )
                for key, value
                in params.items()
            )
        )

    def _generate_joint_candidates(
        self,
        *,
        grid: dict[str, list[Any]],
        initial_params: dict[str, Any],
        max_candidates: int,
        random_state: int,
    ) -> list[dict[str, Any]]:

        if max_candidates < 1:
            raise ValueError(
                "max_candidates must be >= 1"
            )

        names = list(
            grid
        )

        total_combinations = (
            math.prod(
                len(
                    grid[name]
                )
                for name in names
            )
        )

        if (
            total_combinations
            <= max_candidates
        ):
            return [
                dict(
                    zip(
                        names,
                        values,
                    )
                )
                for values
                in itertools.product(
                    *(
                        grid[name]
                        for name in names
                    )
                )
            ]

        rng = (
            np.random.default_rng(
                random_state
            )
        )

        candidates = []
        seen = set()

        def add(
            params: dict[str, Any]
        ) -> None:

            key = self._params_key(
                params
            )

            if (
                key not in seen
                and len(candidates)
                < max_candidates
            ):
                seen.add(key)

                candidates.append(
                    dict(params)
                )

        # Always include current config.
        add(initial_params)

        remaining = (
            max_candidates
            - len(candidates)
        )

        if remaining <= 0:
            return candidates

        # Stratify every dimension.
        columns = {}

        for name in names:

            values = list(
                grid[name]
            )

            repeated = []

            while (
                len(repeated)
                < remaining * 2
            ):

                cycle = list(
                    values
                )

                rng.shuffle(
                    cycle
                )

                repeated.extend(
                    cycle
                )

            columns[
                name
            ] = repeated

        for i in range(
            remaining * 2
        ):

            add(
                {
                    name: (
                        columns[
                            name
                        ][i]
                    )
                    for name in names
                }
            )

            if (
                len(candidates)
                >= max_candidates
            ):
                return candidates

        attempts = 0

        while (
            len(candidates)
            < max_candidates
            and attempts
            < max_candidates * 100
        ):

            attempts += 1

            add(
                {
                    name: grid[
                        name
                    ][
                        int(
                            rng.integers(
                                0,
                                len(
                                    grid[
                                        name
                                    ]
                                ),
                            )
                        )
                    ]
                    for name in names
                }
            )

        return candidates

    # =========================================================
    # Cross-validation
    # =========================================================

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

        groups = train_df[
            self.group_column
        ].to_numpy()

        n_splits = min(
            int(
                self.cv_splits
            ),
            len(
                np.unique(
                    groups
                )
            ),
        )

        if n_splits < 2:
            raise ValueError(
                "HGP tuning requires "
                "at least two unique groups."
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        candidate_config = (
            self._candidate_config(
                params=params,
                base_model_config=(
                    base_model_config
                ),
            )
        )

        fold_metrics = []

        for (
            train_index,
            validation_index,
        ) in splitter.split(
            train_df,
            groups=groups,
        ):

            fold_train_df = (
                train_df.iloc[
                    train_index
                ]
                .reset_index(
                    drop=True
                )
            )

            fold_validation_df = (
                train_df.iloc[
                    validation_index
                ]
                .reset_index(
                    drop=True
                )
            )

            _, predictions = (
                train_and_predict(
                    train_df=(
                        fold_train_df
                    ),
                    test_df=(
                        fold_validation_df
                    ),
                    feature_columns=(
                        feature_columns
                    ),
                    target_column=(
                        target_column
                    ),
                    model_config=(
                        candidate_config
                    ),
                    group_column=(
                        self.group_column
                    ),
                    aggregation_columns=(
                        aggregation_columns
                    ),
                )
            )

            fold_metrics.append(
                self._score_predictions(
                    frame=(
                        fold_validation_df
                    ),
                    target_column=(
                        target_column
                    ),
                    predictions=(
                        predictions
                    ),
                    base_model_config=(
                        base_model_config
                    ),
                )
            )

        return {
            key: float(
                np.mean(
                    [
                        fold[key]
                        for fold
                        in fold_metrics
                    ]
                )
            )
            for key
            in fold_metrics[0]
        }

    # =========================================================
    # Scoring
    # =========================================================

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
                "group_id": frame[
                    self.group_column
                ].to_numpy(),
                "order": pd.to_numeric(
                    frame[
                        self.curve_order_column
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=float
                ),
                "y_true": pd.to_numeric(
                    frame[
                        target_column
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=float
                ),
                "y_mean": np.asarray(
                    predictions.mean,
                    dtype=float,
                ),
                "y_lower": np.asarray(
                    predictions.lower,
                    dtype=float,
                ),
                "y_upper": np.asarray(
                    predictions.upper,
                    dtype=float,
                ),
                "latent_std": np.asarray(
                    predictions.latent_std,
                    dtype=float,
                ),
                "aleatoric_std": np.asarray(
                    predictions.aleatoric_std,
                    dtype=float,
                ),
            }
        )

        scoring_df[
            "width"
        ] = (
            scoring_df[
                "y_upper"
            ]
            - scoring_df[
                "y_lower"
            ]
        )

        global_range = (
            self._safe_range(
                scoring_df[
                    "y_true"
                ].to_numpy()
            )
        )

        group_rows = []

        for (
            _,
            raw_group,
        ) in scoring_df.groupby(
            "group_id",
            sort=False,
        ):

            raw_group = (
                raw_group.sort_values(
                    "order",
                    kind="stable",
                )
            )

            raw_y_true = (
                raw_group[
                    "y_true"
                ].to_numpy()
            )

            raw_y_lower = (
                raw_group[
                    "y_lower"
                ].to_numpy()
            )

            raw_y_upper = (
                raw_group[
                    "y_upper"
                ].to_numpy()
            )

            coverage = float(
                100.0
                * np.mean(
                    (
                        raw_y_true
                        >= raw_y_lower
                    )
                    & (
                        raw_y_true
                        <= raw_y_upper
                    )
                )
            )

            angle_group = (
                raw_group
                .groupby(
                    "order",
                    as_index=False,
                    sort=True,
                )
                .agg(
                    y_true=(
                        "y_true",
                        "mean",
                    ),
                    y_mean=(
                        "y_mean",
                        "mean",
                    ),
                    y_lower=(
                        "y_lower",
                        "mean",
                    ),
                    y_upper=(
                        "y_upper",
                        "mean",
                    ),
                    width=(
                        "width",
                        "mean",
                    ),
                    latent_std=(
                        "latent_std",
                        "mean",
                    ),
                    aleatoric_std=(
                        "aleatoric_std",
                        "mean",
                    ),
                    repeat_count=(
                        "y_true",
                        "size",
                    ),
                    observed_std=(
                        "y_true",
                        lambda values: (
                            float(
                                np.std(
                                    values,
                                    ddof=1,
                                )
                            )
                            if len(values) >= 2
                            else np.nan
                        ),
                    ),
                )
            )

            y_true = (
                angle_group[
                    "y_true"
                ].to_numpy(
                    dtype=float
                )
            )

            y_mean = (
                angle_group[
                    "y_mean"
                ].to_numpy(
                    dtype=float
                )
            )

            predicted_width = (
                angle_group[
                    "width"
                ].to_numpy(
                    dtype=float
                )
            )

            group_range = (
                self._safe_range(
                    y_true
                )
            )

            if np.isclose(
                np.ptp(y_true),
                0.0,
            ):
                group_range = (
                    global_range
                )

            curve_distance_norm = float(
                np.mean(
                    np.abs(
                        y_true
                        - y_mean
                    )
                )
                / group_range
            )

            rmse_norm = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            y_true
                            - y_mean
                        )
                    )
                )
                / group_range
            )

            if len(y_true) >= 2:

                true_diff = (
                    np.diff(
                        y_true
                    )
                )

                predicted_diff = (
                    np.diff(
                        y_mean
                    )
                )

                slope_error = float(
                    np.mean(
                        np.abs(
                            true_diff
                            - predicted_diff
                        )
                    )
                    / group_range
                )

                direction_error = float(
                    np.mean(
                        np.sign(
                            true_diff
                        )
                        != np.sign(
                            predicted_diff
                        )
                    )
                )

                trend_shape_loss = (
                    0.5
                    * slope_error
                    + 0.5
                    * direction_error
                )

            else:
                trend_shape_loss = 0.0

            observed_stats = (
                angle_group.loc[
                    angle_group[
                        "repeat_count"
                    ] >= 2,
                    [
                        "order",
                        "observed_std",
                    ],
                ]
                .copy()
            )

            observed_stats = (
                observed_stats.rename(
                    columns={
                        "order": (
                            self.curve_order_column
                        ),
                        "observed_std": (
                            "std"
                        ),
                    }
                )
                .dropna(
                    subset=[
                        "std"
                    ]
                )
            )

            predicted_angle_width = (
                angle_group[
                    [
                        "order",
                        "width",
                    ]
                ]
                .rename(
                    columns={
                        "order": (
                            self.curve_order_column
                        ),
                        "width": (
                            "predicted_width"
                        ),
                    }
                )
            )

            interval_shape_loss = (
                self._interval_shape_loss(
                    observed_stats=(
                        observed_stats
                    ),
                    predicted_width_df=(
                        predicted_angle_width
                    ),
                )
            )

            pinaw = float(
                np.mean(
                    predicted_width
                )
                / group_range
            )

            width_mean = float(
                np.mean(
                    predicted_width
                )
            )

            if (
                len(predicted_width)
                >= 2
                and width_mean > 1e-12
            ):
                width_cv = float(
                    np.std(
                        predicted_width
                    )
                    / width_mean
                )
            else:
                width_cv = 0.0

            group_rows.append(
                {
                    "curve_distance_norm": (
                        curve_distance_norm
                    ),
                    "trend_shape_loss": (
                        trend_shape_loss
                    ),
                    "interval_shape_loss": (
                        interval_shape_loss
                    ),
                    "coverage": coverage,
                    "pinaw": pinaw,
                    "rmse_norm": rmse_norm,
                    "width_cv": width_cv,
                    "latent_std_norm": float(
                        angle_group[
                            "latent_std"
                        ].mean()
                        / group_range
                    ),
                    "aleatoric_std_norm": float(
                        angle_group[
                            "aleatoric_std"
                        ].mean()
                        / group_range
                    ),
                }
            )

        metrics_df = (
            pd.DataFrame(
                group_rows
            )
        )

        coverage_target = (
            self._coverage_target(
                base_model_config
            )
        )

        median_coverage = float(
            metrics_df[
                "coverage"
            ].median()
        )

        valid_shape = (
            metrics_df[
                "interval_shape_loss"
            ]
            .dropna()
        )

        median_interval_shape_loss = (
            float(
                valid_shape.median()
            )
            if not valid_shape.empty
            else 0.0
        )

        return {
            "curve_distance_norm": float(
                metrics_df[
                    "curve_distance_norm"
                ].median()
            ),
            "trend_shape_loss": float(
                metrics_df[
                    "trend_shape_loss"
                ].median()
            ),
            "interval_shape_loss": (
                median_interval_shape_loss
            ),
            "coverage_target": float(
                coverage_target
            ),
            "coverage": (
                median_coverage
            ),
            "coverage_shortfall": max(
                0.0,
                float(
                    self.minimum_coverage
                )
                - median_coverage,
            ),
            "coverage_error": abs(
                coverage_target
                - median_coverage
            ),
            "pinaw": float(
                metrics_df[
                    "pinaw"
                ].median()
            ),
            "rmse_norm": float(
                metrics_df[
                    "rmse_norm"
                ].median()
            ),
            "p90_curve_distance_norm": float(
                metrics_df[
                    "curve_distance_norm"
                ].quantile(
                    0.90
                )
            ),
            "p10_coverage": float(
                metrics_df[
                    "coverage"
                ].quantile(
                    0.10
                )
            ),
            "p90_pinaw": float(
                metrics_df[
                    "pinaw"
                ].quantile(
                    0.90
                )
            ),
            "median_width_cv": float(
                metrics_df[
                    "width_cv"
                ].median()
            ),
            "median_latent_std_norm": float(
                metrics_df[
                    "latent_std_norm"
                ].median()
            ),
            "median_aleatoric_std_norm": float(
                metrics_df[
                    "aleatoric_std_norm"
                ].median()
            ),
            "interval_shape_group_count": float(
                len(
                    valid_shape
                )
            ),
        }

    # =========================================================
    # Interval shape
    # =========================================================

    @staticmethod
    def _interval_shape_loss(
        *,
        observed_stats: pd.DataFrame,
        predicted_width_df: pd.DataFrame,
    ) -> float:

        if len(
            observed_stats
        ) < 2:
            return float(
                "nan"
            )

        observed = (
            observed_stats.rename(
                columns={
                    observed_stats.columns[
                        0
                    ]: "angle"
                }
            )
            .copy()
        )

        predicted = (
            predicted_width_df.rename(
                columns={
                    predicted_width_df.columns[
                        0
                    ]: "angle"
                }
            )
            .copy()
        )

        merged = (
            observed.merge(
                predicted,
                on="angle",
                how="inner",
            )
        )

        if len(merged) < 2:
            return float(
                "nan"
            )

        observed_std = (
            merged[
                "std"
            ].to_numpy(
                dtype=float
            )
        )

        predicted_width = (
            merged[
                "predicted_width"
            ].to_numpy(
                dtype=float
            )
        )

        if (
            np.ptp(
                observed_std
            )
            < 1e-12
        ):
            return 0.0

        observed_mean = float(
            np.mean(
                observed_std
            )
        )

        predicted_mean = float(
            np.mean(
                predicted_width
            )
        )

        if observed_mean <= 1e-12:
            return float(
                "nan"
            )

        if predicted_mean <= 1e-12:
            return 1.0

        observed_shape = (
            observed_std
            / observed_mean
        )

        predicted_shape = (
            predicted_width
            / predicted_mean
        )

        mae_shape = float(
            np.mean(
                np.abs(
                    observed_shape
                    - predicted_shape
                )
            )
        )

        observed_diff = (
            np.diff(
                observed_shape
            )
        )

        predicted_diff = (
            np.diff(
                predicted_shape
            )
        )

        direction_loss = float(
            np.mean(
                np.sign(
                    observed_diff
                )
                != np.sign(
                    predicted_diff
                )
            )
        )

        return (
            0.7
            * mae_shape
            + 0.3
            * direction_loss
        )

    # =========================================================
    # Objective strategies
    # =========================================================

    def _objective_orders(
        self,
        *,
        axis: str,
    ) -> dict[
        str,
        list[str],
    ]:

        tuning = dict(
            self.config.get(
                "tuning",
                {},
            )
        )

        objective_orders = (
            tuning.get(
                f"{axis}_objective_orders"
            )
        )

        if objective_orders is None:
            objective_orders = (
                tuning.get(
                    "objective_orders"
                )
            )

        if not objective_orders:
            raise ValueError(
                "No objective orders configured "
                f"for axis={axis!r}."
            )

        allowed_metrics = {
            "minimum_coverage",
            "coverage_error",
            "interval_shape_loss",
            "curve_distance_norm",
            "trend_shape_loss",
            "pinaw",
            "rmse_norm",
            "p10_coverage",
        }

        validated = {}

        for (
            strategy_name,
            order,
        ) in objective_orders.items():

            order = list(
                order
            )

            if not order:
                raise ValueError(
                    "Empty objective order: "
                    f"{strategy_name!r}"
                )

            unknown = (
                set(order)
                .difference(
                    allowed_metrics
                )
            )

            if unknown:
                raise ValueError(
                    "Unknown objective metrics "
                    f"for {strategy_name!r}: "
                    f"{sorted(unknown)}"
                )

            validated[
                str(
                    strategy_name
                )
            ] = order

        return validated

    def _objective_is_better_by_order(
        self,
        *,
        candidate: dict[str, float],
        incumbent: dict[str, float],
        order: list[str],
    ) -> bool:

        for metric in order:

            decision = (
                self._compare_metric(
                    metric=metric,
                    candidate=candidate,
                    incumbent=incumbent,
                )
            )

            if decision is not None:
                return decision

        return False

    def _compare_metric(
        self,
        *,
        metric: str,
        candidate: dict[str, float],
        incumbent: dict[str, float],
    ) -> bool | None:

        if metric == "minimum_coverage":

            minimum = float(
                self.minimum_coverage
            )

            cand_cov = float(
                candidate[
                    "coverage"
                ]
            )

            inc_cov = float(
                incumbent[
                    "coverage"
                ]
            )

            cand_valid = (
                cand_cov >= minimum
            )

            inc_valid = (
                inc_cov >= minimum
            )

            if (
                cand_valid
                != inc_valid
            ):
                return cand_valid

            if not cand_valid:

                cand_short = max(
                    0.0,
                    minimum
                    - cand_cov,
                )

                inc_short = max(
                    0.0,
                    minimum
                    - inc_cov,
                )

                delta = (
                    cand_short
                    - inc_short
                )

                tolerance = float(
                    self.coverage_tolerance
                )

                if delta < -tolerance:
                    return True

                if delta > tolerance:
                    return False

            return None

        if metric == "p10_coverage":

            delta = (
                float(
                    candidate[
                        metric
                    ]
                )
                - float(
                    incumbent[
                        metric
                    ]
                )
            )

            tolerance = float(
                self.p10_coverage_tolerance
            )

            # Larger p10 coverage is better.
            if delta > tolerance:
                return True

            if delta < -tolerance:
                return False

            return None

        tolerance_by_metric = {
            "curve_distance_norm": float(
                self.curve_distance_tolerance
            ),
            "trend_shape_loss": float(
                self.trend_tolerance
            ),
            "interval_shape_loss": float(
                self.interval_shape_tolerance
            ),
            "coverage_error": float(
                self.coverage_tolerance
            ),
            "pinaw": float(
                self.pinaw_tolerance
            ),
            "rmse_norm": float(
                self.rmse_tolerance
            ),
        }

        candidate_value = float(
            candidate[
                metric
            ]
        )

        incumbent_value = float(
            incumbent[
                metric
            ]
        )

        if np.isnan(
            candidate_value
        ):
            candidate_value = (
                np.inf
            )

        if np.isnan(
            incumbent_value
        ):
            incumbent_value = (
                np.inf
            )

        delta = (
            candidate_value
            - incumbent_value
        )

        tolerance = (
            tolerance_by_metric[
                metric
            ]
        )

        # All metrics here are losses:
        # smaller is better.
        if delta < -tolerance:
            return True

        if delta > tolerance:
            return False

        return None

    # =========================================================
    # Candidate config
    # =========================================================

    def _candidate_config(
        self,
        *,
        params: dict[str, Any],
        base_model_config: dict[str, Any],
    ) -> dict[str, Any]:

        config = (
            self._deep_merge_dicts(
                {},
                base_model_config,
            )
        )

        mean_params = dict(
            config.get(
                "mean_kernel_params",
                {},
            )
        )

        feature_length_scales = dict(
            mean_params.get(
                "feature_length_scales",
                {},
            )
        )

        fallback_length_scale = float(
            mean_params.get(
                "initial_length_scale",
                1.0,
            )
        )

        for (
            feature,
            parameter_name,
        ) in (
            MEAN_ARD_PARAM_BY_FEATURE.items()
        ):

            feature_length_scales[
                feature
            ] = float(
                params.get(
                    parameter_name,
                    feature_length_scales.get(
                        feature,
                        fallback_length_scale,
                    ),
                )
            )

        mean_params[
            "feature_length_scales"
        ] = feature_length_scales

        mean_params[
            "initial_length_scale"
        ] = fallback_length_scale

        mean_params[
            "constant_value"
        ] = float(
            params[
                "constant_value"
            ]
        )

        noise_params = dict(
            config.get(
                "noise_kernel_params",
                {},
            )
        )

        if (
            "noise_initial_length_scale"
            in params
        ):
            noise_params[
                "initial_length_scale"
            ] = float(
                params[
                    "noise_initial_length_scale"
                ]
            )

        config[
            "mean_kernel"
        ] = str(
            params[
                "mean_kernel"
            ]
        )

        config[
            "mean_gp_alpha"
        ] = float(
            params[
                "mean_gp_alpha"
            ]
        )

        config[
            "mean_kernel_params"
        ] = mean_params

        if "noise_kernel" in params:
            config[
                "noise_kernel"
            ] = str(
                params[
                    "noise_kernel"
                ]
            )

        if "noise_gp_alpha" in params:
            config[
                "noise_gp_alpha"
            ] = float(
                params[
                    "noise_gp_alpha"
                ]
            )

        config[
            "noise_kernel_params"
        ] = noise_params

        if (
            "noise_variance_floor"
            in params
        ):
            config[
                "noise_variance_floor"
            ] = float(
                params[
                    "noise_variance_floor"
                ]
            )

        config[
            "noise_min_repeat_count"
        ] = int(
            config.get(
                "noise_min_repeat_count",
                2,
            )
        )

        config[
            "optimizer"
        ] = None

        config[
            "ensemble_enabled"
        ] = False

        return config

    # =========================================================
    # Grid
    # =========================================================

    def _validated_param_grid(
        self,
        *,
        axis: str,
    ) -> dict[
        str,
        list[Any],
    ]:

        if self.param_grid is not None:
            raw_grid = (
                self.param_grid
            )

        else:

            tuning = dict(
                self.config.get(
                    "tuning",
                    {},
                )
            )

            raw_grid = (
                tuning.get(
                    f"{axis}_param_grid"
                )
            )

            if raw_grid is None:
                raise KeyError(
                    "No parameter grid configured "
                    f"for axis={axis!r}."
                )

        validated = {
            key: list(
                values
            )
            for key, values
            in raw_grid.items()
        }

        validated.pop(
            "interval_scale",
            None,
        )

        required = {
            "mean_gp_alpha",
            "mean_kernel",
            "constant_value",
            *(
                MEAN_ARD_PARAM_BY_FEATURE
                .values()
            ),
        }

        missing = (
            required.difference(
                validated
            )
        )

        if missing:
            raise KeyError(
                "Tuning grid is missing: "
                f"{sorted(missing)}"
            )

        for (
            key,
            values,
        ) in validated.items():

            if not values:
                raise ValueError(
                    "Tuning parameter "
                    f"{key!r} cannot be empty."
                )

        return validated

    # =========================================================
    # Initial params
    # =========================================================

    @staticmethod
    def _initial_params(
        *,
        grid: dict[str, list[Any]],
        base_model_config: dict[str, Any],
    ) -> dict[str, Any]:

        mean_params = dict(
            base_model_config.get(
                "mean_kernel_params",
                {},
            )
        )

        noise_params = dict(
            base_model_config.get(
                "noise_kernel_params",
                {},
            )
        )

        fallback_mean_length_scale = (
            float(
                mean_params.get(
                    "initial_length_scale",
                    1.0,
                )
            )
        )

        configured_feature_scales = dict(
            mean_params.get(
                "feature_length_scales",
                {},
            )
        )

        configured = {
            "mean_gp_alpha": (
                base_model_config.get(
                    "mean_gp_alpha"
                )
            ),
            "mean_kernel": (
                base_model_config.get(
                    "mean_kernel"
                )
            ),
            "constant_value": (
                mean_params.get(
                    "constant_value"
                )
            ),
            "noise_kernel": (
                base_model_config.get(
                    "noise_kernel"
                )
            ),
            "noise_initial_length_scale": (
                noise_params.get(
                    "initial_length_scale"
                )
            ),
            "noise_gp_alpha": (
                base_model_config.get(
                    "noise_gp_alpha"
                )
            ),
            "noise_variance_floor": (
                base_model_config.get(
                    "noise_variance_floor"
                )
            ),
        }

        for (
            feature,
            parameter_name,
        ) in (
            MEAN_ARD_PARAM_BY_FEATURE.items()
        ):

            configured[
                parameter_name
            ] = float(
                configured_feature_scales.get(
                    feature,
                    fallback_mean_length_scale,
                )
            )

        return {
            key: (
                configured.get(
                    key
                )
                if configured.get(
                    key
                ) in values
                else values[0]
            )
            for key, values
            in grid.items()
        }

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _deep_merge_dicts(
        base: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:

        merged = dict(
            base
        )

        for (
            key,
            value,
        ) in update.items():

            if (
                isinstance(
                    value,
                    dict,
                )
                and isinstance(
                    merged.get(
                        key
                    ),
                    dict,
                )
            ):

                merged[
                    key
                ] = (
                    MainHyperParameterTuning
                    ._deep_merge_dicts(
                        merged[
                            key
                        ],
                        value,
                    )
                )

            else:
                merged[
                    key
                ] = value

        return merged

    def _coverage_target(
        self,
        base_model_config: dict[str, Any],
    ) -> float:

        if (
            self.coverage_target
            is not None
        ):
            return float(
                self.coverage_target
            )

        return (
            100.0
            * float(
                base_model_config.get(
                    "confidence_level",
                    0.90,
                )
            )
        )

    @staticmethod
    def _decode_id_list(
        value: Any,
    ) -> list[int]:

        if isinstance(
            value,
            str,
        ):
            value = (
                ast.literal_eval(
                    value
                )
            )

        if isinstance(
            value,
            np.ndarray,
        ):
            value = (
                value.tolist()
            )

        if not isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            raise TypeError(
                "Experiment IDs must "
                "be list-like."
            )

        return sorted(
            {
                int(item)
                for item in value
            }
        )

    @staticmethod
    def _safe_range(
        values: np.ndarray,
    ) -> float:

        value_range = float(
            np.max(
                values
            )
            - np.min(
                values
            )
        )

        return (
            1.0
            if np.isclose(
                value_range,
                0.0,
            )
            else value_range
        )

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> float | None:

        if (
            value is None
            or pd.isna(
                value
            )
        ):
            return None

        return float(
            value
        )

    @staticmethod
    def _resolve_path(
        project_root: Path,
        value: str | Path,
    ) -> Path:

        path = Path(
            value
        )

        if path.is_absolute():
            return path

        return (
            project_root
            / path
        )

    # =========================================================
    # Output
    # =========================================================

    @staticmethod
    def _order_columns(
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:

        preferred = [
            "geometry_source",
            "target_axis",
            "target_column",

            "objective_strategy",
            "objective_order",

            "candidate_search",
            "candidate_pool_size",
            "unique_test_winner_count",

            "best_mean_kernel",
            "best_mean_gp_alpha",
            "best_mean_ls_angle",

            "best_mean_ls_collet_boost",
            "best_mean_ls_pressure_die_distance",
            "best_mean_ls_mandrel_retraction_timing",
            "best_mean_ls_pressure_die_boost",
            "best_mean_ls_clamp_die_lateral_position",
            "best_mean_ls_mandrel_position",

            "best_constant_value",

            "best_noise_kernel",
            "best_noise_initial_length_scale",
            "best_noise_gp_alpha",
            "best_noise_variance_floor",

            "cv_curve_distance_norm",
            "cv_trend_shape_loss",
            "cv_interval_shape_loss",
            "cv_coverage_target",
            "cv_coverage",
            "cv_coverage_error",
            "cv_coverage_shortfall",
            "cv_p10_coverage",
            "cv_pinaw",
            "cv_rmse_norm",

            "test_curve_distance_norm",
            "test_trend_shape_loss",
            "test_interval_shape_loss",
            "test_coverage",
            "test_coverage_error",
            "test_p10_coverage",
            "test_pinaw",
            "test_rmse_norm",

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
            if column
            in results_df.columns
        ]

        remaining = [
            column
            for column
            in results_df.columns
            if column
            not in ordered
        ]

        return results_df[
            ordered
            + remaining
        ]


HGPMainHyperParameterTuning = (
    MainHyperParameterTuning
)