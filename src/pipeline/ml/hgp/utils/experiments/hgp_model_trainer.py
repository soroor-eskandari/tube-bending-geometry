from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, RBF
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# Prediction / model containers
# ============================================================

@dataclass(frozen=True)
class HGPPredictions:
    lower: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    latent_std: np.ndarray
    aleatoric_std: np.ndarray
    total_std: np.ndarray


@dataclass
class HGPSingleModel:
    """
    One independently trained heteroscedastic GP member.
    """
    noise_model: GaussianProcessRegressor
    final_mean_model: GaussianProcessRegressor
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: list[str]
    target_column: str
    confidence_level: float
    noise_variance_floor: float
    noise_variance_ceiling: float
    model_config: dict[str, Any]


@dataclass
class HGPModelBundle:
    """
    Final HGP bundle.

    If ensemble_enabled=False:
        members contains exactly one model trained on all training data.

    If ensemble_enabled=True:
        members contains K GroupKFold members.

    noise_model / final_mean_model / feature_scaler / target_scaler are
    retained for backward compatibility and point to the first member.
    """
    noise_model: GaussianProcessRegressor
    final_mean_model: GaussianProcessRegressor
    feature_scaler: StandardScaler
    target_scaler: StandardScaler

    feature_columns: list[str]
    target_column: str
    confidence_level: float
    noise_variance_floor: float
    noise_variance_ceiling: float
    model_config: dict[str, Any]

    members: list[HGPSingleModel]
    ensemble_enabled: bool
    ensemble_cv_splits: int


# ============================================================
# Validation / conversion
# ============================================================

def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    missing_columns = set(required_columns).difference(dataframe.columns)

    if missing_columns:
        raise KeyError(
            f"{dataframe_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def _as_numeric_matrix(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    values = (
        dataframe[columns]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values found in columns: {columns}"
        )

    return values


def _as_numeric_vector(
    dataframe: pd.DataFrame,
    column: str,
) -> np.ndarray:
    values = pd.to_numeric(
        dataframe[column],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values found in target column {column!r}."
        )

    return values


# ============================================================
# Kernels
# ============================================================

def _build_base_kernel(
    *,
    kernel_name: str,
    n_features: int,
    config: dict[str, Any],
) -> Kernel:
    kernel_name = kernel_name.lower().strip()

    initial_length_scale = config.get(
        "initial_length_scale",
        1.0,
    )

    if np.isscalar(initial_length_scale):
        length_scale = np.full(
            n_features,
            float(initial_length_scale),
        )
    else:
        length_scale = np.asarray(
            initial_length_scale,
            dtype=float,
        )

        if length_scale.shape != (n_features,):
            raise ValueError(
                "initial_length_scale must be a scalar or "
                f"a sequence of length {n_features}."
            )

    length_scale_bounds = tuple(
        config.get(
            "length_scale_bounds",
            (1e-2, 1e2),
        )
    )

    if kernel_name == "rbf":
        base_kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
        )

    elif kernel_name in {
        "matern",
        "matern_1.5",
        "matern_2.5",
    }:
        if kernel_name == "matern_1.5":
            nu = 1.5
        elif kernel_name == "matern_2.5":
            nu = 2.5
        else:
            nu = float(
                config.get(
                    "matern_nu",
                    2.5,
                )
            )

        base_kernel = Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=nu,
        )

    else:
        raise ValueError(
            "kernel_name must be one of: "
            "'rbf', 'matern', 'matern_1.5', 'matern_2.5'."
        )

    return ConstantKernel(
        constant_value=float(
            config.get(
                "constant_value",
                1.0,
            )
        ),
        constant_value_bounds=tuple(
            config.get(
                "constant_value_bounds",
                (1e-3, 1e3),
            )
        ),
    ) * base_kernel


def _resolve_mean_length_scales(
    *,
    feature_columns: list[str],
    model_config: dict[str, Any],
) -> np.ndarray:
    """
    Resolve one mean-GP ARD length scale per input feature.
    """
    mean_params = dict(
        model_config.get(
            "mean_kernel_params",
            {},
        )
    )

    fallback = mean_params.get(
        "initial_length_scale",
        1.0,
    )

    feature_length_scales = mean_params.get(
        "feature_length_scales",
        {},
    )

    if feature_length_scales is None:
        feature_length_scales = {}

    if not isinstance(feature_length_scales, dict):
        raise TypeError(
            "mean_kernel_params['feature_length_scales'] "
            "must be a dictionary."
        )

    if np.isscalar(fallback):
        fallback_values = {
            feature: float(fallback)
            for feature in feature_columns
        }
    else:
        fallback_array = np.asarray(
            fallback,
            dtype=float,
        )

        if fallback_array.shape != (len(feature_columns),):
            raise ValueError(
                "mean initial_length_scale must be a scalar "
                f"or a sequence of length {len(feature_columns)}."
            )

        fallback_values = {
            feature: float(value)
            for feature, value
            in zip(feature_columns, fallback_array)
        }

    resolved = np.asarray(
        [
            float(
                feature_length_scales.get(
                    feature,
                    fallback_values[feature],
                )
            )
            for feature in feature_columns
        ],
        dtype=float,
    )

    if (
        not np.isfinite(resolved).all()
        or np.any(resolved <= 0.0)
    ):
        raise ValueError(
            "All mean-GP feature length scales "
            "must be finite and positive."
        )

    return resolved


def build_mean_kernel(
    n_features: int,
    model_config: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> Kernel:
    """
    Build the mean GP kernel with ARD length scales.
    """
    if feature_columns is None:
        feature_columns = [
            f"feature_{index}"
            for index in range(n_features)
        ]

    if len(feature_columns) != n_features:
        raise ValueError(
            "feature_columns length does not match n_features."
        )

    config = dict(
        model_config.get(
            "mean_kernel_params",
            {},
        )
    )

    config["initial_length_scale"] = (
        _resolve_mean_length_scales(
            feature_columns=feature_columns,
            model_config=model_config,
        )
    )

    return _build_base_kernel(
        kernel_name=str(
            model_config.get(
                "mean_kernel",
                "matern_2.5",
            )
        ),
        n_features=n_features,
        config=config,
    )

def build_noise_kernel(
    n_features: int,
    model_config: dict[str, Any],
) -> Kernel:
    return _build_base_kernel(
        kernel_name=str(
            model_config.get(
                "noise_kernel",
                "rbf",
            )
        ),
        n_features=n_features,
        config=dict(
            model_config.get(
                "noise_kernel_params",
                {},
            )
        ),
    )


def _build_gp(
    *,
    kernel: Kernel,
    alpha: float | np.ndarray,
    model_config: dict[str, Any],
    normalize_y: bool,
) -> GaussianProcessRegressor:
    """
    Build a GP with fixed kernel hyperparameters.

    optimizer=None is intentionally enforced.
    """
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        optimizer=None,
        normalize_y=normalize_y,
        random_state=int(
            model_config.get(
                "random_state",
                1100,
            )
        ),
        copy_X_train=True,
    )


# ============================================================
# Aggregation
# ============================================================

def _aggregate_repeated_inputs(
    train_df: pd.DataFrame,
    aggregation_columns: list[str],
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """
    Aggregate repeated experiments by Group_ID x angle.

    target_mean:
        mean geometry at one setup/angle.

    target_variance:
        empirical repeated-experiment variance at one setup/angle.

    Group_ID is used for aggregation only and must not be a GP feature.
    """

    required_columns = list(
        dict.fromkeys(
            [
                *aggregation_columns,
                *feature_columns,
                target_column,
            ]
        )
    )

    _validate_columns(
        train_df,
        required_columns,
        "Training data",
    )

    working_df = train_df[
        required_columns
    ].copy()

    for column in required_columns:
        working_df[column] = pd.to_numeric(
            working_df[column],
            errors="raise",
        )

    retained_feature_columns = [
        column
        for column in feature_columns
        if column not in aggregation_columns
    ]

    if retained_feature_columns:
        feature_unique_counts = (
            working_df
            .groupby(
                aggregation_columns,
                sort=False,
            )[retained_feature_columns]
            .nunique(dropna=False)
        )

        inconsistent_mask = (
            feature_unique_counts.gt(1)
        )

        if inconsistent_mask.any().any():
            inconsistent_columns = (
                inconsistent_mask
                .any(axis=0)
                .loc[lambda values: values]
                .index
                .tolist()
            )

            raise ValueError(
                "Setup features are not constant inside some "
                "aggregated GP inputs. Inconsistent columns: "
                f"{inconsistent_columns}"
            )

    aggregation_spec: dict[
        str,
        tuple[str, Any],
    ] = {
        "target_mean": (
            target_column,
            "mean",
        ),
        "target_variance": (
            target_column,
            lambda values: (
                float(np.var(values, ddof=1))
                if len(values) >= 2
                else np.nan
            ),
        ),
        "repeat_count": (
            target_column,
            "size",
        ),
    }

    for feature_column in retained_feature_columns:
        aggregation_spec[feature_column] = (
            feature_column,
            "first",
        )

    grouped = (
        working_df
        .groupby(
            aggregation_columns,
            as_index=False,
            sort=True,
        )
        .agg(**aggregation_spec)
    )

    if grouped.empty:
        raise ValueError(
            "Aggregated GP training dataset is empty."
        )

    missing_features = set(
        feature_columns
    ).difference(
        grouped.columns
    )

    if missing_features:
        raise RuntimeError(
            "Aggregated training data lost GP feature columns: "
            f"{sorted(missing_features)}"
        )

    return grouped


def _select_training_angles(
    aggregated_df: pd.DataFrame,
    *,
    angle_column: str,
    model_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[float]]:
    """
    Optionally retain every Nth physical angle for GP training.

    Prediction is still made for every requested test angle.
    """

    if angle_column not in aggregated_df.columns:
        raise KeyError(
            f"Angle column {angle_column!r} is missing from "
            "the aggregated training data."
        )

    training_angle_step = model_config.get(
        "training_angle_step"
    )

    if training_angle_step is None:
        all_angles = sorted(
            pd.to_numeric(
                aggregated_df[angle_column],
                errors="raise",
            )
            .astype(float)
            .unique()
            .tolist()
        )

        return (
            aggregated_df.reset_index(drop=True),
            all_angles,
        )

    training_angle_step = float(
        training_angle_step
    )

    training_angle_start = float(
        model_config.get(
            "training_angle_start",
            0.0,
        )
    )

    angle_selection_tolerance = float(
        model_config.get(
            "angle_selection_tolerance",
            1e-8,
        )
    )

    if training_angle_step <= 0.0:
        raise ValueError(
            "training_angle_step must be positive or None."
        )

    angle_values = pd.to_numeric(
        aggregated_df[angle_column],
        errors="raise",
    ).to_numpy(dtype=float)

    distance_from_grid = (
        angle_values
        - training_angle_start
    )

    nearest_grid_index = np.rint(
        distance_from_grid
        / training_angle_step
    )

    expected_grid_angle = (
        training_angle_start
        + nearest_grid_index
        * training_angle_step
    )

    selected_mask = np.isclose(
        angle_values,
        expected_grid_angle,
        atol=angle_selection_tolerance,
        rtol=0.0,
    )

    selected_df = (
        aggregated_df.loc[
            selected_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if selected_df.empty:
        available_angles = sorted(
            np.unique(
                angle_values
            ).tolist()
        )

        raise ValueError(
            "No aggregated training rows matched the configured "
            "angle grid. "
            f"start={training_angle_start}, "
            f"step={training_angle_step}, "
            f"available_angles={available_angles}"
        )

    selected_angles = sorted(
        pd.to_numeric(
            selected_df[angle_column],
            errors="raise",
        )
        .astype(float)
        .unique()
        .tolist()
    )

    return (
        selected_df,
        selected_angles,
    )


def _clip_noise_variance(
    values: np.ndarray,
    *,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            values,
            dtype=float,
        ),
        floor,
        ceiling,
    )


# ============================================================
# Single HGP member
# ============================================================

def _fit_single_hgp(
    *,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_config: dict[str, Any],
    group_column: str,
    aggregation_columns: list[str],
) -> HGPSingleModel:
    """
    Fit one complete heteroscedastic GP:

        Noise GP:
            X -> log empirical repeated-experiment variance

        Mean GP:
            X -> grouped target mean
            alpha = predicted local noise variance + mean_gp_alpha
    """

    aggregated_df = _aggregate_repeated_inputs(
        train_df=train_df,
        aggregation_columns=aggregation_columns,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    aggregated_rows_before_angle_selection = len(
        aggregated_df
    )

    angle_columns = [
        column
        for column in aggregation_columns
        if column != group_column
    ]

    if len(angle_columns) != 1:
        raise ValueError(
            "Exactly one curve-position column must exist in "
            "aggregation_columns besides group_column. "
            f"Received aggregation_columns={aggregation_columns!r}, "
            f"group_column={group_column!r}."
        )

    angle_column = angle_columns[0]

    aggregated_df, training_angles = (
        _select_training_angles(
            aggregated_df,
            angle_column=angle_column,
            model_config=model_config,
        )
    )

    X_train_raw = _as_numeric_matrix(
        aggregated_df,
        feature_columns,
    )

    y_mean_raw = _as_numeric_vector(
        aggregated_df,
        "target_mean",
    )

    # target_variance is intentionally NaN for singleton Group_ID x Angle
    # inputs. A single observation contains information about the mean, but
    # it cannot provide an empirical within-input variance estimate.
    y_variance_raw = pd.to_numeric(
        aggregated_df["target_variance"],
        errors="raise",
    ).to_numpy(dtype=float)

    if np.isinf(y_variance_raw).any():
        raise ValueError(
            "Infinite values found in target_variance."
        )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(
        X_train_raw
    )

    y_mean_scaled = (
        target_scaler
        .fit_transform(
            y_mean_raw.reshape(-1, 1)
        )
        .reshape(-1)
    )

    target_scale = float(
        target_scaler.scale_[0]
    )

    if target_scale <= 0.0:
        raise ValueError(
            "Target scale must be positive."
        )

    noise_variance_floor = float(
        model_config.get(
            "noise_variance_floor",
            1e-6,
        )
    )

    noise_variance_ceiling = float(
        model_config.get(
            "noise_variance_ceiling",
            10.0,
        )
    )

    residual_variance_epsilon = float(
        model_config.get(
            "residual_variance_epsilon",
            1e-8,
        )
    )

    y_variance_scaled = (
        y_variance_raw
        / (target_scale ** 2)
    )

    # --------------------------------------------------------
    # Train the Noise GP ONLY where repeated experiments exist.
    # A singleton does not provide an empirical process variance.
    # --------------------------------------------------------

    noise_min_repeat_count = int(
        model_config.get(
            "noise_min_repeat_count",
            2,
        )
    )

    if noise_min_repeat_count < 2:
        raise ValueError(
            "noise_min_repeat_count must be >= 2."
        )

    repeat_count = pd.to_numeric(
        aggregated_df["repeat_count"],
        errors="raise",
    ).to_numpy(dtype=int)

    noise_training_mask = (
        (repeat_count >= noise_min_repeat_count)
        & np.isfinite(y_variance_scaled)
        & (y_variance_scaled >= 0.0)
    )

    noise_training_rows = int(
        np.sum(noise_training_mask)
    )

    if noise_training_rows < 2:
        raise ValueError(
            "Noise GP requires at least two aggregated inputs "
            f"with repeat_count >= {noise_min_repeat_count}. "
            f"Found {noise_training_rows}."
        )

    noise_target_variance = (
        _clip_noise_variance(
            y_variance_scaled[
                noise_training_mask
            ],
            floor=noise_variance_floor,
            ceiling=noise_variance_ceiling,
        )
    )

    log_noise_target = np.log(
        noise_target_variance
        + residual_variance_epsilon
    )

    n_features = X_train.shape[1]

    # Sample variance estimates are much less reliable when only a few
    # repeated experiments are available.  For approximately Gaussian
    # observations, Var(log(s^2)) is roughly 2 / (n - 1).  Use this as a
    # repeat-count-aware observation-noise term for the Noise GP, with the
    # configured noise_gp_alpha retained as a numerical floor.
    base_noise_gp_alpha = float(
        model_config.get(
            "noise_gp_alpha",
            1e-4,
        )
    )

    if base_noise_gp_alpha < 0.0:
        raise ValueError(
            "noise_gp_alpha must be non-negative."
        )

    repeat_aware_noise_alpha = bool(
        model_config.get(
            "repeat_aware_noise_alpha",
            True,
        )
    )

    if repeat_aware_noise_alpha:
        noise_alpha_scale = float(
            model_config.get(
                "noise_gp_repeat_alpha_scale",
                1.0,
            )
        )

        noise_repeat_counts = repeat_count[
            noise_training_mask
        ].astype(float)

        noise_gp_alpha = (
            base_noise_gp_alpha
            + noise_alpha_scale
            * 2.0
            / np.maximum(
                noise_repeat_counts - 1.0,
                1.0,
            )
        )
    else:
        noise_gp_alpha = base_noise_gp_alpha

    noise_model = _build_gp(
        kernel=build_noise_kernel(
            n_features,
            model_config,
        ),
        alpha=noise_gp_alpha,
        model_config=model_config,
        normalize_y=True,
    )

    noise_model.fit(
        X_train[
            noise_training_mask
        ],
        log_noise_target,
    )

    # Infer noise for every mean-GP training input, including
    # singleton setups, from the repeated-observation subset.
    train_noise_variance = (
        _clip_noise_variance(
            np.exp(
                noise_model.predict(
                    X_train
                )
            ),
            floor=noise_variance_floor,
            ceiling=noise_variance_ceiling,
        )
    )

    mean_gp_alpha = float(
        model_config.get(
            "mean_gp_alpha",
            0.0,
        )
    )

    if mean_gp_alpha < 0.0:
        raise ValueError(
            "mean_gp_alpha must be non-negative."
        )

    # The Mean GP is fitted to target_mean, not to an individual
    # experiment.  If the per-experiment process variance is sigma^2, the
    # observation variance of a mean based on n repeats is sigma^2 / n.
    mean_observation_variance = (
        train_noise_variance
        / np.maximum(
            repeat_count.astype(float),
            1.0,
        )
    )

    final_mean_model = _build_gp(
        kernel=build_mean_kernel(
            n_features,
            model_config,
            feature_columns=feature_columns,
        ),
        alpha=(
            mean_observation_variance
            + mean_gp_alpha
        ),
        model_config=model_config,
        normalize_y=False,
    )

    final_mean_model.fit(
        X_train,
        y_mean_scaled,
    )

    confidence_level = float(
        model_config.get(
            "confidence_level",
            0.90,
        )
    )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    return HGPSingleModel(
        noise_model=noise_model,
        final_mean_model=final_mean_model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=list(
            feature_columns
        ),
        target_column=target_column,
        confidence_level=confidence_level,
        noise_variance_floor=noise_variance_floor,
        noise_variance_ceiling=noise_variance_ceiling,
        model_config={
            **dict(model_config),
            "optimizer": None,
            "training_mode": (
                "aggregated_group_variance"
            ),
            "raw_train_rows": int(
                len(train_df)
            ),
            "aggregated_rows_before_angle_selection": int(
                aggregated_rows_before_angle_selection
            ),
            "aggregated_train_rows": int(
                len(aggregated_df)
            ),
            "aggregation_columns": list(
                aggregation_columns
            ),
            "angle_column": angle_column,
            "training_angles": training_angles,
            "training_angle_count": int(
                len(training_angles)
            ),
            "mean_gp_alpha": mean_gp_alpha,
            "noise_min_repeat_count": int(
                noise_min_repeat_count
            ),
            "noise_training_rows": int(
                noise_training_rows
            ),
            "noise_training_fraction": float(
                noise_training_rows
                / max(
                    1,
                    len(aggregated_df),
                )
            ),
            "empirical_variance_ddof": 1,
            "repeat_aware_noise_alpha": repeat_aware_noise_alpha,
            "noise_gp_repeat_alpha_scale": float(
                model_config.get(
                    "noise_gp_repeat_alpha_scale",
                    1.0,
                )
            ),
            "mean_gp_uses_variance_of_mean": True,
            "resolved_mean_feature_length_scales": {
                feature: float(value)
                for feature, value
                in zip(
                    feature_columns,
                    _resolve_mean_length_scales(
                        feature_columns=feature_columns,
                        model_config=model_config,
                    ),
                )
            },
        },
    )


def _predict_single_member(
    *,
    member: HGPSingleModel,
    dataframe: pd.DataFrame,
) -> HGPPredictions:
    """
    Predict with one HGP member.
    """

    _validate_columns(
        dataframe,
        member.feature_columns,
        "Prediction data",
    )

    X_raw = _as_numeric_matrix(
        dataframe,
        member.feature_columns,
    )

    X = member.feature_scaler.transform(
        X_raw
    )

    (
        mean_scaled,
        latent_std_scaled,
    ) = member.final_mean_model.predict(
        X,
        return_std=True,
    )

    noise_variance_scaled = (
        _clip_noise_variance(
            np.exp(
                member.noise_model.predict(
                    X
                )
            ),
            floor=member.noise_variance_floor,
            ceiling=member.noise_variance_ceiling,
        )
    )

    aleatoric_std_scaled = np.sqrt(
        noise_variance_scaled
    )

    total_std_scaled = np.sqrt(
        np.square(
            latent_std_scaled
        )
        + noise_variance_scaled
    )

    target_scale = float(
        member.target_scaler.scale_[0]
    )

    mean = (
        member.target_scaler
        .inverse_transform(
            mean_scaled.reshape(-1, 1)
        )
        .reshape(-1)
    )

    latent_std = (
        latent_std_scaled
        * target_scale
    )

    aleatoric_std = (
        aleatoric_std_scaled
        * target_scale
    )

    total_std = (
        total_std_scaled
        * target_scale
    )

    z_value = NormalDist().inv_cdf(
        0.5
        + member.confidence_level
        / 2.0
    )

    return HGPPredictions(
        lower=(
            mean
            - z_value
            * total_std
        ),
        mean=mean,
        median=mean.copy(),
        upper=(
            mean
            + z_value
            * total_std
        ),
        latent_std=latent_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
    )


# ============================================================
# Ensemble combination
# ============================================================

def _combine_member_predictions(
    *,
    member_predictions: list[HGPPredictions],
    confidence_level: float,
) -> HGPPredictions:
    """
    Combine K HGP predictions using the law of total variance.

    Ensemble mean:
        mean_k(mu_k)

    Aleatoric variance:
        mean_k(aleatoric_var_k)

    Epistemic / latent variance:
        mean_k(latent_var_k)
        + variance_k(mu_k)

    Total variance:
        aleatoric variance
        + latent variance
        + between-model mean disagreement

    This means fold disagreement explicitly widens the final interval.
    """

    if not member_predictions:
        raise ValueError(
            "member_predictions cannot be empty."
        )

    means = np.stack(
        [
            prediction.mean
            for prediction
            in member_predictions
        ],
        axis=0,
    )

    latent_variances = np.stack(
        [
            np.square(
                prediction.latent_std
            )
            for prediction
            in member_predictions
        ],
        axis=0,
    )

    aleatoric_variances = np.stack(
        [
            np.square(
                prediction.aleatoric_std
            )
            for prediction
            in member_predictions
        ],
        axis=0,
    )

    ensemble_mean = np.mean(
        means,
        axis=0,
    )

    mean_disagreement_variance = np.var(
        means,
        axis=0,
        ddof=0,
    )

    average_latent_variance = np.mean(
        latent_variances,
        axis=0,
    )

    average_aleatoric_variance = np.mean(
        aleatoric_variances,
        axis=0,
    )

    ensemble_latent_variance = (
        average_latent_variance
        + mean_disagreement_variance
    )

    ensemble_total_variance = (
        ensemble_latent_variance
        + average_aleatoric_variance
    )

    ensemble_latent_std = np.sqrt(
        np.maximum(
            ensemble_latent_variance,
            0.0,
        )
    )

    ensemble_aleatoric_std = np.sqrt(
        np.maximum(
            average_aleatoric_variance,
            0.0,
        )
    )

    ensemble_total_std = np.sqrt(
        np.maximum(
            ensemble_total_variance,
            0.0,
        )
    )

    z_value = NormalDist().inv_cdf(
        0.5
        + confidence_level
        / 2.0
    )

    return HGPPredictions(
        lower=(
            ensemble_mean
            - z_value
            * ensemble_total_std
        ),
        mean=ensemble_mean,
        median=ensemble_mean.copy(),
        upper=(
            ensemble_mean
            + z_value
            * ensemble_total_std
        ),
        latent_std=ensemble_latent_std,
        aleatoric_std=ensemble_aleatoric_std,
        total_std=ensemble_total_std,
    )


# ============================================================
# Public training API
# ============================================================

def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_config: dict[str, Any],
    group_column: str = "Group_ID",
    aggregation_columns: list[str] | None = None,
) -> tuple[HGPModelBundle, HGPPredictions]:
    """
    Train and predict with either:

    A) one HGP on all training groups, or
    B) a GroupKFold HGP ensemble.

    Config:
        ensemble_enabled: bool
            False -> original single-model behavior.
            True  -> train K HGP members.

        ensemble_cv_splits: int
            Number of GroupKFold members, default 5.

    Important:
        Fold splitting is performed by Group_ID, never by row.
        Therefore one bending setup cannot appear in both training and
        held-out portion of the same ensemble member.
    """

    if aggregation_columns is None:
        angle_candidates = [
            "Angle[degree]ORDistance[mm]",
            "Angle [degree]",
            "Angle [deg]",
            "Angle",
        ]

        angle_column = next(
            (
                column
                for column
                in angle_candidates
                if column
                in train_df.columns
            ),
            None,
        )

        if angle_column is None:
            raise KeyError(
                "Could not infer the curve-position column. "
                "Supply aggregation_columns explicitly, for example "
                "['Group_ID', 'Angle[degree]ORDistance[mm]']."
            )

        aggregation_columns = [
            group_column,
            angle_column,
        ]

    else:
        aggregation_columns = list(
            aggregation_columns
        )

    if group_column in feature_columns:
        raise ValueError(
            f"{group_column!r} must not be included in "
            "feature_columns. Use real setup features as GP inputs "
            "and keep Group_ID only for aggregation/fold splitting."
        )

    _validate_columns(
        train_df,
        list(
            dict.fromkeys(
                [
                    *aggregation_columns,
                    *feature_columns,
                    target_column,
                ]
            )
        ),
        "Training data",
    )

    _validate_columns(
        test_df,
        feature_columns,
        "Test data",
    )

    ensemble_enabled = bool(
        model_config.get(
            "ensemble_enabled",
            False,
        )
    )

    requested_cv_splits = int(
        model_config.get(
            "ensemble_cv_splits",
            5,
        )
    )

    if requested_cv_splits < 2:
        raise ValueError(
            "ensemble_cv_splits must be >= 2."
        )

    confidence_level = float(
        model_config.get(
            "confidence_level",
            0.90,
        )
    )

    # --------------------------------------------------------
    # Original single-model mode
    # --------------------------------------------------------

    if not ensemble_enabled:
        member = _fit_single_hgp(
            train_df=train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_config=model_config,
            group_column=group_column,
            aggregation_columns=aggregation_columns,
        )

        predictions = _predict_single_member(
            member=member,
            dataframe=test_df,
        )

        bundle = HGPModelBundle(
            noise_model=member.noise_model,
            final_mean_model=member.final_mean_model,
            feature_scaler=member.feature_scaler,
            target_scaler=member.target_scaler,
            feature_columns=list(
                feature_columns
            ),
            target_column=target_column,
            confidence_level=confidence_level,
            noise_variance_floor=member.noise_variance_floor,
            noise_variance_ceiling=member.noise_variance_ceiling,
            model_config={
                # Keep ALL metadata produced by the fitted member so the
                # existing hgp_pipeline.py can still read keys such as
                # aggregated_train_rows, raw_train_rows, training_angles, ...
                **dict(member.model_config),

                "optimizer": None,
                "ensemble_enabled": False,
                "ensemble_cv_splits": 1,
                "ensemble_member_count": 1,
                "training_mode": (
                    "single_aggregated_hgp"
                ),
            },
            members=[member],
            ensemble_enabled=False,
            ensemble_cv_splits=1,
        )

        return (
            bundle,
            predictions,
        )

    # --------------------------------------------------------
    # GroupKFold ensemble mode
    # --------------------------------------------------------

    groups = train_df[
        group_column
    ].to_numpy()

    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        requested_cv_splits,
        len(unique_groups),
    )

    if n_splits < 2:
        raise ValueError(
            "GroupKFold ensemble requires at least "
            "two unique training groups."
        )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    members: list[
        HGPSingleModel
    ] = []

    member_predictions: list[
        HGPPredictions
    ] = []

    member_metadata: list[
        dict[str, Any]
    ] = []

    # X is not used by GroupKFold for grouping decisions,
    # but sklearn requires a positional X argument.
    dummy_X = np.zeros(
        (
            len(train_df),
            1,
        ),
        dtype=float,
    )

    for (
        member_index,
        (
            member_train_index,
            member_holdout_index,
        ),
    ) in enumerate(
        splitter.split(
            dummy_X,
            groups=groups,
        ),
        start=1,
    ):
        member_train_df = (
            train_df
            .iloc[
                member_train_index
            ]
            .reset_index(
                drop=True
            )
        )

        member_holdout_df = (
            train_df
            .iloc[
                member_holdout_index
            ]
            .reset_index(
                drop=True
            )
        )

        train_group_ids = sorted(
            pd.to_numeric(
                member_train_df[
                    group_column
                ],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        )

        holdout_group_ids = sorted(
            pd.to_numeric(
                member_holdout_df[
                    group_column
                ],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        )

        overlap = set(
            train_group_ids
        ).intersection(
            holdout_group_ids
        )

        if overlap:
            raise RuntimeError(
                "Group leakage detected inside ensemble member "
                f"{member_index}: {sorted(overlap)}"
            )

        member_config = {
            **dict(model_config),
            "ensemble_enabled": False,
            "ensemble_member_index": member_index,
            "ensemble_member_train_groups": train_group_ids,
            "ensemble_member_holdout_groups": holdout_group_ids,
        }

        member = _fit_single_hgp(
            train_df=member_train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_config=member_config,
            group_column=group_column,
            aggregation_columns=aggregation_columns,
        )

        prediction = _predict_single_member(
            member=member,
            dataframe=test_df,
        )

        members.append(
            member
        )

        member_predictions.append(
            prediction
        )

        member_metadata.append(
            {
                "member_index": member_index,
                "train_rows": int(
                    len(member_train_df)
                ),
                "holdout_rows": int(
                    len(member_holdout_df)
                ),
                "train_group_count": int(
                    len(train_group_ids)
                ),
                "holdout_group_count": int(
                    len(holdout_group_ids)
                ),
                "train_group_ids": train_group_ids,
                "holdout_group_ids": holdout_group_ids,
            }
        )

    predictions = _combine_member_predictions(
        member_predictions=member_predictions,
        confidence_level=confidence_level,
    )

    first_member = members[0]

    bundle = HGPModelBundle(
        noise_model=first_member.noise_model,
        final_mean_model=first_member.final_mean_model,
        feature_scaler=first_member.feature_scaler,
        target_scaler=first_member.target_scaler,
        feature_columns=list(
            feature_columns
        ),
        target_column=target_column,
        confidence_level=confidence_level,
        noise_variance_floor=first_member.noise_variance_floor,
        noise_variance_ceiling=first_member.noise_variance_ceiling,
        model_config={
            **dict(model_config),
            "optimizer": None,
            "ensemble_enabled": True,
            "ensemble_cv_splits": int(
                n_splits
            ),
            "ensemble_member_count": int(
                len(members)
            ),

            # ------------------------------------------------
            # Backward-compatible metadata expected by the
            # existing hgp_pipeline.py.
            #
            # These values summarize the ensemble members.
            # ------------------------------------------------
            "raw_train_rows": int(
                len(train_df)
            ),
            "aggregated_rows_before_angle_selection": int(
                round(
                    np.mean(
                        [
                            member.model_config[
                                "aggregated_rows_before_angle_selection"
                            ]
                            for member in members
                        ]
                    )
                )
            ),
            "aggregated_train_rows": int(
                round(
                    np.mean(
                        [
                            member.model_config[
                                "aggregated_train_rows"
                            ]
                            for member in members
                        ]
                    )
                )
            ),
            "aggregation_columns": list(
                aggregation_columns
            ),
            "angle_column": (
                members[0].model_config[
                    "angle_column"
                ]
            ),
            "training_angles": list(
                members[0].model_config[
                    "training_angles"
                ]
            ),
            "training_angle_count": int(
                members[0].model_config[
                    "training_angle_count"
                ]
            ),
            "mean_gp_alpha": float(
                members[0].model_config[
                    "mean_gp_alpha"
                ]
            ),

            "ensemble_member_metadata": member_metadata,
            "training_mode": (
                "groupkfold_hgp_ensemble"
            ),
            "ensemble_uncertainty_formula": (
                "mean_aleatoric_variance + "
                "mean_latent_variance + "
                "variance_of_member_means"
            ),
        },
        members=members,
        ensemble_enabled=True,
        ensemble_cv_splits=int(
            n_splits
        ),
    )

    return (
        bundle,
        predictions,
    )


# ============================================================
# Public prediction API
# ============================================================

def predict(
    model_bundle: HGPModelBundle,
    dataframe: pd.DataFrame,
) -> HGPPredictions:
    """
    Predict with a fitted single-model or ensemble HGP bundle.
    """

    _validate_columns(
        dataframe,
        model_bundle.feature_columns,
        "Prediction data",
    )

    # Backward-compatible single model bundle.
    if not getattr(
        model_bundle,
        "ensemble_enabled",
        False,
    ):
        if getattr(
            model_bundle,
            "members",
            None,
        ):
            member = model_bundle.members[0]

        else:
            # Supports older serialized bundles that were created before
            # the ensemble fields existed.
            member = HGPSingleModel(
                noise_model=model_bundle.noise_model,
                final_mean_model=model_bundle.final_mean_model,
                feature_scaler=model_bundle.feature_scaler,
                target_scaler=model_bundle.target_scaler,
                feature_columns=list(
                    model_bundle.feature_columns
                ),
                target_column=model_bundle.target_column,
                confidence_level=model_bundle.confidence_level,
                noise_variance_floor=model_bundle.noise_variance_floor,
                noise_variance_ceiling=model_bundle.noise_variance_ceiling,
                model_config=dict(
                    model_bundle.model_config
                ),
            )

        return _predict_single_member(
            member=member,
            dataframe=dataframe,
        )

    members = getattr(
        model_bundle,
        "members",
        None,
    )

    if not members:
        raise ValueError(
            "Ensemble model bundle contains no members."
        )

    member_predictions = [
        _predict_single_member(
            member=member,
            dataframe=dataframe,
        )
        for member
        in members
    ]

    return _combine_member_predictions(
        member_predictions=member_predictions,
        confidence_level=float(
            model_bundle.confidence_level
        ),
    )
