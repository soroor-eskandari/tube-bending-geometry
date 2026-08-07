from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, RBF
from sklearn.preprocessing import StandardScaler


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
class HGPModelBundle:
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


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    missing_columns = set(required_columns).difference(dataframe.columns)
    if missing_columns:
        raise KeyError(
            f"{dataframe_name} is missing columns: {sorted(missing_columns)}"
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
        raise ValueError(f"Non-finite values found in columns: {columns}")
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


def _build_base_kernel(
    *,
    kernel_name: str,
    n_features: int,
    config: dict[str, Any],
) -> Kernel:
    kernel_name = kernel_name.lower().strip()
    initial_length_scale = config.get("initial_length_scale", 1.0)

    if np.isscalar(initial_length_scale):
        length_scale = np.full(n_features, float(initial_length_scale))
    else:
        length_scale = np.asarray(initial_length_scale, dtype=float)
        if length_scale.shape != (n_features,):
            raise ValueError(
                "initial_length_scale must be a scalar or "
                f"a sequence of length {n_features}."
            )

    length_scale_bounds = tuple(
        config.get("length_scale_bounds", (1e-2, 1e2))
    )

    if kernel_name == "rbf":
        base_kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
        )
    elif kernel_name in {"matern", "matern_1.5", "matern_2.5"}:
        if kernel_name == "matern_1.5":
            nu = 1.5
        elif kernel_name == "matern_2.5":
            nu = 2.5
        else:
            nu = float(config.get("matern_nu", 2.5))

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
        constant_value=float(config.get("constant_value", 1.0)),
        constant_value_bounds=tuple(
            config.get("constant_value_bounds", (1e-3, 1e3))
        ),
    ) * base_kernel


def build_mean_kernel(
    n_features: int,
    model_config: dict[str, Any],
) -> Kernel:
    return _build_base_kernel(
        kernel_name=str(model_config.get("mean_kernel", "matern_2.5")),
        n_features=n_features,
        config=dict(model_config.get("mean_kernel_params", {})),
    )


def build_noise_kernel(
    n_features: int,
    model_config: dict[str, Any],
) -> Kernel:
    return _build_base_kernel(
        kernel_name=str(model_config.get("noise_kernel", "rbf")),
        n_features=n_features,
        config=dict(model_config.get("noise_kernel_params", {})),
    )


def _build_gp(
    *,
    kernel: Kernel,
    alpha: float | np.ndarray,
    model_config: dict[str, Any],
    normalize_y: bool,
) -> GaussianProcessRegressor:
    """Build a GP with fixed kernel hyperparameters and no optimization."""
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        optimizer=None,
        normalize_y=normalize_y,
        random_state=int(model_config.get("random_state", 1100)),
        copy_X_train=True,
    )


def _aggregate_repeated_inputs(
    train_df: pd.DataFrame,
    aggregation_columns: list[str],
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """
    Aggregate repeated experiments by Group_ID and curve position.

    aggregation_columns determine repeated-input groups, normally:
        Group_ID × Angle[degree]ORDistance[mm]

    feature_columns are the actual GP inputs. Group_ID is not passed to
    the GP. Setup features are retained using their first value after
    verifying that they are constant inside every aggregated input.
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

    working_df = train_df[required_columns].copy()

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

    # Each setup feature must have one value inside each Group_ID × angle.
    if retained_feature_columns:
        feature_unique_counts = (
            working_df
            .groupby(
                aggregation_columns,
                sort=False,
            )[retained_feature_columns]
            .nunique(dropna=False)
        )

        inconsistent_mask = feature_unique_counts.gt(1)

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

    aggregation_spec: dict[str, tuple[str, Any]] = {
        "target_mean": (
            target_column,
            "mean",
        ),
        "target_variance": (
            target_column,
            lambda values: float(
                np.var(values, ddof=0)
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
    ).difference(grouped.columns)

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
    Retain every Nth physical angle for GP training.

    Example with:
        training_angle_start = 0
        training_angle_step = 4

    Selected angles:
        0, 4, 8, ..., 44

    Prediction is still possible at every angle because test rows are not
    filtered.
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
        angle_values - training_angle_start
    )

    nearest_grid_index = np.rint(
        distance_from_grid / training_angle_step
    )

    expected_grid_angle = (
        training_angle_start
        + nearest_grid_index * training_angle_step
    )

    selected_mask = np.isclose(
        angle_values,
        expected_grid_angle,
        atol=angle_selection_tolerance,
        rtol=0.0,
    )

    selected_df = (
        aggregated_df.loc[selected_mask]
        .copy()
        .reset_index(drop=True)
    )

    if selected_df.empty:
        available_angles = sorted(
            np.unique(angle_values).tolist()
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

    return selected_df, selected_angles

def _clip_noise_variance(
    values: np.ndarray,
    *,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), floor, ceiling)


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
    Train an aggregated heteroscedastic Gaussian Process.

    Procedure:
        1. Aggregate repeated training rows by feature_columns.
        2. Train the noise GP directly on within-group target variance.
        3. Train one mean GP on grouped target means using predicted noise
           variance as per-row alpha.
        4. Predict intervals on the original row-level test data.

    group_column identifies the setup/group column used only for aggregation.
    It is not included in the GP input unless explicitly present in
    feature_columns.
    """

    if aggregation_columns is None:
        angle_candidates = [
            "Angle[degree]ORDistance[mm]",
            "Angle [degree]",
            "Angle [deg]",
            "Angle",
        ]
        angle_column = next(
            (column for column in angle_candidates if column in train_df.columns),
            None,
        )
        if angle_column is None:
            raise KeyError(
                "Could not infer the curve-position column. Supply "
                "aggregation_columns explicitly, for example "
                "['Group_ID', 'Angle[degree]ORDistance[mm]']."
            )
        aggregation_columns = [group_column, angle_column]
    else:
        aggregation_columns = list(aggregation_columns)

    if group_column in feature_columns:
        raise ValueError(
            f"{group_column!r} must not be included in feature_columns. "
            "Use real setup features as GP inputs and keep Group_ID only "
            "for aggregation and validation."
        )

    _validate_columns(
        train_df,
        list(dict.fromkeys([*aggregation_columns, *feature_columns, target_column])),
        "Training data",
    )
    _validate_columns(test_df, feature_columns, "Test data")

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
    X_test_raw = _as_numeric_matrix(test_df, feature_columns)
    y_mean_raw = _as_numeric_vector(aggregated_df, "target_mean")
    y_variance_raw = _as_numeric_vector(aggregated_df, "target_variance")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)
    y_mean_scaled = target_scaler.fit_transform(
        y_mean_raw.reshape(-1, 1)
    ).reshape(-1)

    target_scale = float(target_scaler.scale_[0])
    if target_scale <= 0.0:
        raise ValueError("Target scale must be positive.")

    noise_variance_floor = float(
        model_config.get("noise_variance_floor", 1e-6)
    )
    noise_variance_ceiling = float(
        model_config.get("noise_variance_ceiling", 10.0)
    )
    residual_variance_epsilon = float(
        model_config.get("residual_variance_epsilon", 1e-8)
    )

    # Convert raw target variance to the standardized target scale.
    y_variance_scaled = y_variance_raw / (target_scale ** 2)
    y_variance_scaled = _clip_noise_variance(
        y_variance_scaled,
        floor=noise_variance_floor,
        ceiling=noise_variance_ceiling,
    )
    log_noise_target = np.log(
        y_variance_scaled + residual_variance_epsilon
    )

    n_features = X_train.shape[1]
    noise_model = _build_gp(
        kernel=build_noise_kernel(n_features, model_config),
        alpha=float(model_config.get("noise_gp_alpha", 1e-4)),
        model_config=model_config,
        normalize_y=True,
    )
    noise_model.fit(X_train, log_noise_target)

    train_noise_variance = _clip_noise_variance(
        np.exp(noise_model.predict(X_train)),
        floor=noise_variance_floor,
        ceiling=noise_variance_ceiling,
    )

    mean_gp_alpha = float(model_config.get("mean_gp_alpha", 0.0))
    if mean_gp_alpha < 0.0:
        raise ValueError("mean_gp_alpha must be non-negative.")

    final_mean_model = _build_gp(
        kernel=build_mean_kernel(n_features, model_config),
        alpha=train_noise_variance + mean_gp_alpha,
        model_config=model_config,
        normalize_y=False,
    )
    final_mean_model.fit(X_train, y_mean_scaled)

    mean_scaled, latent_std_scaled = final_mean_model.predict(
        X_test,
        return_std=True,
    )

    test_noise_variance_scaled = _clip_noise_variance(
        np.exp(noise_model.predict(X_test)),
        floor=noise_variance_floor,
        ceiling=noise_variance_ceiling,
    )
    aleatoric_std_scaled = np.sqrt(test_noise_variance_scaled)
    total_std_scaled = np.sqrt(
        np.square(latent_std_scaled) + test_noise_variance_scaled
    )

    mean = target_scaler.inverse_transform(
        mean_scaled.reshape(-1, 1)
    ).reshape(-1)
    latent_std = latent_std_scaled * target_scale
    aleatoric_std = aleatoric_std_scaled * target_scale
    total_std = total_std_scaled * target_scale

    confidence_level = float(model_config.get("confidence_level", 0.90))
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)

    predictions = HGPPredictions(
        lower=mean - z_value * total_std,
        mean=mean,
        median=mean.copy(),
        upper=mean + z_value * total_std,
        latent_std=latent_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
    )

    model_bundle = HGPModelBundle(
        noise_model=noise_model,
        final_mean_model=final_mean_model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=list(feature_columns),
        target_column=target_column,
        confidence_level=confidence_level,
        noise_variance_floor=noise_variance_floor,
        noise_variance_ceiling=noise_variance_ceiling,
        model_config={
            **dict(model_config),
            "optimizer": None,
            "training_mode": (
                "aggregated_group_variance_angle_subsampled"
            ),
            "raw_train_rows": len(train_df),
            "aggregated_rows_before_angle_selection": (
                aggregated_rows_before_angle_selection
            ),
            "aggregated_train_rows": len(
                aggregated_df
            ),
            "aggregation_columns": list(
                aggregation_columns
            ),
            "angle_column": angle_column,
            "training_angles": training_angles,
            "training_angle_count": len(
                training_angles
            ),
            "mean_gp_alpha": mean_gp_alpha,
        },
    )

    return model_bundle, predictions


def predict(
    model_bundle: HGPModelBundle,
    dataframe: pd.DataFrame,
) -> HGPPredictions:
    """Predict with a fitted aggregated HGP model bundle."""
    _validate_columns(
        dataframe,
        model_bundle.feature_columns,
        "Prediction data",
    )

    X_raw = _as_numeric_matrix(dataframe, model_bundle.feature_columns)
    X = model_bundle.feature_scaler.transform(X_raw)

    mean_scaled, latent_std_scaled = model_bundle.final_mean_model.predict(
        X,
        return_std=True,
    )

    noise_variance_scaled = _clip_noise_variance(
        np.exp(model_bundle.noise_model.predict(X)),
        floor=model_bundle.noise_variance_floor,
        ceiling=model_bundle.noise_variance_ceiling,
    )

    aleatoric_std_scaled = np.sqrt(noise_variance_scaled)
    total_std_scaled = np.sqrt(
        np.square(latent_std_scaled) + noise_variance_scaled
    )

    target_scale = float(model_bundle.target_scaler.scale_[0])
    mean = model_bundle.target_scaler.inverse_transform(
        mean_scaled.reshape(-1, 1)
    ).reshape(-1)
    latent_std = latent_std_scaled * target_scale
    aleatoric_std = aleatoric_std_scaled * target_scale
    total_std = total_std_scaled * target_scale

    z_value = NormalDist().inv_cdf(
        0.5 + model_bundle.confidence_level / 2.0
    )

    return HGPPredictions(
        lower=mean - z_value * total_std,
        mean=mean,
        median=mean.copy(),
        upper=mean + z_value * total_std,
        latent_std=latent_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
    )
