from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    RBF,
)
from sklearn.model_selection import GroupKFold, KFold
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
    initial_mean_model: GaussianProcessRegressor
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
    missing_columns = set(required_columns).difference(
        dataframe.columns
    )

    if missing_columns:
        raise KeyError(
            f"{dataframe_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def _as_numeric_matrix(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    numeric_frame = dataframe[columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    values = numeric_frame.to_numpy(dtype=float)

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
                config.get("matern_nu", 2.5)
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
            config.get("constant_value", 1.0)
        ),
        constant_value_bounds=tuple(
            config.get(
                "constant_value_bounds",
                (1e-3, 1e3),
            )
        ),
    ) * base_kernel


def build_mean_kernel(
    n_features: int,
    model_config: dict[str, Any],
) -> Kernel:
    return _build_base_kernel(
        kernel_name=str(
            model_config.get(
                "mean_kernel",
                "matern_2.5",
            )
        ),
        n_features=n_features,
        config=dict(
            model_config.get(
                "mean_kernel_params",
                {},
            )
        ),
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
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=int(
            model_config.get(
                "n_restarts_optimizer",
                3,
            )
        ),
        random_state=int(
            model_config.get(
                "random_state",
                1100,
            )
        ),
        copy_X_train=True,
    )


def _cross_validated_residuals(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray | None,
    mean_kernel: Kernel,
    model_config: dict[str, Any],
) -> np.ndarray:
    requested_splits = int(
        model_config.get(
            "noise_cv_splits",
            5,
        )
    )

    if groups is not None:
        unique_group_count = np.unique(groups).size
        n_splits = min(
            requested_splits,
            unique_group_count,
        )

        if n_splits < 2:
            raise ValueError(
                "At least two unique groups are required "
                "for grouped residual cross-validation."
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )
        split_iterator = splitter.split(
            X_train,
            y_train,
            groups,
        )
    else:
        n_splits = min(
            requested_splits,
            len(y_train),
        )

        if n_splits < 2:
            raise ValueError(
                "At least two rows are required "
                "for residual cross-validation."
            )

        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(
                model_config.get(
                    "random_state",
                    1100,
                )
            ),
        )
        split_iterator = splitter.split(
            X_train,
            y_train,
        )

    predictions = np.empty_like(
        y_train,
        dtype=float,
    )

    base_alpha = float(
        model_config.get(
            "initial_alpha",
            1e-6,
        )
    )

    for train_indices, validation_indices in split_iterator:
        fold_model = _build_gp(
            kernel=clone(mean_kernel),
            alpha=base_alpha,
            model_config=model_config,
            normalize_y=False,
        )
        fold_model.fit(
            X_train[train_indices],
            y_train[train_indices],
        )
        predictions[validation_indices] = (
            fold_model.predict(
                X_train[validation_indices]
            )
        )

    return y_train - predictions


def _estimate_training_residuals(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray | None,
    initial_mean_model: GaussianProcessRegressor,
    mean_kernel: Kernel,
    model_config: dict[str, Any],
) -> np.ndarray:
    residual_mode = str(
        model_config.get(
            "residual_mode",
            "cross_validated",
        )
    ).lower()

    if residual_mode == "cross_validated":
        return _cross_validated_residuals(
            X_train=X_train,
            y_train=y_train,
            groups=groups,
            mean_kernel=mean_kernel,
            model_config=model_config,
        )

    if residual_mode == "in_sample":
        return (
            y_train
            - initial_mean_model.predict(X_train)
        )

    raise ValueError(
        "residual_mode must be 'cross_validated' "
        "or 'in_sample'."
    )


def _clip_noise_variance(
    values: np.ndarray,
    *,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=float),
        floor,
        ceiling,
    )


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_config: dict[str, Any],
    group_column: str = "Experiment_ID",
) -> tuple[HGPModelBundle, HGPPredictions]:
    """
    Train an approximate heteroscedastic Gaussian Process.

    Procedure:
        1. Scale X and y using training data only.
        2. Fit an initial mean GP.
        3. Estimate residuals, preferably with grouped CV.
        4. Fit a second GP to log residual variance.
        5. Fit the final mean GP using per-row noise variance as alpha.
        6. Return total predictive intervals:
           latent variance + predicted aleatoric variance.
    """
    required_train_columns = [
        *feature_columns,
        target_column,
    ]
    _validate_columns(
        train_df,
        required_train_columns,
        "Training data",
    )
    _validate_columns(
        test_df,
        feature_columns,
        "Test data",
    )

    X_train_raw = _as_numeric_matrix(
        train_df,
        feature_columns,
    )
    X_test_raw = _as_numeric_matrix(
        test_df,
        feature_columns,
    )
    y_train_raw = _as_numeric_vector(
        train_df,
        target_column,
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(
        X_train_raw
    )
    X_test = feature_scaler.transform(
        X_test_raw
    )
    y_train = target_scaler.fit_transform(
        y_train_raw.reshape(-1, 1)
    ).reshape(-1)

    groups: np.ndarray | None = None

    if group_column in train_df.columns:
        groups = train_df[group_column].to_numpy()

    n_features = X_train.shape[1]
    mean_kernel = build_mean_kernel(
        n_features=n_features,
        model_config=model_config,
    )
    noise_kernel = build_noise_kernel(
        n_features=n_features,
        model_config=model_config,
    )

    initial_mean_model = _build_gp(
        kernel=clone(mean_kernel),
        alpha=float(
            model_config.get(
                "initial_alpha",
                1e-6,
            )
        ),
        model_config=model_config,
        normalize_y=False,
    )
    initial_mean_model.fit(
        X_train,
        y_train,
    )

    residuals = _estimate_training_residuals(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        initial_mean_model=initial_mean_model,
        mean_kernel=mean_kernel,
        model_config=model_config,
    )

    residual_variance_epsilon = float(
        model_config.get(
            "residual_variance_epsilon",
            1e-8,
        )
    )
    log_noise_target = np.log(
        np.square(residuals)
        + residual_variance_epsilon
    )

    noise_model = _build_gp(
        kernel=noise_kernel,
        alpha=float(
            model_config.get(
                "noise_gp_alpha",
                1e-4,
            )
        ),
        model_config=model_config,
        normalize_y=True,
    )
    noise_model.fit(
        X_train,
        log_noise_target,
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

    train_noise_variance = _clip_noise_variance(
        np.exp(
            noise_model.predict(X_train)
        ),
        floor=noise_variance_floor,
        ceiling=noise_variance_ceiling,
    )

    final_mean_model = _build_gp(
        kernel=clone(mean_kernel),
        alpha=train_noise_variance,
        model_config=model_config,
        normalize_y=False,
    )
    final_mean_model.fit(
        X_train,
        y_train,
    )

    mean_scaled, latent_std_scaled = (
        final_mean_model.predict(
            X_test,
            return_std=True,
        )
    )

    test_noise_variance_scaled = (
        _clip_noise_variance(
            np.exp(
                noise_model.predict(X_test)
            ),
            floor=noise_variance_floor,
            ceiling=noise_variance_ceiling,
        )
    )
    aleatoric_std_scaled = np.sqrt(
        test_noise_variance_scaled
    )
    total_std_scaled = np.sqrt(
        np.square(latent_std_scaled)
        + test_noise_variance_scaled
    )

    target_scale = float(
        target_scaler.scale_[0]
    )
    mean = target_scaler.inverse_transform(
        mean_scaled.reshape(-1, 1)
    ).reshape(-1)
    latent_std = (
        latent_std_scaled * target_scale
    )
    aleatoric_std = (
        aleatoric_std_scaled * target_scale
    )
    total_std = (
        total_std_scaled * target_scale
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

    z_value = NormalDist().inv_cdf(
        0.5 + confidence_level / 2.0
    )

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
        initial_mean_model=initial_mean_model,
        noise_model=noise_model,
        final_mean_model=final_mean_model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=list(feature_columns),
        target_column=target_column,
        confidence_level=confidence_level,
        noise_variance_floor=noise_variance_floor,
        noise_variance_ceiling=noise_variance_ceiling,
        model_config=dict(model_config),
    )

    return model_bundle, predictions


def predict(
    model_bundle: HGPModelBundle,
    dataframe: pd.DataFrame,
) -> HGPPredictions:
    """Predict with a fitted HGP model bundle."""
    _validate_columns(
        dataframe,
        model_bundle.feature_columns,
        "Prediction data",
    )

    X_raw = _as_numeric_matrix(
        dataframe,
        model_bundle.feature_columns,
    )
    X = model_bundle.feature_scaler.transform(
        X_raw
    )

    mean_scaled, latent_std_scaled = (
        model_bundle.final_mean_model.predict(
            X,
            return_std=True,
        )
    )

    noise_variance_scaled = _clip_noise_variance(
        np.exp(
            model_bundle.noise_model.predict(X)
        ),
        floor=model_bundle.noise_variance_floor,
        ceiling=model_bundle.noise_variance_ceiling,
    )

    aleatoric_std_scaled = np.sqrt(
        noise_variance_scaled
    )
    total_std_scaled = np.sqrt(
        np.square(latent_std_scaled)
        + noise_variance_scaled
    )

    target_scale = float(
        model_bundle.target_scaler.scale_[0]
    )
    mean = (
        model_bundle.target_scaler
        .inverse_transform(
            mean_scaled.reshape(-1, 1)
        )
        .reshape(-1)
    )
    latent_std = (
        latent_std_scaled * target_scale
    )
    aleatoric_std = (
        aleatoric_std_scaled * target_scale
    )
    total_std = (
        total_std_scaled * target_scale
    )

    z_value = NormalDist().inv_cdf(
        0.5
        + model_bundle.confidence_level / 2.0
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
