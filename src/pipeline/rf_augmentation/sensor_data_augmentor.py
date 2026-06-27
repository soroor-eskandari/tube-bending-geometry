import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.io_utils import write_table

logger = logging.getLogger(__name__)


class SensorDataAugmentor:
    ID_COL = "Experiment_ID"
    TIME_COL = "Time_[s]"
    BASE_AUGMENTATION_MODES = (
        "noise",
        "time-wrapping",
        "scaling",
        "jittering",
    )
    AUGMENTATION_MODES = {
        "raw",
        "noise",
        "time-wrapping",
        "scaling",
        "jittering",
        "noise+time-wrapping",
        "noise+scaling",
        "noise+jittering",
        "time-wrapping+scaling",
        "time-wrapping+jittering",
        "scaling+jittering",
        "noise+time-wrapping+scaling",
        "noise+time-wrapping+jittering",
        "noise+scaling+jittering",
        "time-wrapping+scaling+jittering",
        "noise+time-wrapping+scaling+jittering",
    }
    ALL_METHODS_MODE = "+".join(BASE_AUGMENTATION_MODES)

    @staticmethod
    @log_function
    def run(
        machine_movement_df: pd.DataFrame,
        output_dir: Path,
        augmentation_mode: str,
        random_state: int = 42,
        noise_target_snr_db: float = 40.0,
    ) -> pd.DataFrame:
        if augmentation_mode == "all":
            augmentation_mode = "+".join(SensorDataAugmentor.BASE_AUGMENTATION_MODES)

        if augmentation_mode not in SensorDataAugmentor.AUGMENTATION_MODES:
            raise ValueError(
                "augmentation_mode must be one of "
                f"{sorted(SensorDataAugmentor.AUGMENTATION_MODES)}. "
                f"Got: {augmentation_mode}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(random_state)
        numeric_signal_cols = SensorDataAugmentor._numeric_signal_columns(
            machine_movement_df
        )

        augmented_df = machine_movement_df.copy()
        for method in SensorDataAugmentor.parse_augmentation_mode(augmentation_mode):
            if method == "noise":
                augmented_df = SensorDataAugmentor.add_small_noise(
                    augmented_df,
                    numeric_signal_cols,
                    rng,
                    target_snr_db=noise_target_snr_db,
                )
            elif method == "time-wrapping":
                augmented_df = SensorDataAugmentor.apply_time_warping(
                    augmented_df,
                    numeric_signal_cols,
                    rng,
                )
            elif method == "scaling":
                augmented_df = SensorDataAugmentor.apply_scaling(
                    augmented_df,
                    numeric_signal_cols,
                    rng,
                )
            elif method == "jittering":
                augmented_df = SensorDataAugmentor.apply_jittering(
                    augmented_df,
                    numeric_signal_cols,
                    rng,
                )

        output_name = SensorDataAugmentor.mode_to_suffix(augmentation_mode)
        output_path = write_table(
            augmented_df,
            output_dir / f"machine_movement_{output_name}.parquet",
            index=False,
        )
        logger.info(
            "Saved sensor-data augmentation %s -> %s",
            augmentation_mode,
            output_path,
        )

        return augmented_df

    @staticmethod
    def parse_augmentation_mode(augmentation_mode: str) -> tuple[str, ...]:
        if augmentation_mode in ("raw", "", None):
            return ()
        if augmentation_mode == "all":
            return SensorDataAugmentor.BASE_AUGMENTATION_MODES

        methods = tuple(part.strip() for part in augmentation_mode.split("+"))
        unknown_methods = [
            method
            for method in methods
            if method not in SensorDataAugmentor.BASE_AUGMENTATION_MODES
        ]
        if unknown_methods:
            raise ValueError(
                "Unknown augmentation method(s): "
                f"{unknown_methods}. Valid methods: "
                f"{list(SensorDataAugmentor.BASE_AUGMENTATION_MODES)}"
            )
        return methods

    @staticmethod
    def mode_to_suffix(augmentation_mode: str) -> str:
        methods = SensorDataAugmentor.parse_augmentation_mode(augmentation_mode)
        if not methods:
            return "raw"
        return "__".join(method.replace("-", "_") for method in methods)

    @staticmethod
    def all_augmentation_modes() -> list[str]:
        return ["raw", SensorDataAugmentor.ALL_METHODS_MODE]

    @staticmethod
    def _numeric_signal_columns(df: pd.DataFrame) -> list[str]:
        excluded_cols = {SensorDataAugmentor.ID_COL, SensorDataAugmentor.TIME_COL}
        return [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in excluded_cols
        ]

    @staticmethod
    def apply_all_methods_from_group_ranges(
        df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, dict]:
        signal_cols = SensorDataAugmentor._numeric_signal_columns(df)
        if not signal_cols:
            return df.copy(), {}

        stats = SensorDataAugmentor._group_signal_augmentation_ranges(
            df,
            signal_cols,
        )
        augmented = SensorDataAugmentor.add_noise_from_group_range(
            df,
            signal_cols,
            rng,
            stats["noise_std_max"],
        )
        augmented = SensorDataAugmentor.apply_time_warping_from_group_range(
            augmented,
            signal_cols,
            rng,
            stats["gamma_min"],
            stats["gamma_max"],
        )
        augmented = SensorDataAugmentor.apply_scaling_from_group_range(
            augmented,
            signal_cols,
            rng,
            stats["scale_min"],
            stats["scale_max"],
        )
        augmented = SensorDataAugmentor.apply_jittering_from_group_range(
            augmented,
            signal_cols,
            rng,
            stats["jitter_std_max"],
        )
        return augmented, stats

    @staticmethod
    def _group_signal_augmentation_ranges(
        df: pd.DataFrame,
        signal_cols: list[str],
    ) -> dict:
        grouped = df.groupby(SensorDataAugmentor.ID_COL, sort=False)
        values = df[signal_cols].to_numpy(dtype=float)
        signal_std = np.nanstd(values, axis=0)
        signal_std = np.where(np.isfinite(signal_std), signal_std, 0.0)

        noise_limits = []
        jitter_limits = []
        scale_rows = []
        durations = []

        for _, group in grouped:
            group = group.sort_values(SensorDataAugmentor.TIME_COL)
            group_values = group[signal_cols].to_numpy(dtype=float)

            if group_values.shape[0] > 1:
                diff_std = np.nanstd(np.diff(group_values, axis=0), axis=0)
                diff_std = np.where(np.isfinite(diff_std), diff_std, 0.0)
                noise_limits.append(0.5 * diff_std)

                rolling_mean = (
                    pd.DataFrame(group_values)
                    .rolling(window=5, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                residual_std = np.nanstd(group_values - rolling_mean, axis=0)
                residual_std = np.where(np.isfinite(residual_std), residual_std, 0.0)
                jitter_limits.append(residual_std)

            exp_std = np.nanstd(group_values, axis=0)
            exp_std = np.where(np.isfinite(exp_std), exp_std, 0.0)
            scale_rows.append(exp_std)

            if SensorDataAugmentor.TIME_COL in group:
                time_values = group[SensorDataAugmentor.TIME_COL].to_numpy(dtype=float)
                duration = np.nanmax(time_values) - np.nanmin(time_values)
                if np.isfinite(duration) and duration > 0:
                    durations.append(duration)

        if noise_limits:
            noise_std_max = np.nanmax(np.vstack(noise_limits), axis=0)
        else:
            noise_std_max = np.zeros(len(signal_cols), dtype=float)
        if jitter_limits:
            jitter_std_max = np.nanmax(np.vstack(jitter_limits), axis=0)
        else:
            jitter_std_max = np.zeros(len(signal_cols), dtype=float)

        noise_std_max = np.minimum(
            np.where(np.isfinite(noise_std_max), noise_std_max, 0.0),
            0.10 * signal_std,
        )
        jitter_std_max = np.minimum(
            np.where(np.isfinite(jitter_std_max), jitter_std_max, 0.0),
            0.10 * signal_std,
        )

        scale_matrix = (
            np.vstack(scale_rows)
            if scale_rows
            else np.zeros((1, len(signal_cols)))
        )
        scale_reference = np.nanmedian(scale_matrix, axis=0)
        scale_reference = np.where(scale_reference > 1e-12, scale_reference, np.nan)
        scale_ratios = scale_matrix / scale_reference
        scale_min = np.nanmin(scale_ratios, axis=0)
        scale_max = np.nanmax(scale_ratios, axis=0)
        scale_min = np.where(np.isfinite(scale_min), scale_min, 1.0)
        scale_max = np.where(np.isfinite(scale_max), scale_max, 1.0)
        scale_min = np.clip(scale_min, 0.90, 1.10)
        scale_max = np.clip(scale_max, 0.90, 1.10)
        scale_min = np.minimum(scale_min, 1.0)
        scale_max = np.maximum(scale_max, 1.0)

        if len(durations) > 1:
            duration_reference = np.median(durations)
            duration_ratios = np.array(durations) / duration_reference
            gamma_min = float(np.clip(np.nanmin(duration_ratios), 0.88, 1.12))
            gamma_max = float(np.clip(np.nanmax(duration_ratios), 0.88, 1.12))
        else:
            gamma_min = 1.0
            gamma_max = 1.0

        return {
            "signal_cols": signal_cols,
            "noise_std_max": noise_std_max,
            "jitter_std_max": jitter_std_max,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "gamma_min": gamma_min,
            "gamma_max": gamma_max,
        }

    @staticmethod
    def add_noise_from_group_range(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        noise_std_max: np.ndarray,
    ) -> pd.DataFrame:
        augmented = df.copy()

        for _, index in augmented.groupby(
            SensorDataAugmentor.ID_COL,
            sort=False,
        ).groups.items():
            std = rng.uniform(
                low=np.zeros(len(signal_cols), dtype=float),
                high=noise_std_max,
            )
            noise = rng.normal(
                loc=0.0,
                scale=std,
                size=(len(index), len(signal_cols)),
            )
            augmented.loc[index, signal_cols] = (
                augmented.loc[index, signal_cols].to_numpy(dtype=float) + noise
            )

        return augmented

    @staticmethod
    def apply_time_warping_from_group_range(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        gamma_min: float,
        gamma_max: float,
    ) -> pd.DataFrame:
        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()
            time_values = group[SensorDataAugmentor.TIME_COL].to_numpy(dtype=float)

            if len(group) < 3 or np.ptp(time_values) <= 0:
                augmented_groups.append(group)
                continue

            normalized_time = (time_values - time_values[0]) / (
                time_values[-1] - time_values[0]
            )
            gamma = rng.uniform(gamma_min, gamma_max)
            warped_time = time_values[0] + (normalized_time ** gamma) * (
                time_values[-1] - time_values[0]
            )

            for col in signal_cols:
                group[col] = np.interp(
                    warped_time,
                    time_values,
                    group[col].to_numpy(),
                )

            augmented_groups.append(group)

        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

    @staticmethod
    def apply_scaling_from_group_range(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        scale_min: np.ndarray,
        scale_max: np.ndarray,
    ) -> pd.DataFrame:
        augmented = df.copy()

        for _, index in augmented.groupby(
            SensorDataAugmentor.ID_COL,
            sort=False,
        ).groups.items():
            factors = rng.uniform(
                low=scale_min,
                high=scale_max,
                size=len(signal_cols),
            )
            augmented.loc[index, signal_cols] = (
                augmented.loc[index, signal_cols].to_numpy(dtype=float) * factors
            )

        return augmented

    @staticmethod
    def apply_jittering_from_group_range(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        jitter_std_max: np.ndarray,
        n_control_points: int = 8,
    ) -> pd.DataFrame:
        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()
            n_rows = len(group)

            if n_rows < 2:
                augmented_groups.append(group)
                continue

            control_x = np.linspace(0, n_rows - 1, min(n_control_points, n_rows))
            row_x = np.arange(n_rows)

            for col_idx, col in enumerate(signal_cols):
                control_std = rng.uniform(0.0, jitter_std_max[col_idx])
                control_noise = rng.normal(
                    loc=0.0,
                    scale=control_std,
                    size=len(control_x),
                )
                smooth_noise = np.interp(row_x, control_x, control_noise)
                group[col] = group[col].to_numpy(dtype=float) + smooth_noise

            augmented_groups.append(group)

        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

    @staticmethod
    def add_small_noise(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        target_snr_db: float = 40.0,
        min_power: float = 1e-12,
    ) -> pd.DataFrame:
        augmented = df.copy()

        for _, index in augmented.groupby(
            SensorDataAugmentor.ID_COL,
            sort=False,
        ).groups.items():
            group_values = augmented.loc[index, signal_cols].to_numpy(dtype=float)
            col_mean = np.nanmean(group_values, axis=0)
            centered_values = group_values - col_mean
            signal_power = np.nanmean(centered_values**2, axis=0)
            active_cols = signal_power > min_power

            if not np.any(active_cols):
                continue

            snr_linear = 10.0 ** (target_snr_db / 10.0)
            noise_std = np.zeros(len(signal_cols), dtype=float)
            noise_std[active_cols] = np.sqrt(signal_power[active_cols] / snr_linear)
            noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=group_values.shape,
            )
            group_values[:, active_cols] = (
                group_values[:, active_cols] + noise[:, active_cols]
            )
            augmented.loc[index, signal_cols] = group_values

        return augmented

    @staticmethod
    def apply_time_warping(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        warp_strength: float = 0.12,
    ) -> pd.DataFrame:
        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()
            time_values = group[SensorDataAugmentor.TIME_COL].to_numpy()

            if len(group) < 3 or np.ptp(time_values) <= 0:
                augmented_groups.append(group)
                continue

            normalized_time = (time_values - time_values[0]) / (
                time_values[-1] - time_values[0]
            )
            gamma = rng.uniform(1.0 - warp_strength, 1.0 + warp_strength)
            warped_time = time_values[0] + (normalized_time ** gamma) * (
                time_values[-1] - time_values[0]
            )

            for col in signal_cols:
                group[col] = np.interp(
                    warped_time,
                    time_values,
                    group[col].to_numpy(),
                )

            augmented_groups.append(group)

        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

    @staticmethod
    def apply_scaling(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        scale_std: float = 0.03,
    ) -> pd.DataFrame:
        augmented = df.copy()

        for _, index in augmented.groupby(SensorDataAugmentor.ID_COL, sort=False).groups.items():
            factors = rng.normal(
                loc=1.0,
                scale=scale_std,
                size=len(signal_cols),
            )
            augmented.loc[index, signal_cols] = (
                augmented.loc[index, signal_cols].to_numpy() * factors
            )

        return augmented

    @staticmethod
    def apply_jittering(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        jitter_scale: float = 0.01,
        n_control_points: int = 8,
    ) -> pd.DataFrame:
        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()
            n_rows = len(group)

            if n_rows < 2:
                augmented_groups.append(group)
                continue

            control_x = np.linspace(0, n_rows - 1, min(n_control_points, n_rows))
            row_x = np.arange(n_rows)
            col_std = group[signal_cols].std(axis=0).fillna(0.0).to_numpy()

            for col_idx, col in enumerate(signal_cols):
                control_noise = rng.normal(
                    loc=0.0,
                    scale=jitter_scale * col_std[col_idx],
                    size=len(control_x),
                )
                smooth_noise = np.interp(row_x, control_x, control_noise)
                group[col] = group[col].to_numpy() + smooth_noise

            augmented_groups.append(group)

        return pd.concat(augmented_groups, ignore_index=True)[df.columns]
