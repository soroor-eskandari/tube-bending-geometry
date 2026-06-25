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
        modes = ["raw"]
        base_modes = SensorDataAugmentor.BASE_AUGMENTATION_MODES
        for size in range(1, len(base_modes) + 1):
            for mask in range(1, 1 << len(base_modes)):
                if bin(mask).count("1") != size:
                    continue
                modes.append(
                    "+".join(
                        mode
                        for idx, mode in enumerate(base_modes)
                        if mask & (1 << idx)
                    )
                )
        return modes

    @staticmethod
    def _numeric_signal_columns(df: pd.DataFrame) -> list[str]:
        excluded_cols = {SensorDataAugmentor.ID_COL, SensorDataAugmentor.TIME_COL}
        return [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in excluded_cols
        ]

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
