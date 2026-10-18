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

    TORQUE_NOISE_SNR_DB = 40.0
    NON_TORQUE_NOISE_SNR_DB = 53.0
    NOISE_METHOD = "block-residual-bootstrap"
    NOISE_BLOCK_SIZE = 5
    NOISE_RESIDUAL_SCALE = 0.20
    NOISE_TREND_WINDOW = 11
    NOISE_CLIP_QUANTILE = 0.01
    NOISE_MAX_STD_FRACTION = 0.05
    TIME_WARP_STRENGTH = 0.12
    TIME_WARP_MIN_ORDER_STABILITY = 0.80
    TIME_WARP_MIN_TIMING_RANGE = 0.05
    TIME_WARP_MIN_EXPERIMENTS = 10
    TIME_WARP_GRID_POINTS = 500
    TIME_WARP_SMOOTHING_WINDOW = 15
    TIME_WARP_PROBABILITY = 0.70
    SCALING_STD = 0.03
    SCALING_PROBABILITY = 0.70
    SCALING_MAX_FACTOR_DEVIATION = 0.05
    SCALING_BASELINE_FRACTION = 0.05
    SCALING_STD_BY_SIGNAL = {
        # Movement and angle channels. Channels with negligible observed
        # amplitude variation are intentionally left unchanged.
        "BEND-DIE_LATERAL_Movement_[mm]": 0.0,
        "BEND-DIE_ROTATING_Angle_[°]": 0.0,
        "CLAMP-DIE_LATERAL_Movement_[mm]": 0.0,
        "COLLET_AXIAL_Movement_[mm]": 0.015,
        "MANDREL_AXIAL_Movement_[mm]": 0.0,
        "PRESSURE-DIE_AXIAL_Movement_[mm]": 0.0,
        # Torque channels. The values are conservative and signal-specific.
        "MACHINE_BEND-DIE_LATERAL_Max_Torque_[%]": 0.03,
        "MACHINE_BEND-DIE_ROTATING_Max_Torque_[%]": 0.02,
        "MACHINE_CLAMP-DIE_LATERAL_Max_Torque_[%]": 0.02,
        "MACHINE_COLLET_AXIAL_Max_Torque_[%]": 0.03,
        "MACHINE_MANDREL_AXIAL_Max_Torque_[%]": 0.03,
        "MACHINE_PRESSURE-DIE_LATERAL_Max_Torque_[%]": 0.005,
    }
    # ``None`` activates data-driven, signal-specific torque jittering.
    # Passing a float to ``run`` remains supported for backward compatibility.
    JITTER_SCALE = None
    JITTER_CONTROL_POINTS = 64
    JITTER_TREND_WINDOW = 101
    JITTER_SMOOTHING_WINDOW = 21
    JITTER_MAX_STD_FRACTION = 0.10

    @staticmethod
    @log_function
    def run(
        machine_movement_df: pd.DataFrame,
        output_dir: Path,
        random_state: int = 42,
        apply_noise: bool = True,
        apply_time_wrapping: bool = True,
        apply_scaling: bool = True,
        apply_jittering: bool = True,
        torque_noise_snr_db: float = TORQUE_NOISE_SNR_DB,
        non_torque_noise_snr_db: float = NON_TORQUE_NOISE_SNR_DB,
        noise_method: str = NOISE_METHOD,
        noise_block_size: int = NOISE_BLOCK_SIZE,
        noise_residual_scale: float = NOISE_RESIDUAL_SCALE,
        noise_trend_window: int = NOISE_TREND_WINDOW,
        noise_clip_quantile: float = NOISE_CLIP_QUANTILE,
        noise_max_std_fraction: float = NOISE_MAX_STD_FRACTION,
        warp_strength: float = TIME_WARP_STRENGTH,
        scale_std: float = SCALING_STD,
        scale_std_by_signal: dict[str, float] | None = None,
        scaling_probability: float = SCALING_PROBABILITY,
        scaling_max_factor_deviation: float = SCALING_MAX_FACTOR_DEVIATION,
        scaling_baseline_fraction: float = SCALING_BASELINE_FRACTION,
        jitter_scale: float | None = JITTER_SCALE,
        jitter_control_points: int = JITTER_CONTROL_POINTS,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(random_state)
        numeric_signal_cols = SensorDataAugmentor._numeric_signal_columns(
            machine_movement_df
        )

        enabled_methods = []
        augmented_df = machine_movement_df.copy()

        if apply_noise:
            if noise_method == "block-residual-bootstrap":
                augmented_df = SensorDataAugmentor.add_block_residual_noise(
                    augmented_df,
                    numeric_signal_cols,
                    rng,
                    block_size=noise_block_size,
                    residual_scale=noise_residual_scale,
                    trend_window=noise_trend_window,
                    clip_quantile=noise_clip_quantile,
                    max_noise_std_fraction=noise_max_std_fraction,
                )
            elif noise_method == "gaussian":
                augmented_df = SensorDataAugmentor.add_signal_specific_noise(
                    augmented_df,
                    numeric_signal_cols,
                    rng,
                    torque_snr_db=torque_noise_snr_db,
                    non_torque_snr_db=non_torque_noise_snr_db,
                )
            else:
                raise ValueError(
                    "noise_method must be 'block-residual-bootstrap' "
                    "or 'gaussian'."
                )
            enabled_methods.append("noise")

        if apply_time_wrapping:
            warp_stats = SensorDataAugmentor._empirical_time_warp_statistics(
                machine_movement_df,
                numeric_signal_cols,
                gamma_lower_bound=1.0 - warp_strength,
                gamma_upper_bound=1.0 + warp_strength,
            )
            augmented_df = SensorDataAugmentor.apply_empirical_time_warping(
                augmented_df,
                numeric_signal_cols,
                rng,
                gamma_values=warp_stats["gamma_values"],
                warp_probability=SensorDataAugmentor.TIME_WARP_PROBABILITY,
            )
            enabled_methods.append("time_warping")

        if apply_scaling:
            augmented_df = SensorDataAugmentor.apply_signal_specific_scaling(
                augmented_df,
                numeric_signal_cols,
                rng,
                scale_std=scale_std,
                scale_std_by_signal=scale_std_by_signal,
                scaling_probability=scaling_probability,
                maximum_factor_deviation=scaling_max_factor_deviation,
                baseline_fraction=scaling_baseline_fraction,
            )
            enabled_methods.append("scaling")

        if apply_jittering:
            augmented_df = SensorDataAugmentor.apply_torque_jittering(
                augmented_df, numeric_signal_cols, rng,
                jitter_scale=jitter_scale,
                n_control_points=jitter_control_points,
            )
            enabled_methods.append("jittering")

        output_name = "__".join(enabled_methods) if enabled_methods else "raw"
        output_path = write_table(
            augmented_df,
            output_dir / f"machine_movement_{output_name}.parquet",
            index=False,
        )
        logger.info(
            "Saved sensor-data augmentation %s -> %s",
            enabled_methods or ["raw"],
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
    def _torque_signal_columns(signal_cols: list[str]) -> list[str]:
        return [col for col in signal_cols if "Torque_[%]" in col]

    @staticmethod
    def _normalized_signal_profiles(
        df: pd.DataFrame,
        signal_cols: list[str],
        grid_points: int = TIME_WARP_GRID_POINTS,
    ) -> tuple[np.ndarray, list[dict]]:
        """Interpolate original experiment signals onto a common time grid."""
        if grid_points < 5:
            raise ValueError("grid_points must be at least 5.")

        normalized_grid = np.linspace(0.0, 1.0, int(grid_points))
        profile_rows = []

        for experiment_id, group in df.groupby(
            SensorDataAugmentor.ID_COL,
            sort=False,
        ):
            group = group.sort_values(SensorDataAugmentor.TIME_COL)
            time_values = group[SensorDataAugmentor.TIME_COL].to_numpy(
                dtype=float
            )
            finite_time = np.isfinite(time_values)
            if finite_time.sum() < 3:
                continue

            time_values = time_values[finite_time]
            duration = time_values[-1] - time_values[0]
            if not np.isfinite(duration) or duration <= 1e-12:
                continue

            normalized_time = (time_values - time_values[0]) / duration

            for signal_name in signal_cols:
                values = group[signal_name].to_numpy(dtype=float)[finite_time]
                valid = np.isfinite(normalized_time) & np.isfinite(values)
                if valid.sum() < 3:
                    continue

                unique_time, unique_indices = np.unique(
                    normalized_time[valid],
                    return_index=True,
                )
                unique_values = values[valid][unique_indices]
                if len(unique_time) < 3:
                    continue

                curve = np.interp(
                    normalized_grid,
                    unique_time,
                    unique_values,
                )
                if np.ptp(curve) <= 1e-12:
                    continue

                profile_rows.append({
                    "Experiment_ID": experiment_id,
                    "Signal": signal_name,
                    "Curve": curve,
                })

        return normalized_grid, profile_rows

    @staticmethod
    def _event_timing_table(
        df: pd.DataFrame,
        signal_cols: list[str],
        grid_points: int = TIME_WARP_GRID_POINTS,
        smoothing_window: int = TIME_WARP_SMOOTHING_WINDOW,
    ) -> pd.DataFrame:
        """Extract peak, valley, rise, and fall times from each profile."""
        normalized_grid, profile_rows = (
            SensorDataAugmentor._normalized_signal_profiles(
                df=df,
                signal_cols=signal_cols,
                grid_points=grid_points,
            )
        )
        event_names = ("peak", "valley", "rise", "fall")
        event_rows = []

        for profile in profile_rows:
            curve = np.asarray(profile["Curve"], dtype=float)
            local_window = min(int(smoothing_window), len(curve))
            if local_window % 2 == 0:
                local_window -= 1
            local_window = max(local_window, 1)

            smooth = (
                pd.Series(curve)
                .rolling(local_window, center=True, min_periods=1)
                .median()
                .rolling(local_window, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            derivative = np.gradient(smooth, normalized_grid)
            event_times = {
                "peak": float(normalized_grid[int(np.argmax(smooth))]),
                "valley": float(normalized_grid[int(np.argmin(smooth))]),
                "rise": float(normalized_grid[int(np.argmax(derivative))]),
                "fall": float(normalized_grid[int(np.argmin(derivative))]),
            }
            event_order = tuple(
                name
                for name, _ in sorted(
                    event_times.items(),
                    key=lambda item: (
                        item[1],
                        event_names.index(item[0]),
                    ),
                )
            )
            event_rows.append({
                "Experiment_ID": profile["Experiment_ID"],
                "Signal": profile["Signal"],
                **{f"{name}_time": value for name, value in event_times.items()},
                "event_order": event_order,
            })

        return pd.DataFrame(event_rows)

    @staticmethod
    def _empirical_time_warp_statistics(
        df: pd.DataFrame,
        signal_cols: list[str],
        minimum_order_stability: float = TIME_WARP_MIN_ORDER_STABILITY,
        minimum_timing_range: float = TIME_WARP_MIN_TIMING_RANGE,
        minimum_experiments: int = TIME_WARP_MIN_EXPERIMENTS,
        gamma_lower_bound: float = 1.0 - TIME_WARP_STRENGTH,
        gamma_upper_bound: float = 1.0 + TIME_WARP_STRENGTH,
    ) -> dict:
        """Estimate a group-level empirical distribution of warp factors.

        Only signals with a stable dominant event order and measurable timing
        variation contribute. For each experiment, event-specific gamma values
        are combined with a median so one coherent warp factor represents the
        experiment's temporal displacement across all reliable channels.
        """
        if not 0.0 <= minimum_order_stability <= 1.0:
            raise ValueError("minimum_order_stability must be in [0, 1].")
        if minimum_timing_range < 0.0:
            raise ValueError("minimum_timing_range must be non-negative.")
        if minimum_experiments < 2:
            raise ValueError("minimum_experiments must be at least 2.")
        if not 0.0 < gamma_lower_bound <= 1.0 <= gamma_upper_bound:
            raise ValueError("Gamma bounds must satisfy 0 < lower <= 1 <= upper.")

        timing_df = SensorDataAugmentor._event_timing_table(df, signal_cols)
        identity = np.array([1.0], dtype=float)
        empty_result = {
            "gamma_values": identity,
            "gamma_min": 1.0,
            "gamma_max": 1.0,
            "time_warp_supported": False,
            "time_warp_signal_evidence": {},
            "time_warp_eligible_signals": [],
        }
        if timing_df.empty:
            return empty_result

        event_names = ("peak", "valley", "rise", "fall")
        evidence = {}
        eligible_signals = []
        experiment_gamma_candidates: dict[object, list[float]] = {}

        for signal_name, signal_df in timing_df.groupby("Signal", sort=False):
            experiment_count = signal_df["Experiment_ID"].nunique()
            if experiment_count < minimum_experiments:
                continue

            order_counts = signal_df["event_order"].value_counts()
            if order_counts.empty:
                continue
            dominant_order = order_counts.index[0]
            order_share = float(order_counts.iloc[0] / experiment_count)
            dominant_df = signal_df[
                signal_df["event_order"] == dominant_order
            ].copy()

            event_ranges = {}
            for event_name in event_names:
                column = f"{event_name}_time"
                values = dominant_df[column].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                event_ranges[event_name] = (
                    float(values.quantile(0.90) - values.quantile(0.10))
                    if not values.empty
                    else 0.0
                )

            maximum_range = max(event_ranges.values(), default=0.0)
            supported = (
                order_share >= minimum_order_stability
                and maximum_range >= minimum_timing_range
            )
            evidence[signal_name] = {
                "experiments": int(experiment_count),
                "dominant_order": dominant_order,
                "dominant_order_share": order_share,
                "event_timing_ranges": event_ranges,
                "maximum_event_timing_range": maximum_range,
                "supported": supported,
            }
            if not supported:
                continue

            eligible_signals.append(signal_name)
            for event_name in event_names:
                column = f"{event_name}_time"
                reference_time = float(dominant_df[column].median())
                if not 0.01 < reference_time < 0.99:
                    continue

                for row in dominant_df[["Experiment_ID", column]].itertuples(
                    index=False,
                    name=None,
                ):
                    experiment_id, observed_time = row
                    observed_time = float(observed_time)
                    if not 0.01 < observed_time < 0.99:
                        continue
                    gamma = np.log(reference_time) / np.log(observed_time)
                    if not np.isfinite(gamma):
                        continue
                    gamma = float(
                        np.clip(gamma, gamma_lower_bound, gamma_upper_bound)
                    )
                    experiment_gamma_candidates.setdefault(
                        experiment_id,
                        [],
                    ).append(gamma)

        empirical_gamma_values = np.array(
            [
                np.median(values)
                for values in experiment_gamma_candidates.values()
                if values
            ],
            dtype=float,
        )
        empirical_gamma_values = empirical_gamma_values[
            np.isfinite(empirical_gamma_values)
        ]
        if not eligible_signals or empirical_gamma_values.size < 2:
            return {
                **empty_result,
                "time_warp_signal_evidence": evidence,
            }

        empirical_gamma_values = np.clip(
            empirical_gamma_values,
            gamma_lower_bound,
            gamma_upper_bound,
        )
        # Include the identity transformation explicitly. The separate warp
        # probability also leaves some generated experiments unchanged.
        gamma_values = np.concatenate([empirical_gamma_values, identity])

        return {
            "gamma_values": gamma_values,
            "gamma_min": float(np.min(gamma_values)),
            "gamma_max": float(np.max(gamma_values)),
            "time_warp_supported": True,
            "time_warp_signal_evidence": evidence,
            "time_warp_eligible_signals": eligible_signals,
        }

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
            gamma_values=stats["gamma_values"],
            warp_probability=SensorDataAugmentor.TIME_WARP_PROBABILITY,
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
    def apply_all_methods_from_reference_ranges(
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, dict]:
        signal_cols = SensorDataAugmentor._numeric_signal_columns(df)
        if not signal_cols:
            return df.copy(), {}

        reference_signal_cols = [
            col for col in signal_cols if col in reference_df.columns
        ]
        if reference_signal_cols != signal_cols:
            missing_cols = sorted(set(signal_cols) - set(reference_signal_cols))
            raise ValueError(
                "reference_df is missing signal columns required for augmentation: "
                f"{missing_cols}"
            )

        stats = SensorDataAugmentor._group_signal_augmentation_ranges(
            reference_df,
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
            gamma_values=stats["gamma_values"],
            warp_probability=SensorDataAugmentor.TIME_WARP_PROBABILITY,
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

        warp_stats = SensorDataAugmentor._empirical_time_warp_statistics(
            df,
            signal_cols,
        )

        return {
            "signal_cols": signal_cols,
            "noise_std_max": noise_std_max,
            "jitter_std_max": jitter_std_max,
            "scale_min": scale_min,
            "scale_max": scale_max,
            **warp_stats,
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
        gamma_values: np.ndarray,
        warp_probability: float = TIME_WARP_PROBABILITY,
    ) -> pd.DataFrame:
        return SensorDataAugmentor.apply_empirical_time_warping(
            df=df,
            signal_cols=signal_cols,
            rng=rng,
            gamma_values=gamma_values,
            warp_probability=warp_probability,
        )

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

            for col_idx, col in enumerate(signal_cols):
                control_std = rng.uniform(0.0, jitter_std_max[col_idx])
                smooth_noise = SensorDataAugmentor._smooth_jitter_sequence(
                    n_rows=n_rows,
                    n_control_points=n_control_points,
                    target_std=control_std,
                    rng=rng,
                )
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
    def add_signal_specific_noise(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        torque_snr_db: float = 40.0,
        non_torque_snr_db: float = 53.0,
        min_power: float = 1e-12,
    ) -> pd.DataFrame:
        augmented = df.copy()
        torque_cols = set(SensorDataAugmentor._torque_signal_columns(signal_cols))
        snr_db = np.array([
            torque_snr_db if col in torque_cols else non_torque_snr_db
            for col in signal_cols
        ], dtype=float)
        snr_linear = 10.0 ** (snr_db / 10.0)

        for _, index in augmented.groupby(
            SensorDataAugmentor.ID_COL, sort=False,
        ).groups.items():
            values = augmented.loc[index, signal_cols].to_numpy(dtype=float)
            centered = values - np.nanmean(values, axis=0)
            signal_power = np.nanmean(centered**2, axis=0)
            active = np.isfinite(signal_power) & (signal_power > min_power)
            noise_std = np.zeros(len(signal_cols), dtype=float)
            noise_std[active] = np.sqrt(signal_power[active] / snr_linear[active])
            values = values + rng.normal(0.0, noise_std, size=values.shape)
            augmented.loc[index, signal_cols] = values

        return augmented

    @staticmethod
    def _residual_noise_pool(
        df: pd.DataFrame,
        signal_cols: list[str],
        trend_window: int = NOISE_TREND_WINDOW,
        clip_quantile: float = NOISE_CLIP_QUANTILE,
    ) -> dict[str, list[np.ndarray]]:
        """Extract clipped, zero-centred residual sequences per signal.

        A centred rolling median estimates the local signal trend separately
        for every experiment. Keeping each residual sequence intact allows
        short blocks to be sampled without discarding their local temporal
        dependence.
        """
        if trend_window < 1:
            raise ValueError("trend_window must be a positive integer.")
        if not 0.0 <= clip_quantile < 0.5:
            raise ValueError("clip_quantile must be in [0, 0.5).")

        residual_pool = {col: [] for col in signal_cols}

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL)

            for col in signal_cols:
                values = group[col].to_numpy(dtype=float)
                finite = np.isfinite(values)
                if finite.sum() < 3:
                    continue

                interpolated = (
                    pd.Series(values)
                    .interpolate(limit_direction="both")
                )
                local_window = min(int(trend_window), len(interpolated))
                if local_window % 2 == 0:
                    local_window -= 1
                local_window = max(local_window, 1)

                trend = interpolated.rolling(
                    window=local_window,
                    center=True,
                    min_periods=1,
                ).median()
                residual = (
                    interpolated.to_numpy(dtype=float)
                    - trend.to_numpy(dtype=float)
                )
                residual = residual[np.isfinite(residual)]
                if residual.size < 2 or np.std(residual) <= 1e-12:
                    continue

                residual_pool[col].append(residual)

        if clip_quantile > 0.0:
            for col, sequences in residual_pool.items():
                if not sequences:
                    continue
                combined = np.concatenate(sequences)
                lower, upper = np.quantile(
                    combined,
                    [clip_quantile, 1.0 - clip_quantile],
                )
                residual_pool[col] = [
                    np.clip(sequence, lower, upper)
                    for sequence in sequences
                ]

        return residual_pool

    @staticmethod
    def _sample_residual_blocks(
        sequences: list[np.ndarray],
        output_length: int,
        block_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Create one residual series by sampling contiguous source blocks."""
        sampled_blocks = []
        sampled_length = 0

        while sampled_length < output_length:
            sequence = sequences[int(rng.integers(0, len(sequences)))]
            current_block_size = min(block_size, len(sequence))
            start_max = len(sequence) - current_block_size
            start = int(rng.integers(0, start_max + 1))
            block = sequence[start:start + current_block_size]
            sampled_blocks.append(block)
            sampled_length += len(block)

        sampled = np.concatenate(sampled_blocks)[:output_length].copy()
        sampled -= np.mean(sampled)
        return sampled

    @staticmethod
    def add_block_residual_noise(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        block_size: int = NOISE_BLOCK_SIZE,
        residual_scale: float = NOISE_RESIDUAL_SCALE,
        trend_window: int = NOISE_TREND_WINDOW,
        clip_quantile: float = NOISE_CLIP_QUANTILE,
        max_noise_std_fraction: float = NOISE_MAX_STD_FRACTION,
    ) -> pd.DataFrame:
        """Add signal-specific noise sampled from observed residual blocks.

        This method preserves the non-Gaussian residual distribution and some
        short-range temporal dependence. ``residual_scale`` controls how much
        of the observed residual magnitude is added. A per-experiment cap
        prevents unstable perturbations when a signal has little variation.
        """
        if block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative.")
        if max_noise_std_fraction < 0.0:
            raise ValueError("max_noise_std_fraction must be non-negative.")

        residual_pool = SensorDataAugmentor._residual_noise_pool(
            df=df,
            signal_cols=signal_cols,
            trend_window=trend_window,
            clip_quantile=clip_quantile,
        )
        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()

            for col in signal_cols:
                sequences = residual_pool.get(col, [])
                if not sequences:
                    continue

                values = group[col].to_numpy(dtype=float)
                finite = np.isfinite(values)
                if finite.sum() < 2:
                    continue

                sampled_noise = SensorDataAugmentor._sample_residual_blocks(
                    sequences=sequences,
                    output_length=int(finite.sum()),
                    block_size=block_size,
                    rng=rng,
                )
                sampled_noise *= residual_scale

                signal_std = float(np.std(values[finite], ddof=0))
                noise_std = float(np.std(sampled_noise, ddof=0))
                maximum_noise_std = max_noise_std_fraction * signal_std
                if noise_std > maximum_noise_std and noise_std > 1e-12:
                    sampled_noise *= maximum_noise_std / noise_std

                values[finite] += sampled_noise
                group[col] = values

            augmented_groups.append(group)

        if not augmented_groups:
            return df.copy()
        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

    @staticmethod
    def apply_empirical_time_warping(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        gamma_values: np.ndarray,
        warp_probability: float = TIME_WARP_PROBABILITY,
    ) -> pd.DataFrame:
        """Apply one empirically sampled warp factor per experiment.

        The same gamma is used for every signal in an experiment, preserving
        synchronization between machine and movement channels. Sampling from
        observed group-level gamma values avoids assuming a uniform temporal
        distribution. ``gamma_values=[1.0]`` safely disables warping for a
        group without sufficient event-timing evidence.
        """
        if not 0.0 <= warp_probability <= 1.0:
            raise ValueError("warp_probability must be in [0, 1].")

        gamma_values = np.asarray(gamma_values, dtype=float).reshape(-1)
        gamma_values = gamma_values[
            np.isfinite(gamma_values) & (gamma_values > 0.0)
        ]
        if gamma_values.size == 0:
            gamma_values = np.array([1.0], dtype=float)

        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()
            time_values = group[SensorDataAugmentor.TIME_COL].to_numpy(
                dtype=float
            )
            finite_time = np.isfinite(time_values)
            if (
                len(group) < 3
                or finite_time.sum() < 3
                or not finite_time.all()
                or np.ptp(time_values) <= 1e-12
            ):
                augmented_groups.append(group)
                continue

            gamma = (
                float(rng.choice(gamma_values))
                if rng.random() <= warp_probability
                else 1.0
            )
            if np.isclose(gamma, 1.0):
                augmented_groups.append(group)
                continue

            duration = time_values[-1] - time_values[0]
            normalized_time = (time_values - time_values[0]) / duration
            warped_normalized_time = normalized_time ** gamma
            warped_time = time_values[0] + warped_normalized_time * duration

            for signal_name in signal_cols:
                signal_values = group[signal_name].to_numpy(dtype=float)
                valid = np.isfinite(time_values) & np.isfinite(signal_values)
                if valid.sum() < 3:
                    continue

                unique_time, unique_indices = np.unique(
                    time_values[valid],
                    return_index=True,
                )
                unique_signal = signal_values[valid][unique_indices]
                if len(unique_time) < 3:
                    continue

                warped_values = np.interp(
                    warped_time,
                    unique_time,
                    unique_signal,
                )
                # Preserve missing values from the original signal rather than
                # manufacturing interpolated observations at missing rows.
                warped_values[~np.isfinite(signal_values)] = np.nan
                group[signal_name] = warped_values

            augmented_groups.append(group)

        if not augmented_groups:
            return df.copy()
        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

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
    def apply_signal_specific_scaling(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        scale_std: float = 0.03,
        scale_std_by_signal: dict[str, float] | None = None,
        scaling_probability: float = 0.70,
        maximum_factor_deviation: float = 0.05,
        baseline_fraction: float = 0.05,
    ) -> pd.DataFrame:
        """Apply conservative, signal-specific amplitude scaling.

        Torque channels are multiplied directly. Movement and angle channels
        are scaled around an initial baseline so that the absolute starting
        position is preserved. One positive log-normal factor is sampled per
        signal and experiment and applied to all time points, which preserves
        the temporal profile of the source experiment.

        ``scale_std_by_signal`` takes precedence over ``scale_std``. When it is
        omitted, the empirically selected class defaults are used for known
        channels and ``scale_std`` is retained as a backward-compatible
        fallback for unknown channels.
        """
        if scale_std < 0.0:
            raise ValueError("scale_std must be non-negative.")
        if not 0.0 <= scaling_probability <= 1.0:
            raise ValueError("scaling_probability must be between 0 and 1.")
        if not 0.0 <= maximum_factor_deviation < 1.0:
            raise ValueError(
                "maximum_factor_deviation must be in the interval [0, 1)."
            )
        if not 0.0 < baseline_fraction <= 1.0:
            raise ValueError("baseline_fraction must be in the interval (0, 1].")

        augmented = df.copy()
        torque_cols = set(SensorDataAugmentor._torque_signal_columns(signal_cols))

        configured_std = dict(SensorDataAugmentor.SCALING_STD_BY_SIGNAL)
        if scale_std_by_signal is not None:
            configured_std.update(scale_std_by_signal)

        for _, index in augmented.groupby(
            SensorDataAugmentor.ID_COL, sort=False,
        ).groups.items():
            values = augmented.loc[index, signal_cols].to_numpy(dtype=float)

            for col_idx, col in enumerate(signal_cols):
                signal_scale_std = float(configured_std.get(col, scale_std))
                if not np.isfinite(signal_scale_std) or signal_scale_std < 0.0:
                    raise ValueError(
                        f"Invalid scaling standard deviation for {col!r}: "
                        f"{signal_scale_std}."
                    )
                if signal_scale_std == 0.0 or rng.random() > scaling_probability:
                    continue

                # The mean correction keeps E[factor] approximately equal to 1.
                factor = float(
                    rng.lognormal(
                        mean=-0.5 * signal_scale_std**2,
                        sigma=signal_scale_std,
                    )
                )
                factor = float(
                    np.clip(
                        factor,
                        1.0 - maximum_factor_deviation,
                        1.0 + maximum_factor_deviation,
                    )
                )

                if col in torque_cols:
                    values[:, col_idx] *= factor
                else:
                    finite = np.isfinite(values[:, col_idx])
                    if finite.sum() < 3:
                        continue

                    baseline_count = max(
                        3,
                        int(np.ceil(baseline_fraction * len(values))),
                    )
                    baseline_count = min(baseline_count, len(values))
                    baseline = float(
                        np.nanmedian(values[:baseline_count, col_idx])
                    )
                    if not np.isfinite(baseline):
                        continue

                    dynamic_values = values[:, col_idx] - baseline
                    finite_dynamic = dynamic_values[np.isfinite(dynamic_values)]
                    if len(finite_dynamic) < 3:
                        continue

                    dynamic_amplitude = float(
                        np.nanpercentile(finite_dynamic, 95)
                        - np.nanpercentile(finite_dynamic, 5)
                    )
                    if not np.isfinite(dynamic_amplitude) or dynamic_amplitude <= 1e-12:
                        continue

                    values[:, col_idx] = baseline + factor * dynamic_values

            augmented.loc[index, signal_cols] = values

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

            col_std = group[signal_cols].std(axis=0).fillna(0.0).to_numpy()

            for col_idx, col in enumerate(signal_cols):
                smooth_noise = SensorDataAugmentor._smooth_jitter_sequence(
                    n_rows=n_rows,
                    n_control_points=n_control_points,
                    target_std=jitter_scale * col_std[col_idx],
                    rng=rng,
                )
                group[col] = group[col].to_numpy() + smooth_noise

            augmented_groups.append(group)

        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

    @staticmethod
    def _smooth_jitter_sequence(
        n_rows: int,
        n_control_points: int,
        target_std: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate smooth zero-mean jitter with fixed boundaries and scale.

        Independent standard-normal values define the control-point shape.
        Linear interpolation makes the perturbation temporally smooth. A
        linear endpoint component is removed so that the first and last
        perturbations are zero. A sine-shaped correction then removes the
        remaining mean without changing those boundary values. Finally, the
        sequence is rescaled so that its effective standard deviation equals
        ``target_std``.
        """
        if n_rows < 1:
            raise ValueError("n_rows must be positive.")
        if n_control_points < 2:
            raise ValueError("n_control_points must be at least 2.")
        if not np.isfinite(target_std) or target_std <= 0.0 or n_rows < 4:
            return np.zeros(n_rows, dtype=float)

        control_x = np.linspace(
            0.0,
            float(n_rows - 1),
            min(n_control_points, n_rows),
        )
        row_x = np.arange(n_rows, dtype=float)
        control_noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=len(control_x),
        )
        smooth_noise = np.interp(row_x, control_x, control_noise)

        # Remove the line connecting the two random endpoint values. This
        # preserves the original signal at the beginning and end.
        endpoint_line = np.linspace(
            smooth_noise[0],
            smooth_noise[-1],
            n_rows,
        )
        smooth_noise = smooth_noise - endpoint_line

        # Remove the realization-level mean with a correction that is zero at
        # both boundaries, so endpoint preservation is retained.
        mean_correction_shape = np.sin(
            np.linspace(0.0, np.pi, n_rows)
        )
        correction_mean = float(np.mean(mean_correction_shape))
        if correction_mean > 1e-12:
            smooth_noise = smooth_noise - (
                float(np.mean(smooth_noise))
                / correction_mean
            ) * mean_correction_shape

        effective_std = float(np.std(smooth_noise, ddof=0))
        if not np.isfinite(effective_std) or effective_std <= 1e-12:
            return np.zeros(n_rows, dtype=float)

        smooth_noise = smooth_noise * (target_std / effective_std)

        # Protect the intended invariants against floating-point drift.
        smooth_noise[0] = 0.0
        smooth_noise[-1] = 0.0
        return smooth_noise

    @staticmethod
    def _signal_specific_torque_jitter_std(
        df: pd.DataFrame,
        torque_cols: list[str],
        trend_window: int = JITTER_TREND_WINDOW,
        smoothing_window: int = JITTER_SMOOTHING_WINDOW,
        max_std_fraction: float = JITTER_MAX_STD_FRACTION,
    ) -> dict[str, float]:
        """Estimate a separate smooth-jitter standard deviation per torque.

        The calculation matches the notebook analysis: a rolling-median trend
        is removed, the remaining residual is smoothed with a rolling mean,
        and the median smooth-residual standard deviation across experiments
        becomes the signal's jitter scale.
        """
        if trend_window < 1 or smoothing_window < 1:
            raise ValueError("Jitter windows must be positive integers.")
        if not 0.0 <= max_std_fraction:
            raise ValueError("max_std_fraction must be non-negative.")

        smooth_std_by_signal = {col: [] for col in torque_cols}

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL)

            for col in torque_cols:
                values = group[col].to_numpy(dtype=float)
                finite_count = int(np.isfinite(values).sum())
                if finite_count < 2:
                    continue

                series = pd.Series(values).interpolate(limit_direction="both")
                local_trend_window = min(trend_window, len(series))
                if local_trend_window % 2 == 0:
                    local_trend_window -= 1
                local_trend_window = max(local_trend_window, 1)

                long_term_trend = series.rolling(
                    window=local_trend_window,
                    center=True,
                    min_periods=1,
                ).median()
                detrended = series - long_term_trend
                smooth_residual = detrended.rolling(
                    window=min(smoothing_window, len(series)),
                    center=True,
                    min_periods=1,
                ).mean()
                smooth_std = float(np.nanstd(smooth_residual.to_numpy()))
                if np.isfinite(smooth_std):
                    smooth_std_by_signal[col].append(smooth_std)

        global_std = df[torque_cols].std(axis=0, skipna=True, ddof=0)
        jitter_std = {}
        for col in torque_cols:
            observed_values = smooth_std_by_signal[col]
            observed_std = (
                float(np.nanmedian(observed_values))
                if observed_values
                else 0.0
            )
            signal_std = float(global_std.get(col, 0.0))
            if not np.isfinite(signal_std):
                signal_std = 0.0
            jitter_std[col] = min(
                max(observed_std, 0.0),
                max_std_fraction * signal_std,
            )

        return jitter_std

    @staticmethod
    def apply_signal_specific_jittering(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        jitter_std_by_signal: dict[str, float],
        n_control_points: int = 64,
    ) -> pd.DataFrame:
        """Add smooth jitter using a separate absolute scale per signal."""
        augmented_groups = []

        for _, group in df.groupby(SensorDataAugmentor.ID_COL, sort=False):
            group = group.sort_values(SensorDataAugmentor.TIME_COL).copy()
            n_rows = len(group)
            if n_rows < 2:
                augmented_groups.append(group)
                continue

            for col in signal_cols:
                jitter_std = float(jitter_std_by_signal.get(col, 0.0))
                if not np.isfinite(jitter_std) or jitter_std <= 0.0:
                    continue
                smooth_noise = SensorDataAugmentor._smooth_jitter_sequence(
                    n_rows=n_rows,
                    n_control_points=n_control_points,
                    target_std=jitter_std,
                    rng=rng,
                )
                group[col] = group[col].to_numpy(dtype=float) + smooth_noise

            augmented_groups.append(group)

        return pd.concat(augmented_groups, ignore_index=True)[df.columns]

    @staticmethod
    def apply_torque_jittering(
        df: pd.DataFrame,
        signal_cols: list[str],
        rng: np.random.Generator,
        jitter_scale: float | None = None,
        n_control_points: int = 64,
    ) -> pd.DataFrame:
        torque_cols = SensorDataAugmentor._torque_signal_columns(signal_cols)
        if not torque_cols:
            return df.copy()

        if jitter_scale is None:
            jitter_std_by_signal = (
                SensorDataAugmentor._signal_specific_torque_jitter_std(
                    df,
                    torque_cols,
                )
            )
            return SensorDataAugmentor.apply_signal_specific_jittering(
                df=df,
                signal_cols=torque_cols,
                rng=rng,
                jitter_std_by_signal=jitter_std_by_signal,
                n_control_points=n_control_points,
            )

        # Backward-compatible fixed-ratio mode for explicit callers.
        return SensorDataAugmentor.apply_jittering(
            df=df,
            signal_cols=torque_cols,
            rng=rng,
            jitter_scale=jitter_scale,
            n_control_points=n_control_points,
        )
