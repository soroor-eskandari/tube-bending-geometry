import pandas as pd
import numpy as np
import logging

from scipy.stats import skew, kurtosis
from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class RFDatasetBuilder:

    # ============================================================
    # PUBLIC ENTRY
    # ============================================================
    @staticmethod
    def build(
        machine_movement__df: pd.DataFrame,
        geometry_df: pd.DataFrame,
        bending_df: pd.DataFrame,
        main_selected_features: list | None,
        secondary_selected_features: list | None,
        return_experiment_ids: bool = False,
    ):
        if main_selected_features is not None:
            main_selected_features = set(main_selected_features)

        if secondary_selected_features is not None:
            secondary_selected_features = set(secondary_selected_features)

        # -------------------------
        # Extract features
        # -------------------------
        X_main_df = RFDatasetBuilder._extract_features(
            machine_movement__df,
            selected_features=main_selected_features,
            tag="MAIN"
        )

        X_secondary_df = RFDatasetBuilder._extract_features(
            machine_movement__df,
            selected_features=secondary_selected_features,
            tag="SECONDARY"
        )

        # -------------------------
        # Prepare bending features 
        # -------------------------
        X_bending_df = bending_df.copy()

        if "Experiment_ID" not in X_bending_df.columns:
            raise ValueError("bending_df must contain 'Experiment_ID'")
        
        # -------------------------
        # Merge bending features
        # -------------------------
        X_main_df = X_main_df.merge(X_bending_df, on="Experiment_ID", how="left")
        X_secondary_df = X_secondary_df.merge(X_bending_df, on="Experiment_ID", how="left")

        # Fill missing values (important)
        X_main_df = X_main_df.fillna(0.0)
        X_secondary_df = X_secondary_df.fillna(0.0)

        # -------------------------
        # Prepare targets
        # -------------------------
        y_main, y_secondary, aligned_ids = RFDatasetBuilder._prepare_geometry_targets(
            geometry_df,
            X_main_df["Experiment_ID"].tolist(),
        )

        # -------------------------
        # Align
        # -------------------------
        X_main_df = X_main_df[X_main_df["Experiment_ID"].isin(aligned_ids)]
        X_secondary_df = X_secondary_df[X_secondary_df["Experiment_ID"].isin(aligned_ids)]

        # -------------------------
        # Convert to numpy
        # -------------------------
        X_main_df_no_id = X_main_df.drop(columns=["Experiment_ID"])
        X_secondary_df_no_id = X_secondary_df.drop(columns=["Experiment_ID"])

        X_main = X_main_df_no_id.to_numpy(dtype=np.float32)
        X_secondary = X_secondary_df_no_id.to_numpy(dtype=np.float32)

        feature_names_main = X_main_df_no_id.columns.tolist()
        feature_names_secondary = X_secondary_df_no_id.columns.tolist()

        result = (
            X_main,
            X_secondary,
            y_main,
            y_secondary,
            feature_names_main,
            feature_names_secondary,
        )

        if return_experiment_ids:
            return (*result, aligned_ids)

        return result

    # ============================================================
    # FEATURE EXTRACTION (FILTERED)
    # ============================================================
    @staticmethod
    def _extract_features(
        df: pd.DataFrame,
        selected_features: set | None,
        tag: str
    ) -> pd.DataFrame:

        df = df.copy()
        signal_cols = [c for c in df.columns if c not in ["Experiment_ID", "Time_[s]"]]

        rows = []

        for exp_id, group in df.groupby("Experiment_ID"):

            group = (
                group.sort_values("Time_[s]")
                .drop_duplicates("Time_[s]")
                .reset_index(drop=True)
            )

            row = {"Experiment_ID": exp_id}

            for col in signal_cols:

                signal = group[col].values.astype(float)
                signal = np.nan_to_num(signal)

                if len(signal) < 5:
                    continue

                features = RFDatasetBuilder._compute_manual_features(signal)

                for feat_name, value in features.items():
                    if selected_features is None or feat_name in selected_features:
                        row[f"{col}_{feat_name}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    # ============================================================
    # MANUAL FEATURES (23)
    # ============================================================
    @staticmethod
    def _compute_manual_features(signal: np.ndarray) -> dict:

        if len(signal) < 2:
            return {}

        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)

        if np.allclose(signal, signal[0]):
            skew_val = 0.0
            kurt_val = 0.0
        else:
            skew_val = skew(signal)
            kurt_val = kurtosis(signal)

        diff = np.diff(signal)

        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "median": np.median(signal),
            "skew": skew_val,
            "kurtosis": kurt_val,
            "iqr": np.percentile(signal, 75) - np.percentile(signal, 25),
            "range": max_val - min_val,

            "energy": np.sum(signal ** 2),
            "rms": np.sqrt(np.mean(signal ** 2)),
            "sum": np.sum(signal),
            "mean_abs": np.mean(np.abs(signal)),

            "variance": np.var(signal),
            "mad": np.mean(np.abs(signal - mean_val)),
            "coeff_var": std_val / mean_val if mean_val != 0 else 0.0,

            "p10": np.percentile(signal, 10),
            "p25": np.percentile(signal, 25),
            "p75": np.percentile(signal, 75),
            "p90": np.percentile(signal, 90),

            "zero_crossings": np.sum(np.diff(np.sign(signal)) != 0),
            "peak_count": np.sum(
                (signal[1:-1] > signal[:-2]) &
                (signal[1:-1] > signal[2:])
            ),
            "slope_mean": np.mean(diff),
        }

    # ============================================================
    # TARGET PREPARATION
    # ============================================================
    @staticmethod
    def _prepare_geometry_targets(
        geometry_df: pd.DataFrame,
        experiment_ids: list,
    ):
        df = geometry_df.copy()

        required_cols = [
            "Experiment_ID",
            "Angle[degree]ORDistance[mm]",
            "Main-axis [mm]",
            "Secondary-axis [mm]",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column in geometry_df: {col}")

        df = df[df["Experiment_ID"].isin(experiment_ids)]
        df = df.sort_values(["Experiment_ID", "Angle[degree]ORDistance[mm]"])

        grouped = df.groupby("Experiment_ID")

        lengths = [
            len(grouped.get_group(eid))
            for eid in experiment_ids
            if eid in grouped.groups
        ]

        if len(lengths) == 0:
            raise ValueError("No matching experiments between features and geometry")

        target_len = 45
        min_len = min(min(lengths), target_len)

        y_main, y_secondary, aligned_ids = [], [], []

        for exp_id in experiment_ids:

            if exp_id not in grouped.groups:
                continue

            group = grouped.get_group(exp_id).head(min_len)

            y_main.append(group["Main-axis [mm]"].to_numpy(dtype=np.float32))
            y_secondary.append(group["Secondary-axis [mm]"].to_numpy(dtype=np.float32))
            aligned_ids.append(exp_id)

        if len(y_main) == 0:
            raise ValueError("No valid geometry sequences")

        y_main = np.vstack(y_main)
        y_secondary = np.vstack(y_secondary)

        return y_main, y_secondary, aligned_ids
