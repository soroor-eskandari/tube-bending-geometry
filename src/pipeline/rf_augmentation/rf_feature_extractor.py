import numpy as np
import pandas as pd
import tsfel
import logging

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)

TSFEL_SKIP_FEATURES = {"kurtosis", "skewness"}


def _strip_tsfel_features(cfg: dict, feature_names: set[str]) -> dict:
    feature_names = {name.lower() for name in feature_names}
    for _, domain_features in cfg.items():
        if not isinstance(domain_features, dict):
            continue
        for feat_name in list(domain_features.keys()):
            if feat_name.lower() in feature_names:
                domain_features.pop(feat_name, None)
    return cfg


class TimeSeriesFeatureExtractor:

    @staticmethod
    @log_function
    def extract_features(
        df: pd.DataFrame,
        signal_cols: list,
        tsfel_domains: list = ("statistical", "spectral"),
        tsfel_included: bool = True
    ) -> pd.DataFrame:

        if tsfel_included:
            logger.info("Starting feature extraction (TSFEL + manual)")
        else:
            logger.info("Starting feature extraction (manual only)")

        df = df.copy()
        df["Experiment_ID"] = df["Experiment_ID"].astype(int)

        cfg = None
        if tsfel_included:
            cfg = tsfel.get_features_by_domain(list(tsfel_domains))
            cfg = _strip_tsfel_features(cfg, TSFEL_SKIP_FEATURES)

        features = []

        # -------------------------
        # GROUP PER EXPERIMENT
        # -------------------------
        for exp_id, group in df.groupby("Experiment_ID"):

            group = (
                group
                .sort_values("Time_[s]")
                .drop_duplicates("Time_[s]")
                .reset_index(drop=True)
            )

            row = {"Experiment_ID": exp_id}

            time = group["Time_[s]"].values

            # Sampling frequency
            if len(time) > 1:
                dt = np.median(np.diff(time))
                fs = 1.0 / dt if dt > 0 else 1.0
            else:
                fs = 1.0

            # -------------------------
            # PER SIGNAL
            # -------------------------
            for col in signal_cols:

                if col not in group.columns:
                    logger.warning(f"Missing column: {col}")
                    continue

                signal = group[col].values.astype(float)
                signal = np.nan_to_num(signal)

                # Skip very small signals
                if len(signal) < 5:
                    continue

                # ====================================
                # 1. TSFEL FEATURES
                # ====================================
                if tsfel_included:
                    try:
                        tsfel_df = tsfel.time_series_features_extractor(
                            cfg,
                            signal,
                            fs=fs,
                            verbose=0
                        )

                        tsfel_row = tsfel_df.iloc[0]

                        for f_name, value in tsfel_row.items():
                            row[f"{col}_tsfel_{f_name}"] = value

                    except Exception as e:
                        logger.warning(f"TSFEL failed | Exp={exp_id}, Col={col}, Error={e}")

                # ====================================
                # 2. MANUAL FEATURES (23)
                # ====================================

                # Basic stats
                row[f"{col}_manual_min"] = np.min(signal)
                row[f"{col}_manual_max"] = np.max(signal)
                row[f"{col}_manual_mean"] = np.mean(signal)
                row[f"{col}_manual_std"] = np.std(signal)
                row[f"{col}_manual_median"] = np.median(signal)
                if np.allclose(signal, signal[0], rtol=1e-6, atol=1e-8):
                    row[f"{col}_manual_skew"] = 0.0
                    row[f"{col}_manual_kurtosis"] = 0.0
                else:
                    row[f"{col}_manual_skew"] = skew(signal)
                    row[f"{col}_manual_kurtosis"] = kurtosis(signal)


                # Percentiles
                row[f"{col}_manual_p10"] = np.percentile(signal, 10)
                row[f"{col}_manual_p25"] = np.percentile(signal, 25)
                row[f"{col}_manual_p75"] = np.percentile(signal, 75)
                row[f"{col}_manual_p90"] = np.percentile(signal, 90)

                # Range & IQR
                row[f"{col}_manual_range"] = np.max(signal) - np.min(signal)
                row[f"{col}_manual_iqr"] = (
                    np.percentile(signal, 75) - np.percentile(signal, 25)
                )

                # Energy
                row[f"{col}_manual_energy"] = np.sum(signal ** 2)

                # Zero crossings
                row[f"{col}_manual_zero_crossings"] = np.sum(np.diff(np.sign(signal)) != 0)

                # Peaks
                peaks, _ = find_peaks(signal)
                row[f"{col}_manual_num_peaks"] = len(peaks)

                # Mean absolute diff
                row[f"{col}_manual_mean_abs_diff"] = np.mean(np.abs(np.diff(signal)))

                # RMS
                row[f"{col}_manual_rms"] = np.sqrt(np.mean(signal ** 2))

                # Variance
                row[f"{col}_manual_var"] = np.var(signal)

                # Signal sum
                row[f"{col}_manual_sum"] = np.sum(signal)

                # Signal length
                row[f"{col}_manual_length"] = len(signal)

                # Max absolute
                row[f"{col}_manual_max_abs"] = np.max(np.abs(signal))

            features.append(row)

        df_features = pd.DataFrame(features)

        logger.info(f"Feature extraction completed: {df_features.shape}")

        return df_features
