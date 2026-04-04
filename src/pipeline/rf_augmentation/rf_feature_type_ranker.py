import re
import joblib
import logging
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class RFFeatureTypeRanker:

    @staticmethod
    def _extract_feature_type(feature_name: str) -> str:
        """
        Extract feature type from full feature name.
        Supports TSFEL and manually engineered features.
        """

        # TSFEL features
        if "tsfel" in feature_name:
            match = re.search(r"tsfel_\d+_(.*)", feature_name)
            if match:
                feat = match.group(1)

                # Remove frequency suffix (e.g., _7.74Hz)
                feat = re.sub(r"_\d+(\.\d+)?Hz", "", feat)

                return feat.strip()

        # Manual features → last token
        return feature_name.split("_")[-1]


    @staticmethod
    @log_function
    def rank_feature_types_dual(
        X,
        y_main,
        y_secondary,
        model_output_dir: Path,
        feature_names: list = None,
        n_estimators: int = 300,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Train two RF models (main & secondary) and compute feature-type importance.
        Accepts both DataFrame and ndarray.
        """

        logger.info("Starting dual feature type ranking (main + secondary)")

        # -------------------------
        # Ensure DataFrame
        # -------------------------
        if isinstance(X, np.ndarray):
            logger.warning("X is ndarray → converting to DataFrame")

            if feature_names is None:
                raise ValueError(
                    "feature_names must be provided when X is numpy array"
                )

            X = pd.DataFrame(X, columns=feature_names)

        elif not isinstance(X, pd.DataFrame):
            raise TypeError("X must be pandas DataFrame or numpy ndarray")

        # -------------------------
        # Feature names
        # -------------------------
        feature_names = X.columns.tolist()
        logger.info(f"Number of input features: {len(feature_names)}")

        # -------------------------
        # Train RF models
        # -------------------------
        rf_main = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        rf_secondary = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        rf_main.fit(X, y_main)
        rf_secondary.fit(X, y_secondary)

        logger.info("Both Random Forest models trained successfully")

        # -------------------------
        # Helper: build importance df
        # -------------------------
        def build_importance_df(importances):
            df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            })

            df["feature_type"] = df["feature"].apply(
                RFFeatureTypeRanker._extract_feature_type
            )

            df_type = (
                df.groupby("feature_type", as_index=False)["importance"]
                .sum()
                .sort_values(by="importance", ascending=False)
            )

            return df_type

        # -------------------------
        # Compute feature-type importance
        # -------------------------
        df_main = build_importance_df(rf_main.feature_importances_)
        df_secondary = build_importance_df(rf_secondary.feature_importances_)

        logger.info(f"Number of feature types (main): {df_main.shape[0]}")
        logger.info(f"Number of feature types (secondary): {df_secondary.shape[0]}")

        # -------------------------
        # Save models
        # -------------------------
        model_output_dir.mkdir(parents=True, exist_ok=True)

        main_model_path = model_output_dir / "rf_feature_type_ranker_main.joblib"
        secondary_model_path = model_output_dir / "rf_feature_type_ranker_secondary.joblib"

        joblib.dump(rf_main, main_model_path)
        joblib.dump(rf_secondary, secondary_model_path)

        logger.info(f"Main RF model saved at: {main_model_path}")
        logger.info(f"Secondary RF model saved at: {secondary_model_path}")

        return df_main, df_secondary, rf_main, rf_secondary