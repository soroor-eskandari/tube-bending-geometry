import numpy as np
import logging

from src.pipeline.rf_augmentation.rf_group_feature_selector import RFGroupFeatureSelector

logger = logging.getLogger(__name__)


class RFAugmentationGenerator:
    """
    Generates synthetic RF-based samples by:
    1. Sampling existing feature vectors for traceability
    2. Selecting random feature values within each group's real feature intervals
    3. Predicting geometry using trained RF models
    """

    FEATURE_SAMPLING_MODES = {
        "within-group-interpolation",
    }

    @staticmethod
    def generate(
        X_main: np.ndarray,
        model_main,
        model_secondary,
        X_secondary: np.ndarray | None = None,
        X_main_reference: np.ndarray | None = None,
        X_secondary_reference: np.ndarray | None = None,
        reference_experiment_ids: list | None = None,
        n_new_samples: int = 0,
        noise_scale: float = 0.01,
        use_feature_std: bool = True,
        random_state: int = 42,
        feature_sampling_mode: str = "within-group-interpolation",
        return_selection_details: bool = False,
    ):
        rng = np.random.default_rng(random_state)
        if feature_sampling_mode not in RFAugmentationGenerator.FEATURE_SAMPLING_MODES:
            raise ValueError(
                "feature_sampling_mode must be one of "
                f"{sorted(RFAugmentationGenerator.FEATURE_SAMPLING_MODES)}. "
                f"Got: {feature_sampling_mode}"
            )

        if X_secondary is None:
            X_secondary = X_main

        X_main = X_main.astype(np.float64, copy=False)
        X_secondary = X_secondary.astype(np.float64, copy=False)

        n_original, n_features_main = X_main.shape
        _, n_features_sec = X_secondary.shape

        if n_new_samples <= 0:
            if return_selection_details:
                return X_main, X_secondary, None, None, {}
            return X_main, X_secondary, None, None

        # -------------------------
        # Step 1: Select synthetic feature values
        # -------------------------
        if feature_sampling_mode == "within-group-interpolation":
            selection_details = (
                RFGroupFeatureSelector.interpolate_within_group(
                    X_main=X_main,
                    X_secondary=X_secondary,
                    n_new_samples=n_new_samples,
                    rng=rng,
                    X_main_reference=X_main_reference,
                    X_secondary_reference=X_secondary_reference,
                    reference_experiment_ids=reference_experiment_ids,
                )
            )
            X_main_sampled = selection_details["X_main_actual"]
            X_secondary_sampled = selection_details["X_secondary_actual"]
            X_main_new = selection_details["X_main_selected"]
            X_secondary_new = selection_details["X_secondary_selected"]
        else:
            raise ValueError(f"Unsupported feature_sampling_mode: {feature_sampling_mode}")

        # -------------------------
        # Step 2: Predict geometry
        # -------------------------
        y_main_new = model_main.predict(X_main_new)
        y_secondary_new = model_secondary.predict(X_secondary_new)

        # -------------------------
        # Step 3: Combine with original
        # -------------------------
        X_main_aug = np.vstack([X_main, X_main_new])
        X_secondary_aug = np.vstack([X_secondary, X_secondary_new])

        if return_selection_details:
            selection_details["feature_sampling_mode"] = feature_sampling_mode
            return X_main_aug, X_secondary_aug, y_main_new, y_secondary_new, selection_details

        return X_main_aug, X_secondary_aug, y_main_new, y_secondary_new
