import numpy as np
import logging

from src.pipeline.rf_augmentation.rf_group_feature_selector import RFGroupFeatureSelector

logger = logging.getLogger(__name__)


class RFAugmentationGenerator:
    """
    Generates synthetic RF-based samples by:
    1. Sampling existing feature vectors
    2. Adding controlled noise
    3. Predicting geometry using trained RF models
    """

    @staticmethod
    def generate(
        X_main: np.ndarray,
        model_main,
        model_secondary,
        X_secondary: np.ndarray | None = None,
        n_new_samples: int = 0,
        noise_scale: float = 0.01,
        use_feature_std: bool = True,
        random_state: int = 42,
        sample_within_group_range: bool = True,
        return_selection_details: bool = False,
    ):
        logger.info("Starting RF-based data augmentation")

        rng = np.random.default_rng(random_state)

        if X_secondary is None:
            X_secondary = X_main

        X_main = X_main.astype(np.float64, copy=False)
        X_secondary = X_secondary.astype(np.float64, copy=False)

        n_original, n_features_main = X_main.shape
        _, n_features_sec = X_secondary.shape

        if n_new_samples <= 0:
            logger.info("No augmentation requested (n_new_samples <= 0). Returning originals.")
            if return_selection_details:
                return X_main, X_secondary, None, None, {}
            return X_main, X_secondary, None, None

        # -------------------------
        # Step 1: Sample base rows
        # -------------------------
        indices = rng.choice(n_original, size=n_new_samples, replace=True)
        X_main_sampled = X_main[indices]
        X_secondary_sampled = X_secondary[indices]

        # -------------------------
        # Step 2: Select synthetic feature values
        # -------------------------
        if sample_within_group_range:
            selection_details = (
                RFGroupFeatureSelector.sample_independent_features_within_group_range(
                    X_main=X_main,
                    X_secondary=X_secondary,
                    n_new_samples=n_new_samples,
                    rng=rng,
                )
            )
            X_main_sampled = selection_details["X_main_actual"]
            X_secondary_sampled = selection_details["X_secondary_actual"]
            X_main_new = selection_details["X_main_selected"]
            X_secondary_new = selection_details["X_secondary_selected"]
        elif use_feature_std:
            feature_std_main = np.std(X_main, axis=0)
            noise_main = rng.normal(
                loc=0.0,
                scale=noise_scale * (feature_std_main + 1e-8),
                size=X_main_sampled.shape
            )

            if X_secondary is X_main:
                noise_secondary = noise_main
            else:
                feature_std_secondary = np.std(X_secondary, axis=0)
                noise_secondary = rng.normal(
                    loc=0.0,
                    scale=noise_scale * (feature_std_secondary + 1e-8),
                    size=X_secondary_sampled.shape
                )

            X_main_new = X_main_sampled + noise_main
            X_secondary_new = X_secondary_sampled + noise_secondary
            feature_min_main = np.min(X_main, axis=0)
            feature_max_main = np.max(X_main, axis=0)
            feature_min_secondary = np.min(X_secondary, axis=0)
            feature_max_secondary = np.max(X_secondary, axis=0)
            selection_details = {
                "sampled_indices": indices,
                "X_main_actual": X_main_sampled,
                "X_main_selected": X_main_new,
                "X_main_group_min": feature_min_main,
                "X_main_group_max": feature_max_main,
                "X_secondary_actual": X_secondary_sampled,
                "X_secondary_selected": X_secondary_new,
                "X_secondary_group_min": feature_min_secondary,
                "X_secondary_group_max": feature_max_secondary,
                "selection_method": "std_noise_around_sampled_row",
            }
        else:
            noise_main = rng.normal(0.0, noise_scale, size=X_main_sampled.shape)

            if X_secondary is X_main:
                noise_secondary = noise_main
            else:
                noise_secondary = rng.normal(0.0, noise_scale, size=X_secondary_sampled.shape)

            X_main_new = X_main_sampled + noise_main
            X_secondary_new = X_secondary_sampled + noise_secondary
            feature_min_main = np.min(X_main, axis=0)
            feature_max_main = np.max(X_main, axis=0)
            feature_min_secondary = np.min(X_secondary, axis=0)
            feature_max_secondary = np.max(X_secondary, axis=0)
            selection_details = {
                "sampled_indices": indices,
                "X_main_actual": X_main_sampled,
                "X_main_selected": X_main_new,
                "X_main_group_min": feature_min_main,
                "X_main_group_max": feature_max_main,
                "X_secondary_actual": X_secondary_sampled,
                "X_secondary_selected": X_secondary_new,
                "X_secondary_group_min": feature_min_secondary,
                "X_secondary_group_max": feature_max_secondary,
                "selection_method": "fixed_noise_around_sampled_row",
            }

        # -------------------------
        # Step 3: Predict geometry
        # -------------------------
        logger.info("Predicting geometry for augmented samples")

        y_main_new = model_main.predict(X_main_new)
        y_secondary_new = model_secondary.predict(X_secondary_new)

        # -------------------------
        # Step 4: Combine with original
        # -------------------------
        X_main_aug = np.vstack([X_main, X_main_new])
        X_secondary_aug = np.vstack([X_secondary, X_secondary_new])

        logger.info(
            "Augmentation summary → original: %s, new: %s, total: %s",
            n_original,
            n_new_samples,
            X_main_aug.shape[0],
        )

        if return_selection_details:
            selection_details["sample_within_group_range"] = sample_within_group_range
            return X_main_aug, X_secondary_aug, y_main_new, y_secondary_new, selection_details

        return X_main_aug, X_secondary_aug, y_main_new, y_secondary_new
