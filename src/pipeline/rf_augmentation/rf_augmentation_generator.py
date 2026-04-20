import numpy as np
import logging

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
        random_state: int = 42
    ):
        logger.info("Starting RF-based data augmentation")

        rng = np.random.default_rng(random_state)

        if X_secondary is None:
            X_secondary = X_main

        n_original, n_features_main = X_main.shape
        _, n_features_sec = X_secondary.shape

        if n_new_samples <= 0:
            logger.info("No augmentation requested (n_new_samples <= 0). Returning originals.")
            return X_main, X_secondary, None, None

        # -------------------------
        # Step 1: Sample base rows
        # -------------------------
        indices = rng.choice(n_original, size=n_new_samples, replace=True)
        X_main_sampled = X_main[indices]
        X_secondary_sampled = X_secondary[indices]

        # -------------------------
        # Step 2: Add noise
        # -------------------------
        if use_feature_std:
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
        else:
            noise_main = rng.normal(0.0, noise_scale, size=X_main_sampled.shape)

            if X_secondary is X_main:
                noise_secondary = noise_main
            else:
                noise_secondary = rng.normal(0.0, noise_scale, size=X_secondary_sampled.shape)

        X_main_new = X_main_sampled + noise_main
        X_secondary_new = X_secondary_sampled + noise_secondary

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

        return X_main_aug, X_secondary_aug, y_main_new, y_secondary_new