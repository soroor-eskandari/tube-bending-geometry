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
        X_rf: np.ndarray,
        model_main,
        model_secondary,
        n_samples: int = 1000,
        noise_scale: float = 0.01,
        use_feature_std: bool = True,
        random_state: int = 42
    ):
        """
        Generate augmented dataset.

        Parameters
        ----------
        X_rf : np.ndarray
            Original feature matrix (n_samples, n_features)

        model_main : trained model
            RF model for main axis

        model_secondary : trained model
            RF model for secondary axis

        n_samples : int
            Number of new samples to generate

        noise_scale : float
            Noise intensity factor

        use_feature_std : bool
            If True → scale noise per feature std
            If False → uniform noise

        random_state : int
            Reproducibility

        Returns
        -------
        X_new : np.ndarray
        y_main_new : np.ndarray
        y_secondary_new : np.ndarray
        """

        logger.info("Starting RF-based data augmentation")

        rng = np.random.default_rng(random_state)

        n_original, n_features = X_rf.shape

        # -------------------------
        # Step 1: Sample base rows
        # -------------------------
        indices = rng.choice(n_original, size=n_samples, replace=True)
        X_sampled = X_rf[indices]

        # -------------------------
        # Step 2: Add noise
        # -------------------------
        if use_feature_std:
            feature_std = np.std(X_rf, axis=0)
            noise = rng.normal(
                loc=0.0,
                scale=noise_scale * (feature_std + 1e-8),
                size=X_sampled.shape
            )
        else:
            noise = rng.normal(
                loc=0.0,
                scale=noise_scale,
                size=X_sampled.shape
            )

        X_new = X_sampled + noise

        # -------------------------
        # Step 3: Predict geometry
        # -------------------------
        logger.info("Predicting geometry for augmented samples")

        y_main_new = model_main.predict(X_new)
        y_secondary_new = model_secondary.predict(X_new)

        # -------------------------
        # Step 4: Basic sanity check
        # -------------------------
        if np.isnan(X_new).any():
            logger.warning("NaNs detected in X_new")

        if np.isnan(y_main_new).any():
            logger.warning("NaNs detected in y_main_new")

        if np.isnan(y_secondary_new).any():
            logger.warning("NaNs detected in y_secondary_new")

        logger.info("Augmentation completed successfully")

        return X_new, y_main_new, y_secondary_new