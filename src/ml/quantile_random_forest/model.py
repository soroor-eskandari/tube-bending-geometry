from sklearn_quantile import RandomForestQuantileRegressor


class QRFModel:
    def __init__(self, quantiles, **rf_params):
        """
        quantiles: list of quantiles, e.g. [0.05, 0.5, 0.95]
        rf_params: parameters for RandomForestQuantileRegressor
        """
        self.quantiles = quantiles

        self.model = RandomForestQuantileRegressor(
            q=quantiles,
            **rf_params
        )

    def fit(self, X, y):
        """
        Train the QRF model
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict quantiles for input X

        Returns:
            dict with keys: lower, median, upper
        """
        preds = self.model.predict(X)

        # --- FIX: handle shape inconsistency ---
        # Some implementations return (n_quantiles, n_samples)
        # We want (n_samples, n_quantiles)
        if preds.shape[0] == len(self.quantiles):
            preds = preds.T

        # --- Safety check ---
        if preds.shape[1] != len(self.quantiles):
            raise ValueError(
                f"Unexpected prediction shape {preds.shape}, "
                f"expected second dimension = {len(self.quantiles)}"
            )

        # --- Extract quantiles ---
        return {
            "lower": preds[:, 0],
            "median": preds[:, 1],
            "upper": preds[:, 2],
        }