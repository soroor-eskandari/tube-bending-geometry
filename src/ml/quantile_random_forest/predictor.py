import numpy as np
import pandas as pd


class QRFPredictor:
    def __init__(self, models):
        self.models = models

    def predict_on_grid(self, angles):
        """
        angles: list or array of angle values (must match training feature space)
        """

        # Ensure angles is 1D array
        angles = np.array(angles)

        # ✅ IMPORTANT: use DataFrame with correct column name
        X_pred = pd.DataFrame({
            "Angle[degree]ORDistance[mm]": angles
        })

        # Predict
        sec_preds = self.models["secondary"].predict(X_pred)
        main_preds = self.models["main"].predict(X_pred)

        # Safety check (optional but useful)
        assert len(angles) == len(sec_preds["lower"]) == len(main_preds["lower"]), \
            "Length mismatch in predictions"

        return pd.DataFrame({
            "angle": angles,

            "sec_lower": sec_preds["lower"],
            "sec_median": sec_preds["median"],
            "sec_upper": sec_preds["upper"],

            "main_lower": main_preds["lower"],
            "main_median": main_preds["median"],
            "main_upper": main_preds["upper"],
        })