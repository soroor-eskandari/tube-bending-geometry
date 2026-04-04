from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


class RFModelTrainer:

    @staticmethod
    def train(X_rf, y_main, y_secondary):

        # -------------------------
        # SINGLE consistent split
        # -------------------------
        X_train, X_test, y_main_train, y_main_test, y_sec_train, y_sec_test = train_test_split(
            X_rf,
            y_main,
            y_secondary,
            test_size=0.2,
            random_state=42
        )

        # -------------------------
        # Train models
        # -------------------------
        model_main = RandomForestRegressor(n_estimators=200, random_state=42)
        model_sec = RandomForestRegressor(n_estimators=200, random_state=42)

        model_main.fit(X_train, y_main_train)
        model_sec.fit(X_train, y_sec_train)

        return {
            "model_main": model_main,
            "model_secondary": model_sec,
            "X_train": X_train,
            "X_test": X_test,
            "y_main_train": y_main_train,
            "y_main_test": y_main_test,
            "y_sec_train": y_sec_train,
            "y_sec_test": y_sec_test,
        }

    # -------------------------
    # Prediction
    # -------------------------
    @staticmethod
    def predict(models, X):
        return {
            "y_main_pred": models["model_main"].predict(X),
            "y_sec_pred": models["model_secondary"].predict(X),
        }

    # -------------------------
    # Save models
    # -------------------------
    @staticmethod
    def save(models, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(models["model_main"], output_dir / "rf_main.pkl")
        joblib.dump(models["model_secondary"], output_dir / "rf_secondary.pkl")

    # -------------------------
    # Load models
    # -------------------------
    @staticmethod
    def load(output_dir: Path):
        return {
            "model_main": joblib.load(output_dir / "rf_main.pkl"),
            "model_secondary": joblib.load(output_dir / "rf_secondary.pkl"),
        }