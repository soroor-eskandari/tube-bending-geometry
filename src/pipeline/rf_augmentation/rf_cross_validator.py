from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np
import joblib


class RFModelTrainer:

    # -------------------------
    # Cross-validation training
    # -------------------------
    @staticmethod
    def train_cv(X_rf, y_main, y_secondary, n_splits=5, groups=None, random_state=42):

        # -------------------------
        # Choose splitting strategy
        # -------------------------
        if groups is not None:
            splitter = GroupKFold(n_splits=n_splits)
            split_iterator = splitter.split(X_rf, y_main, groups)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iterator = splitter.split(X_rf)

        fold_results = []

        # -------------------------
        # Iterate over folds
        # -------------------------
        for fold_idx, (train_idx, test_idx) in enumerate(split_iterator):

            # Split data
            X_train = X_rf.iloc[train_idx]
            X_test = X_rf.iloc[test_idx]

            y_main_train = y_main.iloc[train_idx]
            y_main_test = y_main.iloc[test_idx]

            y_sec_train = y_secondary.iloc[train_idx]
            y_sec_test = y_secondary.iloc[test_idx]

            # -------------------------
            # Train models
            # -------------------------
            model_main = RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1
            )

            model_sec = RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1
            )

            model_main.fit(X_train, y_main_train)
            model_sec.fit(X_train, y_sec_train)

            # -------------------------
            # Predictions
            # -------------------------
            y_main_pred = model_main.predict(X_test)
            y_sec_pred = model_sec.predict(X_test)

            # -------------------------
            # Evaluation (RMSE)
            # -------------------------
            rmse_main = np.sqrt(mean_squared_error(y_main_test, y_main_pred))
            rmse_sec = np.sqrt(mean_squared_error(y_sec_test, y_sec_pred))

            # -------------------------
            # Store results
            # -------------------------
            fold_results.append({
                "fold": fold_idx,
                "model_main": model_main,
                "model_secondary": model_sec,
                "rmse_main": rmse_main,
                "rmse_secondary": rmse_sec,
                "X_test": X_test,
                "y_main_test": y_main_test,
                "y_sec_test": y_sec_test,
                "y_main_pred": y_main_pred,
                "y_sec_pred": y_sec_pred,
            })

        return fold_results

    # -------------------------
    # Predict using a single model set
    # -------------------------
    @staticmethod
    def predict(models, X):
        return {
            "y_main_pred": models["model_main"].predict(X),
            "y_sec_pred": models["model_secondary"].predict(X),
        }

    # -------------------------
    # Save best model (or last fold)
    # -------------------------
    @staticmethod
    def save(models, output_dir: Path, target: str = "separate"):
        output_dir.mkdir(parents=True, exist_ok=True)

        # If a list of fold results is provided, select the best fold(s) by RMSE.
        if isinstance(models, list):
            if len(models) == 0:
                raise ValueError("No fold results provided to save().")

            if target == "main":
                best = min(models, key=lambda x: x["rmse_main"])
                model_main = best["model_main"]
                model_secondary = best["model_secondary"]
            elif target == "secondary":
                best = min(models, key=lambda x: x["rmse_secondary"])
                model_main = best["model_main"]
                model_secondary = best["model_secondary"]
            elif target == "combined":
                best = min(
                    models,
                    key=lambda x: (x["rmse_main"] + x["rmse_secondary"]) / 2.0
                )
                model_main = best["model_main"]
                model_secondary = best["model_secondary"]
            elif target == "separate":
                best_main = min(models, key=lambda x: x["rmse_main"])
                best_sec = min(models, key=lambda x: x["rmse_secondary"])
                model_main = best_main["model_main"]
                model_secondary = best_sec["model_secondary"]
            else:
                raise ValueError(
                    "Invalid target. Use 'main', 'secondary', 'combined', or 'separate'."
                )
        else:
            model_main = models["model_main"]
            model_secondary = models["model_secondary"]

        joblib.dump(model_main, output_dir / "rf_main.pkl")
        joblib.dump(model_secondary, output_dir / "rf_secondary.pkl")

    # -------------------------
    # Load models
    # -------------------------
    @staticmethod
    def load(output_dir: Path):
        return {
            "model_main": joblib.load(output_dir / "rf_main.pkl"),
            "model_secondary": joblib.load(output_dir / "rf_secondary.pkl"),
        }

    # -------------------------
    # Select best fold (based on RMSE)
    # -------------------------
    @staticmethod
    def get_best_fold(fold_results, target="main"):
        if target == "main":
            return min(fold_results, key=lambda x: x["rmse_main"])
        else:
            return min(fold_results, key=lambda x: x["rmse_secondary"])
