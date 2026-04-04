from sklearn.model_selection import GroupShuffleSplit
from .model import QRFModel


class QRFTrainer:
    def __init__(self, config):
        self.config = config

        self.models = {
            "secondary": QRFModel(**config),
            "main": QRFModel(**config)
        }

    def split_data(self, df):
        X = df[["Angle[degree]ORDistance[mm]"]]
        y_sec = df["Secondary-axis [mm]"]
        y_main = df["Main-axis [mm]"]
        groups = df["Experiment_ID"]

        gss = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y_sec, groups=groups))

        return (
            X.iloc[train_idx], X.iloc[test_idx],
            y_sec.iloc[train_idx], y_sec.iloc[test_idx],
            y_main.iloc[train_idx], y_main.iloc[test_idx]
        )

    def fit(self, X_train, y_sec_train, y_main_train):
        self.models["secondary"].fit(X_train, y_sec_train)
        self.models["main"].fit(X_train, y_main_train)

    def get_models(self):
        return self.models