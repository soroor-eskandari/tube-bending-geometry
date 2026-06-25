import pandas as pd
from src.ml.quantile_random_forest.trainer import QRFTrainer
from src.ml.quantile_random_forest.predictor import QRFPredictor
from src.ml.quantile_random_forest.config import QRF_CONFIG

# Load data
df = pd.read_csv(r"/Users/soroureskandari/Master Thesis /tube-bending-geometry/data/rf_augmented/geometry_augmented.csv")

# Train
trainer = QRFTrainer(QRF_CONFIG)

X_train, X_test, y_sec_train, y_sec_test, y_main_train, y_main_test = trainer.split_data(df)

trainer.fit(X_train, y_sec_train, y_main_train)

# Predict intervals for 45 angles
predictor = QRFPredictor(trainer.get_models())

angles = sorted(df["Angle[degree]ORDistance[mm]"].unique())
interval_df = predictor.predict_on_grid(angles)

print(interval_df.head())