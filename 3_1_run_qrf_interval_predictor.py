import logging
from pathlib import Path

from src.pipeline.ml.qrf.qrf_pipeline import (
    QRFPipeline,
    qrf_training_geometry_sources,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BEST_QRF_PARAMS_BY_DATASET = {
    "real": {
        "n_estimators": 800,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "sensor_augmented_noise__time_wrapping__scaling__jittering": {
        "n_estimators": 800,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "within_group_interpolation_raw": {
        "n_estimators": 800,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
}


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent
    use_mlflow = False
    use_best_qrf_params = False
    use_experiment_split = False
    train_exp = None
    test_exp = None
    geometry_sources = qrf_training_geometry_sources(project_root)

    # ---------------------------------------
    # Run the QRF interval prediction workflow for the real geometry
    # and the augmented datasets exported for the UI.
    # ---------------------------------------
    for geometry_source, geometry_path in geometry_sources.items():
        qrf_params = (
            BEST_QRF_PARAMS_BY_DATASET[geometry_source]
            if use_best_qrf_params
            else {}
        )
        logger.info(
            "Running QRF interval prediction for %s from %s with %s",
            geometry_source,
            geometry_path,
            qrf_params,
        )
        QRFPipeline.run(
            project_root=project_root,
            geometry_source=geometry_source,
            geometry_path=geometry_path,
            qrf_params=qrf_params,
            use_mlflow=use_mlflow,
            use_experiment_split=use_experiment_split,
            train_exp=train_exp,
            test_exp=test_exp,
        )
