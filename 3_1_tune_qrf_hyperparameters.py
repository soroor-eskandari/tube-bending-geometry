import argparse
import logging
from itertools import product
from pathlib import Path

import pandas as pd

from src.pipeline.ml.qrf.data_splittor import DataSplittor
from src.pipeline.ml.qrf.geometry_data_preprocessor import GeometryPreprocessor
from src.pipeline.ml.qrf.qrf_model_trainer import QRFModelTrainer
from src.pipeline.ml.qrf.qrf_pipeline import qrf_training_geometry_sources
from src.pipeline.rf_augmentation.io_utils import read_table


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


N_ESTIMATORS_GRID = [
    100,
    300,
    500,
    800,
]

MAX_DEPTH_GRID = [
    5,
    10,
    15,
    20,
    None,
]

MIN_SAMPLES_LEAF_GRID = [
    1,
    3,
    5,
    10,
    20,
]

MIN_SAMPLES_SPLIT_GRID = [
    2,
    5,
    10,
]

MAX_FEATURES_GRID = [
    "sqrt",
    0.5,
    0.8,
    1.0,
]

BOOTSTRAP_GRID = [
    True,
]

QUANTILE_GRID = [
    (0.05, 0.95),
]

GLOBAL_COVERAGE_WEIGHT = 10.0
GROUP_COVERAGE_STD_WEIGHT = 5.0
INTERVAL_WIDTH_WEIGHT = 0.5

SORT_COLUMNS = [
    "geometry_source",
    "selection_score",
    "mean_rmse",
    "mean_calibration_error",
    "std_group_coverage",
    "mean_interval_width",
    "mean_abs_bias",
    "mean_mae",
    "group_calibration_error",
    "min_group_coverage",
]

SORT_ASCENDING = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
]


def parse_args():
    dataset_choices = [
        "all",
        *qrf_training_geometry_sources(Path(__file__).resolve().parent).keys(),
    ]

    parser = argparse.ArgumentParser(
        description="Tune QRF hyperparameters with MLflow tracking."
    )
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        default="all",
        help="Dataset source to tune. Defaults to all QRF training datasets.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to MLflow's local ./mlruns.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="QRF_Hyperparameter_Tuning",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def prepare_split(project_root: Path, geometry_path: Path):
    geometry_df = read_table(geometry_path)
    bending_df = pd.read_csv(
        project_root / "data" / "processed" / "processed_bending_setup.csv"
    )
    geometry_clean = GeometryPreprocessor.preprocess(
        geometry_df=geometry_df,
        bending_df=bending_df,
    )
    return DataSplittor.splittor(
        geometry_df=geometry_clean,
        unique_bending_df=bending_df,
        test_size=0.2,
        random_state=42,
    )


def trial_metrics(training_result, lower_quantile: float, upper_quantile: float):
    expected_coverage = upper_quantile - lower_quantile
    main_metrics = QRFModelTrainer._interval_metrics(
        y_true=training_result["main"]["y_true"],
        y_pred_median=training_result["main"]["y_pred_median"],
        y_pred_lower=training_result["main"]["y_pred_lower"],
        y_pred_upper=training_result["main"]["y_pred_upper"],
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    secondary_metrics = QRFModelTrainer._interval_metrics(
        y_true=training_result["secondary"]["y_true"],
        y_pred_median=training_result["secondary"]["y_pred_median"],
        y_pred_lower=training_result["secondary"]["y_pred_lower"],
        y_pred_upper=training_result["secondary"]["y_pred_upper"],
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    main_group_metrics = QRFModelTrainer._group_coverage_metrics(
        training_result["main"]["prediction_df"]
    )
    secondary_group_metrics = QRFModelTrainer._group_coverage_metrics(
        training_result["secondary"]["prediction_df"]
    )
    mean_coverage = (
        main_metrics["coverage"]
        + secondary_metrics["coverage"]
    ) / 2
    mean_calibration_error = abs(mean_coverage - expected_coverage)
    mean_interval_width = (
        main_metrics["mean_interval_width"]
        + secondary_metrics["mean_interval_width"]
    ) / 2
    mean_rmse = (
        main_metrics["rmse"]
        + secondary_metrics["rmse"]
    ) / 2
    mean_mae = (
        main_metrics["mae"]
        + secondary_metrics["mae"]
    ) / 2
    mean_group_coverage = (
        main_group_metrics["mean_group_coverage"]
        + secondary_group_metrics["mean_group_coverage"]
    ) / 2
    std_group_coverage = (
        main_group_metrics["std_group_coverage"]
        + secondary_group_metrics["std_group_coverage"]
    ) / 2
    mean_bias = (
        main_metrics["bias"]
        + secondary_metrics["bias"]
    ) / 2
    mean_abs_bias = (
        abs(main_metrics["bias"])
        + abs(secondary_metrics["bias"])
    ) / 2

    return {
        "main_coverage": main_metrics["coverage"],
        "main_calibration_error": main_metrics["calibration_error"],
        "main_mean_interval_width": main_metrics["mean_interval_width"],
        "main_rmse": main_metrics["rmse"],
        "main_mae": main_metrics["mae"],
        "main_bias": main_metrics["bias"],
        "main_mean_group_coverage": main_group_metrics["mean_group_coverage"],
        "main_min_group_coverage": main_group_metrics["min_group_coverage"],
        "main_std_group_coverage": main_group_metrics["std_group_coverage"],
        "secondary_coverage": secondary_metrics["coverage"],
        "secondary_calibration_error": secondary_metrics["calibration_error"],
        "secondary_mean_interval_width": secondary_metrics["mean_interval_width"],
        "secondary_rmse": secondary_metrics["rmse"],
        "secondary_mae": secondary_metrics["mae"],
        "secondary_bias": secondary_metrics["bias"],
        "secondary_mean_group_coverage": (
            secondary_group_metrics["mean_group_coverage"]
        ),
        "secondary_min_group_coverage": (
            secondary_group_metrics["min_group_coverage"]
        ),
        "secondary_std_group_coverage": (
            secondary_group_metrics["std_group_coverage"]
        ),
        "mean_coverage": mean_coverage,
        "mean_calibration_error": mean_calibration_error,
        "mean_interval_width": mean_interval_width,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "mean_bias": mean_bias,
        "mean_abs_bias": mean_abs_bias,
        "mean_group_coverage": mean_group_coverage,
        "min_group_coverage": min(
            main_group_metrics["min_group_coverage"],
            secondary_group_metrics["min_group_coverage"],
        ),
        "std_group_coverage": std_group_coverage,
    }


def qrf_param_grid():
    for (
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        max_features,
        bootstrap,
        quantiles,
    ) in product(
        N_ESTIMATORS_GRID,
        MAX_DEPTH_GRID,
        MIN_SAMPLES_LEAF_GRID,
        MIN_SAMPLES_SPLIT_GRID,
        MAX_FEATURES_GRID,
        BOOTSTRAP_GRID,
        QUANTILE_GRID,
    ):
        lower_quantile, upper_quantile = quantiles
        yield {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
        }


def qrf_trial_name(geometry_source: str, params: dict, suffix: str = None) -> str:
    depth_label = (
        "none"
        if params["max_depth"] is None
        else str(params["max_depth"])
    )
    max_features_label = str(params["max_features"]).replace(".", "p")
    bootstrap_label = "true" if params["bootstrap"] else "false"
    name = (
        f"qrf_tune_{geometry_source}"
        f"_n{params['n_estimators']}"
        f"_depth{depth_label}"
        f"_leaf{params['min_samples_leaf']}"
        f"_split{params['min_samples_split']}"
        f"_mf{max_features_label}"
        f"_boot{bootstrap_label}"
        f"_q{params['lower_quantile']}_{params['upper_quantile']}"
    )
    if suffix:
        name = f"{name}_{suffix}"
    return name


def add_selection_scores(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["group_calibration_error"] = (
        summary_df["mean_group_coverage"]
        - (summary_df["upper_quantile"] - summary_df["lower_quantile"])
    ).abs()
    summary_df["selection_score"] = (
        summary_df["mean_rmse"]
        + GLOBAL_COVERAGE_WEIGHT * summary_df["mean_calibration_error"]
        + GROUP_COVERAGE_STD_WEIGHT * summary_df["std_group_coverage"]
        + INTERVAL_WIDTH_WEIGHT * summary_df["mean_interval_width"]
    )
    return summary_df.sort_values(
        SORT_COLUMNS,
        ascending=SORT_ASCENDING,
    )


def tune_qrf_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    geometry_source: str,
    model_dir: Path,
    split_name: str = None,
    split_index: int = None,
    param_grid=None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str = None,
    mlflow_experiment: str = "QRF_Hyperparameter_Tuning",
    mlflow_tags: dict = None,
) -> pd.DataFrame:
    model_dir = Path(model_dir)
    param_grid = list(param_grid or qrf_param_grid())
    results = []

    for trial_index, params in enumerate(param_grid):
        paper_name = qrf_trial_name(
            geometry_source=geometry_source,
            params=params,
            suffix=(
                f"split{split_index:03d}"
                if split_index is not None
                else None
            ),
        )
        logger.info(
            "Tuning %s split=%s trial=%s/%s params=%s",
            geometry_source,
            split_name,
            trial_index + 1,
            len(param_grid),
            params,
        )

        trial_tags = {
            "Dataset": geometry_source,
            "geometry_source": geometry_source,
            "model_type": "group_conditional_qrf",
            "run_type": "hyperparameter_tuning",
        }
        if split_name is not None:
            trial_tags["split_name"] = split_name
        if split_index is not None:
            trial_tags["split_index"] = str(split_index)
        if mlflow_tags:
            trial_tags.update(mlflow_tags)

        training_result = QRFModelTrainer.train(
            train_df=train_df,
            test_df=test_df,
            model_dir=model_dir,
            paper_name=paper_name,
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            mlflow_run_name=paper_name,
            mlflow_tags=trial_tags,
            **params,
        )

        metrics = trial_metrics(
            training_result=training_result,
            lower_quantile=params["lower_quantile"],
            upper_quantile=params["upper_quantile"],
        )
        row = {
            "geometry_source": geometry_source,
            "split_index": split_index,
            "split_name": split_name,
            "paper_name": paper_name,
            **params,
            **metrics,
        }
        results.append(row)

    return add_selection_scores(pd.DataFrame(results))


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    geometry_sources = qrf_training_geometry_sources(project_root)

    if args.dataset != "all":
        geometry_sources = {
            args.dataset: geometry_sources[args.dataset],
        }

    model_dir = project_root / "src" / "pipeline" / "ml" / "model" / "qrf_tuning"
    summary_dir = project_root / "src" / "pipeline" / "ml" / "qrf" / "result"
    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for geometry_source, geometry_path in geometry_sources.items():
        logger.info(
            "Preparing split for %s from %s",
            geometry_source,
            geometry_path,
        )
        train_df, test_df = prepare_split(
            project_root=project_root,
            geometry_path=geometry_path,
        )

        for params in qrf_param_grid():
            paper_name = qrf_trial_name(
                geometry_source=geometry_source,
                params=params,
            )

            logger.info(
                "Tuning %s: n_estimators=%s max_depth=%s "
                "min_samples_leaf=%s min_samples_split=%s "
                "max_features=%s bootstrap=%s quantiles=%s/%s",
                geometry_source,
                params["n_estimators"],
                params["max_depth"],
                params["min_samples_leaf"],
                params["min_samples_split"],
                params["max_features"],
                params["bootstrap"],
                params["lower_quantile"],
                params["upper_quantile"],
            )

            training_result = QRFModelTrainer.train(
                train_df=train_df,
                test_df=test_df,
                model_dir=model_dir,
                paper_name=paper_name,
                use_mlflow=True,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                mlflow_experiment=args.mlflow_experiment,
                mlflow_run_name=paper_name,
                mlflow_tags={
                    "Dataset": geometry_source,
                    "geometry_source": geometry_source,
                    "geometry_path": str(geometry_path.relative_to(project_root)),
                    "model_type": "group_conditional_qrf",
                    "run_type": "hyperparameter_tuning",
                },
                **params,
            )

            metrics = trial_metrics(
                training_result=training_result,
                lower_quantile=params["lower_quantile"],
                upper_quantile=params["upper_quantile"],
            )
            results.append(
                {
                    "geometry_source": geometry_source,
                    "paper_name": paper_name,
                    **params,
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(results)
    summary_df = add_selection_scores(summary_df)

    summary_path = summary_dir / "qrf_hyperparameter_tuning_summary.csv"
    summary_df.to_csv(
        summary_path,
        index=False,
    )
    logger.info(
        "Saved QRF tuning summary to %s",
        summary_path,
    )

    best_by_dataset = summary_df.groupby("geometry_source").head(1)
    logger.info(
        "Best configs by dataset:\n%s",
        best_by_dataset[
            [
                "geometry_source",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "min_samples_split",
                "max_features",
                "bootstrap",
                "lower_quantile",
                "upper_quantile",
                "selection_score",
                "mean_coverage",
                "mean_group_coverage",
                "min_group_coverage",
                "std_group_coverage",
                "group_calibration_error",
                "mean_calibration_error",
                "mean_interval_width",
                "mean_rmse",
                "mean_mae",
                "mean_bias",
                "mean_abs_bias",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
