import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_feature_ranking_dataset_builder import (
    RFFeatureRankerDatasetBuilder,
)
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor

logger = logging.getLogger(__name__)


class RFBendingFeatureRankerPipeline:
    @staticmethod
    def _build_rank_df(feature_names, importances, target_name: str) -> pd.DataFrame:
        rank_df = (
            pd.DataFrame(
                {
                    "bending_feature": feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        rank_df.insert(0, "rank", rank_df.index + 1)
        rank_df.insert(1, "target", target_name)
        return rank_df

    @staticmethod
    @log_function
    def run(
        project_root: Path,
        n_estimators: int = 300,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        project_root = Path(project_root)

        logger.info("Loading input datasets")
        bending = pd.read_csv(project_root / "data" / "processed" / "bending.csv")
        geometry = pd.read_csv(project_root / "data" / "processed" / "geometry.csv")

        logger.info("Bending: %s", bending.shape)
        logger.info("Geometry: %s", geometry.shape)

        logger.info("Preprocessing bending data")
        empty_machine = bending[["Experiment_ID"]].copy()
        _, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=empty_machine,
            bending_df=bending,
        )
        logger.info("bending_clean: %s", bending_clean.shape)

        logger.info("Building bending-only feature ranking dataset")
        X_rf, y_main, y_secondary, feature_names = RFFeatureRankerDatasetBuilder.build(
            machine_movement_df=empty_machine,
            bending_df=bending_clean,
            geometry_df=geometry,
            use_bending=True,
        )

        logger.info("X_rf shape: %s", X_rf.shape)
        logger.info("y_main shape: %s", y_main.shape)
        logger.info("y_secondary shape: %s", y_secondary.shape)

        rf_main = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        rf_secondary = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        logger.info("Training bending feature ranker models")
        rf_main.fit(X_rf, y_main)
        rf_secondary.fit(X_rf, y_secondary)

        main_rank = RFBendingFeatureRankerPipeline._build_rank_df(
            feature_names=feature_names,
            importances=rf_main.feature_importances_,
            target_name="main_axis",
        )
        secondary_rank = RFBendingFeatureRankerPipeline._build_rank_df(
            feature_names=feature_names,
            importances=rf_secondary.feature_importances_,
            target_name="secondary_axis",
        )

        combined_rank = (
            main_rank[["bending_feature", "importance"]]
            .rename(columns={"importance": "main_importance"})
            .merge(
                secondary_rank[["bending_feature", "importance"]].rename(
                    columns={"importance": "secondary_importance"}
                ),
                on="bending_feature",
                how="inner",
            )
        )
        combined_rank["mean_importance"] = combined_rank[
            ["main_importance", "secondary_importance"]
        ].mean(axis=1)
        combined_rank = (
            combined_rank.sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )
        combined_rank.insert(0, "rank", combined_rank.index + 1)

        result_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"
        model_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "model"
        result_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        main_rank_path = result_dir / "bending_feature_rank_main_axis.csv"
        secondary_rank_path = result_dir / "bending_feature_rank_secondary_axis.csv"
        combined_rank_path = result_dir / "bending_feature_rank_mean_axis.csv"

        main_rank.to_csv(main_rank_path, index=False)
        secondary_rank.to_csv(secondary_rank_path, index=False)
        combined_rank.to_csv(combined_rank_path, index=False)

        joblib.dump(
            rf_main,
            model_dir / "rf_bending_feature_ranker_main_axis.joblib",
        )
        joblib.dump(
            rf_secondary,
            model_dir / "rf_bending_feature_ranker_secondary_axis.joblib",
        )

        logger.info("Saved main bending feature ranks to: %s", main_rank_path)
        logger.info(
            "Saved secondary bending feature ranks to: %s",
            secondary_rank_path,
        )
        logger.info("Saved mean bending feature ranks to: %s", combined_rank_path)

        return main_rank, secondary_rank, combined_rank
