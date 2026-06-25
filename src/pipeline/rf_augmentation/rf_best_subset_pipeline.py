import pandas as pd
import numpy as np
from pathlib import Path
import logging
from itertools import combinations

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_model_trainer import RFModelTrainer
from src.pipeline.rf_augmentation.rf_augmentation_generator import RFAugmentationGenerator
from src.pipeline.rf_augmentation.geometry_rebuilder import GeometryRebuilder

logger = logging.getLogger(__name__)


class RFSubsetFinderPipeline:

    @staticmethod
    @log_function
    def run(project_root, top_k_feature, output_dir):

        # ============================================================
        # LOAD DATA
        # ============================================================
        logger.info("Reading data")

        machine_movement = pd.read_csv(
            project_root / "data" / "processed" / "machine_and_movement.csv"
        )
        bending = pd.read_csv(
            project_root / "data" / "processed" / "bending.csv"
        )
        geometry = pd.read_csv(
            project_root / "data" / "processed" / "geometry.csv"
        )

        result_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"
        model_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "model"

        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # LOAD FEATURE RANKING
        # ============================================================
        main_top_k = pd.read_csv(
            result_dir / "feature_type_rank_main.csv",
            nrows=top_k_feature
        )
        secondary_top_k = pd.read_csv(
            result_dir / "feature_type_rank_secondary.csv",
            nrows=top_k_feature
        )

        # ============================================================
        # PREPROCESS
        # ============================================================
        logger.info("Preprocessing data")

        machine_movement_clean, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=machine_movement,
            bending_df=bending,
        )

        # ============================================================
        # FEATURE TYPE SELECTION
        # ============================================================
        main_top_features = main_top_k["feature_type"].tolist()
        secondary_top_features = secondary_top_k["feature_type"].tolist()

        logger.info(f"Top-K features loaded: {len(main_top_features)}")

        # ============================================================
        # GREEDY FORWARD FEATURE SELECTION (DECOUPLED)
        # ============================================================
        logger.info("Starting greedy forward feature selection")

        current_main_subset = []
        current_sec_subset = []

        remaining_main = set(main_top_features)
        remaining_sec = set(secondary_top_features)

        best_score = -np.inf
        best_models = None

        search_results = []

        best_mlflow_main_r2 = 0.90
        best_mlflow_secondary_r2 = 0.80

        # early stopping params
        epsilon = 1e-3
        patience = 3
        patience_counter = 0

        step = 0

        while True:
            best_candidate_score = -np.inf
            best_candidate = None
            best_candidate_models = None

            # ------------------------------------------------------------
            # TRY ALL POSSIBLE ONE-STEP EXPANSIONS
            # ------------------------------------------------------------
            for m_feat in remaining_main:
                for s_feat in remaining_sec:

                    trial_main = current_main_subset + [m_feat]
                    trial_sec = current_sec_subset + [s_feat]

                    (
                        X_main_sub,
                        X_sec_sub,
                        y_main_sub,
                        y_sec_sub,
                        feature_names_main_sub,
                        feature_names_secondary_sub,
                    ) = RFTrainingDatasetBuilder.build(
                        machine_movement__df=machine_movement_clean,
                        geometry_df=geometry,
                        main_selected_features=trial_main,
                        secondary_selected_features=trial_sec,
                    )

                    # skip invalid feature sets
                    if X_main_sub.shape[1] == 0 or X_sec_sub.shape[1] == 0:
                        continue

                    models = RFModelTrainer.train(
                        X_main=X_main_sub,
                        X_secondary=X_sec_sub,
                        y_main=y_main_sub,
                        y_secondary=y_sec_sub,
                        model_dir=model_dir,

                        subset_main_size=len(trial_main),
                        subset_secondary_size=len(trial_sec),
                        subset_tag=f"greedy_step{step}",

                        random_state=1100,
                        use_mlflow=True,
                        mlflow_experiment="rf_greedy_search",
                        mlflow_run_name=f"greedy_step{step}_{hash((tuple(trial_main), tuple(trial_sec)))}",

                        mlflow_tags={
                            "search_type": "greedy_forward",
                            "step": step,
                        },

                        subset_indices_main=trial_main,
                        subset_indices_secondary=trial_sec,
                        feature_names_main=trial_main,
                        feature_names_secondary=trial_sec,

                        log_fold_models_to_mlflow=False,
                        log_best_fold_models_to_mlflow=True,
                        log_final_models_to_mlflow=False,
                        save_local_models=False,

                        current_best_mlflow_main_r2=best_mlflow_main_r2,
                        current_best_mlflow_secondary_r2=best_mlflow_secondary_r2,
                        min_main_r2_to_log=0.93,
                        min_secondary_r2_to_log=0.85,
                    )

                    best_mlflow_main_r2 = models["updated_best_mlflow_main_r2"]
                    best_mlflow_secondary_r2 = models["updated_best_mlflow_secondary_r2"]

                    score = (
                        models["metrics_best"]["r2_main_best"] +
                        models["metrics_best"]["r2_secondary_best"]
                    ) / 2.0

                    search_results.append({
                        "step": step,
                        "main_subset": trial_main,
                        "secondary_subset": trial_sec,
                        "r2_main_best": models["metrics_best"]["r2_main_best"],
                        "r2_secondary_best": models["metrics_best"]["r2_secondary_best"],
                        "score": score,
                    })

                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = (m_feat, s_feat)
                        best_candidate_models = models

            # ------------------------------------------------------------
            # EARLY STOP CHECK (SAFE)
            # ------------------------------------------------------------
            if best_candidate is None:
                logger.info("No valid candidates left → stopping")
                break

            improvement = best_candidate_score - best_score

            if improvement < epsilon:
                patience_counter += 1
                logger.info(f"No significant improvement Δ={improvement:.6f}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered (greedy)")
                    break
            else:
                patience_counter = 0

            # ------------------------------------------------------------
            # ACCEPT BEST FEATURE PAIR
            # ------------------------------------------------------------
            m_feat, s_feat = best_candidate

            current_main_subset.append(m_feat)
            current_sec_subset.append(s_feat)

            remaining_main.remove(m_feat)
            remaining_sec.remove(s_feat)

            best_score = best_candidate_score
            best_models = best_candidate_models

            logger.info(
                f"ACCEPTED STEP {step} | score={best_score:.6f} | "
                f"main={current_main_subset} | sec={current_sec_subset}"
            )

            step += 1


        # ============================================================
        # FINAL RESULTS
        # ============================================================
        logger.info("Greedy search finished")

        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best main subset: {current_main_subset}")
        logger.info(f"Best secondary subset: {current_sec_subset}")

        # save results
        search_results_df = pd.DataFrame(search_results)
        search_results_path = output_dir / ".csv"
        search_results_df.to_csv(search_results_path, index=False)

        logger.info(f"Saved greedy search results → {search_results_path}")
