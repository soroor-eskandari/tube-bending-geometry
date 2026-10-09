import pandas as pd
import numpy as np
import logging
import hashlib

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_model_trainer import RFModelTrainer

logger = logging.getLogger(__name__)


class RFSubsetFinderPipeline:
    MANUAL_FEATURE_TYPES = {
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skew",
        "kurtosis",
        "iqr",
        "range",
        "energy",
        "rms",
        "sum",
        "abs",
        "var",
        "mad",
        "coeff_var",
        "p10",
        "p25",
        "p75",
        "p90",
        "crossings",
        "peaks",
        "diff",
        "length",
    }

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
        main_top_k = pd.read_csv(result_dir / "feature_type_rank_main.csv")
        secondary_top_k = pd.read_csv(result_dir / "feature_type_rank_secondary.csv")

        main_top_k = main_top_k[
            main_top_k["feature_type"].isin(RFSubsetFinderPipeline.MANUAL_FEATURE_TYPES)
        ].head(top_k_feature)
        secondary_top_k = secondary_top_k[
            secondary_top_k["feature_type"].isin(
                RFSubsetFinderPipeline.MANUAL_FEATURE_TYPES
            )
        ].head(top_k_feature)

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

        logger.info(
            "Top-K manual features loaded: main=%s secondary=%s",
            len(main_top_features),
            len(secondary_top_features),
        )

        # ============================================================
        # INDEPENDENT GREEDY FORWARD FEATURE SELECTION
        # ============================================================
        logger.info("Starting independent greedy forward feature selection")

        def create_subset_id(features: list[str]) -> str:
            content = "|".join(features)
            return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

        # Preserve the ranking order and remove possible duplicates.
        remaining_main = list(dict.fromkeys(main_top_features))
        remaining_sec = list(dict.fromkeys(secondary_top_features))

        current_main_subset = []
        current_sec_subset = []

        current_main_score = -np.inf
        current_secondary_score = -np.inf

        best_main_score_seen = -np.inf
        best_secondary_score_seen = -np.inf

        best_main_subset_seen = []
        best_secondary_subset_seen = []

        search_results = []
        accepted_steps = []

        best_mlflow_main_r2 = 0.90
        best_mlflow_secondary_r2 = 0.80

        epsilon = 1e-3
        patience = 3

        main_patience_counter = 0
        secondary_patience_counter = 0

        main_search_active = True
        secondary_search_active = True

        step = 0

        while (
            (main_search_active and remaining_main)
            or (secondary_search_active and remaining_sec)
        ):
            accepted_any = False

            if main_search_active and remaining_main:
                logger.info(f"Evaluating MAIN candidates at step {step}")

                best_main_candidate_score = -np.inf
                best_main_candidate = None
                best_main_candidate_result_index = None

                for m_feat in remaining_main:
                    trial_main = current_main_subset + [m_feat]
                    trial_sec = (
                        current_sec_subset.copy()
                        if current_sec_subset
                        else [secondary_top_features[0]]
                    )

                    (
                        X_main_sub,
                        X_sec_sub,
                        y_main_sub,
                        y_sec_sub,
                        feature_names_main_sub,
                        feature_names_secondary_sub,
                    ) = RFTrainingDatasetBuilder.build(
                        machine_movement__df=machine_movement_clean,
                        bending_df=bending_clean,
                        geometry_df=geometry,
                        main_selected_features=trial_main,
                        secondary_selected_features=trial_sec,
                        include_bending_features=False,
                    )

                    if X_main_sub.shape[1] == 0:
                        logger.warning(
                            f"Skipping MAIN candidate {m_feat}: "
                            "empty Main predictor matrix"
                        )
                        continue

                    if X_sec_sub.shape[1] == 0:
                        logger.warning(
                            f"Skipping MAIN candidate {m_feat}: "
                            "empty Secondary predictor matrix"
                        )
                        continue

                    subset_id = create_subset_id(trial_main)

                    models = RFModelTrainer.train(
                        X_main=X_main_sub,
                        X_secondary=X_sec_sub,
                        y_main=y_main_sub,
                        y_secondary=y_sec_sub,
                        model_dir=model_dir,
                        subset_main_size=len(trial_main),
                        subset_secondary_size=len(trial_sec),
                        subset_tag=f"greedy_main_step{step}",
                        random_state=1100,
                        use_mlflow=True,
                        mlflow_experiment="rf_greedy_search",
                        mlflow_run_name=f"greedy_main_step{step}_{subset_id}",
                        mlflow_tags={
                            "search_type": "independent_greedy_forward",
                            "axis": "main",
                            "step": str(step),
                            "candidate_feature": m_feat,
                            "subset_id": subset_id,
                        },
                        subset_indices_main=trial_main,
                        subset_indices_secondary=trial_sec,
                        feature_names_main=feature_names_main_sub,
                        feature_names_secondary=feature_names_secondary_sub,
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
                    best_mlflow_secondary_r2 = models[
                        "updated_best_mlflow_secondary_r2"
                    ]

                    main_score = models["metrics_best"]["r2_main_best"]
                    secondary_diagnostic_score = models["metrics_best"][
                        "r2_secondary_best"
                    ]

                    result_index = len(search_results)
                    search_results.append(
                        {
                            "step": step,
                            "axis": "main",
                            "candidate_feature": m_feat,
                            "subset_id": subset_id,
                            "main_subset": list(trial_main),
                            "secondary_subset": list(trial_sec),
                            "r2_main": main_score,
                            "r2_secondary_diagnostic": secondary_diagnostic_score,
                            "selection_score": main_score,
                            "is_step_winner": False,
                            "is_accepted": False,
                            "is_global_best_at_step": False,
                        }
                    )

                    if main_score > best_main_candidate_score:
                        best_main_candidate_score = main_score
                        best_main_candidate = m_feat
                        best_main_candidate_result_index = result_index

                if best_main_candidate is None:
                    logger.info("No valid MAIN candidates remain; stopping MAIN search")
                    main_search_active = False
                else:
                    main_improvement = best_main_candidate_score - current_main_score
                    search_results[best_main_candidate_result_index][
                        "is_step_winner"
                    ] = True

                    if main_improvement < epsilon:
                        main_patience_counter += 1
                        logger.info(
                            "MAIN improvement below threshold | "
                            f"delta={main_improvement:.6f} | "
                            f"patience={main_patience_counter}/{patience}"
                        )
                    else:
                        main_patience_counter = 0

                    if main_patience_counter >= patience:
                        logger.info("Early stopping triggered for MAIN greedy search")
                        main_search_active = False
                    else:
                        current_main_subset.append(best_main_candidate)
                        remaining_main.remove(best_main_candidate)
                        current_main_score = best_main_candidate_score
                        accepted_any = True

                        search_results[best_main_candidate_result_index][
                            "is_accepted"
                        ] = True

                        is_new_global_best = current_main_score > best_main_score_seen

                        if is_new_global_best:
                            best_main_score_seen = current_main_score
                            best_main_subset_seen = current_main_subset.copy()
                            search_results[best_main_candidate_result_index][
                                "is_global_best_at_step"
                            ] = True

                        accepted_steps.append(
                            {
                                "axis": "main",
                                "step": step,
                                "added_feature": best_main_candidate,
                                "accepted_subset": current_main_subset.copy(),
                                "selection_score": current_main_score,
                                "improvement": main_improvement,
                                "patience_counter": main_patience_counter,
                                "is_global_best": is_new_global_best,
                            }
                        )

                        logger.info(
                            f"ACCEPTED MAIN STEP {step} | "
                            f"feature={best_main_candidate} | "
                            f"r2={current_main_score:.6f} | "
                            f"delta={main_improvement:.6f} | "
                            f"subset={current_main_subset}"
                        )

            if secondary_search_active and remaining_sec:
                logger.info(f"Evaluating SECONDARY candidates at step {step}")

                best_secondary_candidate_score = -np.inf
                best_secondary_candidate = None
                best_secondary_candidate_result_index = None

                for s_feat in remaining_sec:
                    trial_main = (
                        current_main_subset.copy()
                        if current_main_subset
                        else [main_top_features[0]]
                    )
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
                        bending_df=bending_clean,
                        geometry_df=geometry,
                        main_selected_features=trial_main,
                        secondary_selected_features=trial_sec,
                        include_bending_features=False,
                    )

                    if X_main_sub.shape[1] == 0:
                        logger.warning(
                            f"Skipping SECONDARY candidate {s_feat}: "
                            "empty Main predictor matrix"
                        )
                        continue

                    if X_sec_sub.shape[1] == 0:
                        logger.warning(
                            f"Skipping SECONDARY candidate {s_feat}: "
                            "empty Secondary predictor matrix"
                        )
                        continue

                    subset_id = create_subset_id(trial_sec)

                    models = RFModelTrainer.train(
                        X_main=X_main_sub,
                        X_secondary=X_sec_sub,
                        y_main=y_main_sub,
                        y_secondary=y_sec_sub,
                        model_dir=model_dir,
                        subset_main_size=len(trial_main),
                        subset_secondary_size=len(trial_sec),
                        subset_tag=f"greedy_secondary_step{step}",
                        random_state=1100,
                        use_mlflow=True,
                        mlflow_experiment="rf_greedy_search",
                        mlflow_run_name=f"greedy_secondary_step{step}_{subset_id}",
                        mlflow_tags={
                            "search_type": "independent_greedy_forward",
                            "axis": "secondary",
                            "step": str(step),
                            "candidate_feature": s_feat,
                            "subset_id": subset_id,
                        },
                        subset_indices_main=trial_main,
                        subset_indices_secondary=trial_sec,
                        feature_names_main=feature_names_main_sub,
                        feature_names_secondary=feature_names_secondary_sub,
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
                    best_mlflow_secondary_r2 = models[
                        "updated_best_mlflow_secondary_r2"
                    ]

                    secondary_score = models["metrics_best"]["r2_secondary_best"]
                    main_diagnostic_score = models["metrics_best"]["r2_main_best"]

                    result_index = len(search_results)
                    search_results.append(
                        {
                            "step": step,
                            "axis": "secondary",
                            "candidate_feature": s_feat,
                            "subset_id": subset_id,
                            "main_subset": list(trial_main),
                            "secondary_subset": list(trial_sec),
                            "r2_main_diagnostic": main_diagnostic_score,
                            "r2_secondary": secondary_score,
                            "selection_score": secondary_score,
                            "is_step_winner": False,
                            "is_accepted": False,
                            "is_global_best_at_step": False,
                        }
                    )

                    if secondary_score > best_secondary_candidate_score:
                        best_secondary_candidate_score = secondary_score
                        best_secondary_candidate = s_feat
                        best_secondary_candidate_result_index = result_index

                if best_secondary_candidate is None:
                    logger.info(
                        "No valid SECONDARY candidates remain; "
                        "stopping SECONDARY search"
                    )
                    secondary_search_active = False
                else:
                    secondary_improvement = (
                        best_secondary_candidate_score - current_secondary_score
                    )
                    search_results[best_secondary_candidate_result_index][
                        "is_step_winner"
                    ] = True

                    if secondary_improvement < epsilon:
                        secondary_patience_counter += 1
                        logger.info(
                            "SECONDARY improvement below threshold | "
                            f"delta={secondary_improvement:.6f} | "
                            f"patience={secondary_patience_counter}/{patience}"
                        )
                    else:
                        secondary_patience_counter = 0

                    if secondary_patience_counter >= patience:
                        logger.info(
                            "Early stopping triggered for SECONDARY greedy search"
                        )
                        secondary_search_active = False
                    else:
                        current_sec_subset.append(best_secondary_candidate)
                        remaining_sec.remove(best_secondary_candidate)
                        current_secondary_score = best_secondary_candidate_score
                        accepted_any = True

                        search_results[best_secondary_candidate_result_index][
                            "is_accepted"
                        ] = True

                        is_new_global_best = (
                            current_secondary_score > best_secondary_score_seen
                        )

                        if is_new_global_best:
                            best_secondary_score_seen = current_secondary_score
                            best_secondary_subset_seen = current_sec_subset.copy()
                            search_results[best_secondary_candidate_result_index][
                                "is_global_best_at_step"
                            ] = True

                        accepted_steps.append(
                            {
                                "axis": "secondary",
                                "step": step,
                                "added_feature": best_secondary_candidate,
                                "accepted_subset": current_sec_subset.copy(),
                                "selection_score": current_secondary_score,
                                "improvement": secondary_improvement,
                                "patience_counter": secondary_patience_counter,
                                "is_global_best": is_new_global_best,
                            }
                        )

                        logger.info(
                            f"ACCEPTED SECONDARY STEP {step} | "
                            f"feature={best_secondary_candidate} | "
                            f"r2={current_secondary_score:.6f} | "
                            f"delta={secondary_improvement:.6f} | "
                            f"subset={current_sec_subset}"
                        )

            if not accepted_any:
                logger.info("No candidates were accepted for either axis; stopping")
                break

            step += 1

        logger.info("Greedy search finished")
        logger.info(f"Best observed MAIN R2: {best_main_score_seen:.6f}")
        logger.info(
            f"Best observed SECONDARY R2: {best_secondary_score_seen:.6f}"
        )
        logger.info(f"Best observed MAIN subset: {best_main_subset_seen}")
        logger.info(
            f"Best observed SECONDARY subset: {best_secondary_subset_seen}"
        )

        search_results_df = pd.DataFrame(search_results)
        search_results_path = output_dir / "greedy_search_results.csv"
        search_results_df.to_csv(search_results_path, index=False)

        accepted_steps_df = pd.DataFrame(accepted_steps)
        accepted_steps_path = output_dir / "greedy_accepted_steps.csv"
        accepted_steps_df.to_csv(accepted_steps_path, index=False)

        best_subsets_df = pd.DataFrame(
            [
                {
                    "axis": "main",
                    "best_subset": best_main_subset_seen,
                    "subset_size": len(best_main_subset_seen),
                    "best_r2": best_main_score_seen,
                    "epsilon": epsilon,
                    "patience": patience,
                    "include_bending_features": False,
                },
                {
                    "axis": "secondary",
                    "best_subset": best_secondary_subset_seen,
                    "subset_size": len(best_secondary_subset_seen),
                    "best_r2": best_secondary_score_seen,
                    "epsilon": epsilon,
                    "patience": patience,
                    "include_bending_features": False,
                },
            ]
        )

        best_subsets_path = output_dir / "greedy_best_subsets.csv"
        best_subsets_df.to_csv(best_subsets_path, index=False)

        logger.info(f"Saved all candidate results to {search_results_path}")
        logger.info(f"Saved accepted steps to {accepted_steps_path}")
        logger.info(f"Saved best subsets to {best_subsets_path}")
