import logging
from itertools import combinations
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class RFFeatureSubsetSearcher:
    """
    Exhaustive feature subset search for RandomForest multi-output regression.

    This class searches all non-empty feature subsets up to max_k
    and finds the subset with the best mean CV R² score.

    It can search separately for:
        - main target
        - secondary target

    Notes
    -----
    - Works well when the candidate feature count is moderate (e.g. <= 15).
    - For 15 features, full exhaustive search means 2^15 - 1 = 32767 subsets.
    - Uses KFold CV and mean R² as optimization objective.
    """

    @staticmethod
    def _validate_inputs(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]],
        dataset_name: str,
    ) -> List[str]:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"{dataset_name}: X must be 2D, got shape {X.shape}")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.ndim != 2:
            raise ValueError(f"{dataset_name}: y must be 1D or 2D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"{dataset_name}: X and y must have same number of rows, "
                f"got X={X.shape}, y={y.shape}"
            )

        n_features = X.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if len(feature_names) != n_features:
            raise ValueError(
                f"{dataset_name}: len(feature_names)={len(feature_names)} "
                f"does not match X.shape[1]={n_features}"
            )

        return feature_names

    @staticmethod
    def _make_cv(n_splits: int, random_state: int) -> KFold:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    @staticmethod
    def _make_model(
        n_estimators: int,
        random_state: int,
        n_jobs: int,
    ) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    @staticmethod
    def _subset_generator(n_features: int, max_k: Optional[int] = None):
        """
        Yield all feature index combinations from size 1 to max_k.
        If max_k is None, search all subset sizes.
        """
        upper_k = n_features if max_k is None else min(max_k, n_features)

        for k in range(1, upper_k + 1):
            for combo in combinations(range(n_features), k):
                yield combo

    @staticmethod
    def _count_total_subsets(n_features: int, max_k: Optional[int] = None) -> int:
        upper_k = n_features if max_k is None else min(max_k, n_features)
        total = 0
        for k in range(1, upper_k + 1):
            total += len(list(combinations(range(n_features), k)))
        return total

    @staticmethod
    @log_function
    def search_single(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        dataset_name: str = "main",
        max_k: Optional[int] = None,
        n_splits: int = 5,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs_rf: int = -1,
        n_jobs_cv: int = 1,
        top_n_to_save: int = 50,
        result_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Exhaustively search the best feature subset for one dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Target matrix, shape (n_samples, n_outputs) or (n_samples,)
        feature_names : list[str], optional
            Names of candidate features
        dataset_name : str
            Used in logs and saved filenames
        max_k : int or None
            Maximum subset size to search; if None, search all sizes
        n_splits : int
            Number of CV folds
        n_estimators : int
            Number of trees during subset search
        random_state : int
            Random seed
        n_jobs_rf : int
            Parallelism inside RandomForestRegressor
        n_jobs_cv : int
            Parallelism for cross_val_score
        top_n_to_save : int
            How many best subsets to save to CSV
        result_dir : Path or None
            Directory to save result CSV

        Returns
        -------
        dict with:
            best_indices
            best_feature_names
            best_score
            best_k
            results_df
        """
        feature_names = RFFeatureSubsetSearcher._validate_inputs(
            X=X,
            y=y,
            feature_names=feature_names,
            dataset_name=dataset_name,
        )

        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = X.shape[1]
        upper_k = n_features if max_k is None else min(max_k, n_features)

        total_subsets = 0
        for k in range(1, upper_k + 1):
            total_subsets += sum(1 for _ in combinations(range(n_features), k))

        logger.info(
            "[%s] Starting exhaustive subset search over %s features (max_k=%s, total_subsets=%s)",
            dataset_name,
            n_features,
            upper_k,
            total_subsets,
        )

        cv = RFFeatureSubsetSearcher._make_cv(
            n_splits=n_splits,
            random_state=random_state,
        )
        model = RFFeatureSubsetSearcher._make_model(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs_rf,
        )

        scorer = make_scorer(r2_score, multioutput="uniform_average")

        best_score = -np.inf
        best_indices = None
        best_feature_names = None
        results = []

        checked = 0
        log_every = max(1, total_subsets // 20)

        for combo in RFFeatureSubsetSearcher._subset_generator(
            n_features=n_features,
            max_k=max_k,
        ):
            X_sub = X[:, combo]

            scores = cross_val_score(
                estimator=model,
                X=X_sub,
                y=y,
                cv=cv,
                scoring=scorer,
                n_jobs=n_jobs_cv,
            )
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            subset_feature_names = [feature_names[i] for i in combo]

            results.append(
                {
                    "dataset": dataset_name,
                    "k": len(combo),
                    "feature_indices": list(combo),
                    "feature_names": subset_feature_names,
                    "feature_names_joined": " | ".join(subset_feature_names),
                    "mean_cv_r2": mean_score,
                    "std_cv_r2": std_score,
                }
            )

            if mean_score > best_score:
                best_score = mean_score
                best_indices = list(combo)
                best_feature_names = subset_feature_names
                logger.info(
                    "[%s] New best subset found | k=%s | mean_cv_r2=%.6f | features=%s",
                    dataset_name,
                    len(combo),
                    mean_score,
                    best_feature_names,
                )

            checked += 1
            if checked % log_every == 0 or checked == total_subsets:
                logger.info(
                    "[%s] Checked %s / %s subsets",
                    dataset_name,
                    checked,
                    total_subsets,
                )

        results_df = pd.DataFrame(results).sort_values(
            by="mean_cv_r2",
            ascending=False,
        ).reset_index(drop=True)

        if result_dir is not None:
            result_dir.mkdir(parents=True, exist_ok=True)

            full_path = result_dir / f"feature_subset_search_{dataset_name}.csv"
            top_path = result_dir / f"feature_subset_search_{dataset_name}_top{top_n_to_save}.csv"

            results_df.to_csv(full_path, index=False)
            results_df.head(top_n_to_save).to_csv(top_path, index=False)

            logger.info("[%s] Saved full subset search results to: %s", dataset_name, full_path)
            logger.info("[%s] Saved top-%s subset results to: %s", dataset_name, top_n_to_save, top_path)

        logger.info(
            "[%s] Search completed | best_k=%s | best_score=%.6f | best_features=%s",
            dataset_name,
            len(best_indices) if best_indices is not None else None,
            best_score,
            best_feature_names,
        )

        return {
            "best_indices": best_indices,
            "best_feature_names": best_feature_names,
            "best_score": best_score,
            "best_k": len(best_indices) if best_indices is not None else None,
            "results_df": results_df,
        }

    @staticmethod
    @log_function
    def search(
        X_main: np.ndarray,
        X_secondary: np.ndarray,
        y_main: np.ndarray,
        y_secondary: np.ndarray,
        feature_names_main: Optional[List[str]] = None,
        feature_names_secondary: Optional[List[str]] = None,
        max_k: Optional[int] = None,
        n_splits: int = 5,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs_rf: int = -1,
        n_jobs_cv: int = 1,
        top_n_to_save: int = 50,
        result_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Search best subset separately for main and secondary datasets.
        """
        logger.info("Starting feature subset search for MAIN")
        main_result = RFFeatureSubsetSearcher.search_single(
            X=X_main,
            y=y_main,
            feature_names=feature_names_main,
            dataset_name="main",
            max_k=max_k,
            n_splits=n_splits,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs_rf=n_jobs_rf,
            n_jobs_cv=n_jobs_cv,
            top_n_to_save=top_n_to_save,
            result_dir=result_dir,
        )

        logger.info("Starting feature subset search for SECONDARY")
        secondary_result = RFFeatureSubsetSearcher.search_single(
            X=X_secondary,
            y=y_secondary,
            feature_names=feature_names_secondary,
            dataset_name="secondary",
            max_k=max_k,
            n_splits=n_splits,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs_rf=n_jobs_rf,
            n_jobs_cv=n_jobs_cv,
            top_n_to_save=top_n_to_save,
            result_dir=result_dir,
        )

        return {
            "best_main_idx": main_result["best_indices"],
            "best_main_features": main_result["best_feature_names"],
            "best_main_score": main_result["best_score"],
            "best_main_k": main_result["best_k"],
            "best_secondary_idx": secondary_result["best_indices"],
            "best_secondary_features": secondary_result["best_feature_names"],
            "best_secondary_score": secondary_result["best_score"],
            "best_secondary_k": secondary_result["best_k"],
            "main_results_df": main_result["results_df"],
            "secondary_results_df": secondary_result["results_df"],
        }