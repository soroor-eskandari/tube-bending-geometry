import numpy as np


class RFGroupFeatureSelector:
    @staticmethod
    def _normalized_distance_matrix(
        left_main: np.ndarray,
        left_secondary: np.ndarray,
        right_main: np.ndarray,
        right_secondary: np.ndarray,
    ) -> np.ndarray:
        main_scale = np.std(right_main, axis=0) + 1e-8
        secondary_scale = np.std(right_secondary, axis=0) + 1e-8

        main_delta = (
            left_main[:, None, :] - right_main[None, :, :]
        ) / main_scale
        secondary_delta = (
            left_secondary[:, None, :] - right_secondary[None, :, :]
        ) / secondary_scale

        return (
            np.mean(main_delta ** 2, axis=2)
            + np.mean(secondary_delta ** 2, axis=2)
        )

    @staticmethod
    def _choose_diverse_pair_indices(
        *,
        left_main: np.ndarray,
        left_secondary: np.ndarray,
        right_main: np.ndarray,
        right_secondary: np.ndarray,
        rng: np.random.Generator,
        forbidden_indices: np.ndarray | None = None,
        top_fraction: float = 0.25,
    ) -> np.ndarray:
        distances = RFGroupFeatureSelector._normalized_distance_matrix(
            left_main=left_main,
            left_secondary=left_secondary,
            right_main=right_main,
            right_secondary=right_secondary,
        )

        if forbidden_indices is not None:
            for row_idx, forbidden_idx in enumerate(forbidden_indices):
                distances[row_idx, int(forbidden_idx)] = -np.inf

        n_candidates = right_main.shape[0]
        top_k = max(1, int(np.ceil(n_candidates * top_fraction)))
        paired_indices = np.empty(left_main.shape[0], dtype=int)

        for row_idx, row_distances in enumerate(distances):
            finite_indices = np.flatnonzero(np.isfinite(row_distances))
            if finite_indices.size == 0:
                paired_indices[row_idx] = int(forbidden_indices[row_idx])
                continue

            sorted_indices = finite_indices[
                np.argsort(row_distances[finite_indices])[::-1]
            ]
            top_indices = sorted_indices[: min(top_k, sorted_indices.size)]
            paired_indices[row_idx] = rng.choice(top_indices)

        return paired_indices

    @staticmethod
    def _replace_copied_values_with_interval_values(
        *,
        selected: np.ndarray,
        actual: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int, int]:
        selected = selected.copy()
        feature_span = feature_max - feature_min
        can_change = feature_span > 1e-12
        copied_mask = np.isclose(selected, actual, rtol=1e-9, atol=1e-12)
        replace_mask = copied_mask & can_change.reshape(1, -1)

        if not np.any(replace_mask):
            unchanged_count = int(
                np.count_nonzero(copied_mask & ~can_change.reshape(1, -1))
            )
            return selected, 0, unchanged_count

        row_indices, feature_indices = np.where(replace_mask)
        low = feature_min[feature_indices]
        high = feature_max[feature_indices]
        actual_values = actual[row_indices, feature_indices]
        span = high - low

        replacement = rng.uniform(low=low, high=high)
        too_close = np.isclose(replacement, actual_values, rtol=1e-9, atol=1e-12)

        if np.any(too_close):
            replacement[too_close] = np.where(
                np.abs(actual_values[too_close] - low[too_close])
                > np.abs(high[too_close] - actual_values[too_close]),
                low[too_close] + 0.05 * span[too_close],
                high[too_close] - 0.05 * span[too_close],
            )

        selected[row_indices, feature_indices] = replacement
        unchanged_count = int(
            np.count_nonzero(copied_mask & ~can_change.reshape(1, -1))
        )
        return selected, int(len(row_indices)), unchanged_count

    @staticmethod
    def sample_balanced_base_indices(
        *,
        n_original: int,
        n_new_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_original == 0:
            raise ValueError("Cannot select synthetic features from an empty group.")

        repeats = int(np.ceil(n_new_samples / n_original))
        sampled_indices = np.tile(np.arange(n_original), repeats)[:n_new_samples]
        rng.shuffle(sampled_indices)
        return sampled_indices

    @staticmethod
    def sample_independent_features_within_group_range(
        *,
        X_main: np.ndarray,
        X_secondary: np.ndarray,
        n_new_samples: int,
        rng: np.random.Generator,
    ) -> dict:
        n_original = X_main.shape[0]

        if n_original == 0:
            raise ValueError("Cannot select synthetic features from an empty group.")

        sampled_indices = rng.choice(n_original, size=n_new_samples, replace=True)
        X_main_actual = X_main[sampled_indices]
        X_secondary_actual = X_secondary[sampled_indices]

        X_main_group_min = np.min(X_main, axis=0)
        X_main_group_max = np.max(X_main, axis=0)
        X_secondary_group_min = np.min(X_secondary, axis=0)
        X_secondary_group_max = np.max(X_secondary, axis=0)

        return {
            "sampled_indices": sampled_indices,
            "X_main_actual": X_main_actual,
            "X_main_selected": rng.uniform(
                low=X_main_group_min,
                high=X_main_group_max,
                size=X_main_actual.shape,
            ),
            "X_main_group_min": X_main_group_min,
            "X_main_group_max": X_main_group_max,
            "X_secondary_actual": X_secondary_actual,
            "X_secondary_selected": rng.uniform(
                low=X_secondary_group_min,
                high=X_secondary_group_max,
                size=X_secondary_actual.shape,
            ),
            "X_secondary_group_min": X_secondary_group_min,
            "X_secondary_group_max": X_secondary_group_max,
            "selection_method": "independent_uniform_between_group_min_max",
        }

    @staticmethod
    def sample_independent_features_within_overall_range(
        *,
        X_main: np.ndarray,
        X_secondary: np.ndarray,
        n_new_samples: int,
        rng: np.random.Generator,
        X_main_reference: np.ndarray | None = None,
        X_secondary_reference: np.ndarray | None = None,
    ) -> dict:
        n_original = X_main.shape[0]

        if n_original == 0:
            raise ValueError("Cannot select synthetic features from an empty group.")

        sampled_indices = RFGroupFeatureSelector.sample_balanced_base_indices(
            n_original=n_original,
            n_new_samples=n_new_samples,
            rng=rng,
        )
        X_main_actual = X_main[sampled_indices]
        X_secondary_actual = X_secondary[sampled_indices]

        X_main_bounds = (
            np.vstack([X_main, X_main_reference])
            if X_main_reference is not None and X_main_reference.shape[0] > 0
            else X_main
        )
        X_secondary_bounds = (
            np.vstack([X_secondary, X_secondary_reference])
            if X_secondary_reference is not None and X_secondary_reference.shape[0] > 0
            else X_secondary
        )

        X_main_overall_min = np.min(X_main_bounds, axis=0)
        X_main_overall_max = np.max(X_main_bounds, axis=0)
        X_secondary_overall_min = np.min(X_secondary_bounds, axis=0)
        X_secondary_overall_max = np.max(X_secondary_bounds, axis=0)

        X_main_selected = rng.uniform(
            low=X_main_overall_min,
            high=X_main_overall_max,
            size=X_main_actual.shape,
        )
        X_secondary_selected = rng.uniform(
            low=X_secondary_overall_min,
            high=X_secondary_overall_max,
            size=X_secondary_actual.shape,
        )

        (
            X_main_selected,
            main_replaced_count,
            main_unchangeable_count,
        ) = RFGroupFeatureSelector._replace_copied_values_with_interval_values(
            selected=X_main_selected,
            actual=X_main_actual,
            feature_min=X_main_overall_min,
            feature_max=X_main_overall_max,
            rng=rng,
        )
        (
            X_secondary_selected,
            secondary_replaced_count,
            secondary_unchangeable_count,
        ) = RFGroupFeatureSelector._replace_copied_values_with_interval_values(
            selected=X_secondary_selected,
            actual=X_secondary_actual,
            feature_min=X_secondary_overall_min,
            feature_max=X_secondary_overall_max,
            rng=rng,
        )

        return {
            "sampled_indices": sampled_indices,
            "X_main_actual": X_main_actual,
            "X_main_selected": X_main_selected,
            "X_main_group_min": X_main_overall_min,
            "X_main_group_max": X_main_overall_max,
            "X_secondary_actual": X_secondary_actual,
            "X_secondary_selected": X_secondary_selected,
            "X_secondary_group_min": X_secondary_overall_min,
            "X_secondary_group_max": X_secondary_overall_max,
            "selection_method": "independent_uniform_between_overall_min_max",
            "main_replaced_copied_feature_count": main_replaced_count,
            "secondary_replaced_copied_feature_count": secondary_replaced_count,
            "main_unchangeable_copied_feature_count": main_unchangeable_count,
            "secondary_unchangeable_copied_feature_count": secondary_unchangeable_count,
        }

    @staticmethod
    def interpolate_within_group(
        *,
        X_main: np.ndarray,
        X_secondary: np.ndarray,
        n_new_samples: int,
        rng: np.random.Generator,
        X_main_reference: np.ndarray | None = None,
        X_secondary_reference: np.ndarray | None = None,
        reference_experiment_ids: list | None = None,
        alpha_min: float = 0.35,
        alpha_max: float = 0.95,
        singleton_alpha_min: float = 0.20,
        singleton_alpha_max: float = 0.75,
    ) -> dict:
        n_original = X_main.shape[0]

        if n_original == 0:
            raise ValueError("Cannot select synthetic features from an empty group.")

        left_indices = RFGroupFeatureSelector.sample_balanced_base_indices(
            n_original=n_original,
            n_new_samples=n_new_samples,
            rng=rng,
        )

        use_reference_pool = (
            n_original == 1
            and X_main_reference is not None
            and X_secondary_reference is not None
            and X_main_reference.shape[0] > 0
            and X_secondary_reference.shape[0] > 0
        )

        if use_reference_pool:
            n_reference = X_main_reference.shape[0]
            right_indices = RFGroupFeatureSelector._choose_diverse_pair_indices(
                left_main=X_main[left_indices],
                left_secondary=X_secondary[left_indices],
                right_main=X_main_reference,
                right_secondary=X_secondary_reference,
                rng=rng,
            )
            weights = rng.uniform(
                singleton_alpha_min,
                singleton_alpha_max,
                size=n_new_samples,
            )
            X_main_right = X_main_reference[right_indices]
            X_secondary_right = X_secondary_reference[right_indices]
            paired_experiment_ids = (
                [reference_experiment_ids[idx] for idx in right_indices]
                if reference_experiment_ids is not None
                else None
            )
        elif n_original == 1:
            right_indices = left_indices.copy()
            weights = np.zeros(n_new_samples, dtype=np.float64)
            X_main_right = X_main[right_indices]
            X_secondary_right = X_secondary[right_indices]
            paired_experiment_ids = None
        else:
            right_indices = RFGroupFeatureSelector._choose_diverse_pair_indices(
                left_main=X_main[left_indices],
                left_secondary=X_secondary[left_indices],
                right_main=X_main,
                right_secondary=X_secondary,
                rng=rng,
                forbidden_indices=left_indices,
            )
            weights = rng.uniform(alpha_min, alpha_max, size=n_new_samples)
            X_main_right = X_main[right_indices]
            X_secondary_right = X_secondary[right_indices]
            paired_experiment_ids = None

        weight_col = weights.reshape(-1, 1)

        X_main_actual = X_main[left_indices]
        X_secondary_actual = X_secondary[left_indices]
        X_main_paired_actual = X_main_right
        X_secondary_paired_actual = X_secondary_right

        X_main_bounds = (
            np.vstack([X_main, X_main_reference])
            if use_reference_pool
            else X_main
        )
        X_secondary_bounds = (
            np.vstack([X_secondary, X_secondary_reference])
            if use_reference_pool
            else X_secondary
        )
        X_main_group_min = np.min(X_main_bounds, axis=0)
        X_main_group_max = np.max(X_main_bounds, axis=0)
        X_secondary_group_min = np.min(X_secondary_bounds, axis=0)
        X_secondary_group_max = np.max(X_secondary_bounds, axis=0)

        X_main_selected = (
            (1.0 - weight_col) * X_main[left_indices]
            + weight_col * X_main_right
        )
        X_secondary_selected = (
            (1.0 - weight_col) * X_secondary[left_indices]
            + weight_col * X_secondary_right
        )

        (
            X_main_selected,
            main_replaced_count,
            main_unchangeable_count,
        ) = RFGroupFeatureSelector._replace_copied_values_with_interval_values(
            selected=X_main_selected,
            actual=X_main_actual,
            feature_min=X_main_group_min,
            feature_max=X_main_group_max,
            rng=rng,
        )
        (
            X_secondary_selected,
            secondary_replaced_count,
            secondary_unchangeable_count,
        ) = RFGroupFeatureSelector._replace_copied_values_with_interval_values(
            selected=X_secondary_selected,
            actual=X_secondary_actual,
            feature_min=X_secondary_group_min,
            feature_max=X_secondary_group_max,
            rng=rng,
        )

        return {
            "sampled_indices": left_indices,
            "paired_indices": right_indices,
            "paired_experiment_ids": paired_experiment_ids,
            "interpolation_weight": weights,
            "X_main_actual": X_main_actual,
            "X_main_paired_actual": X_main_paired_actual,
            "X_main_selected": X_main_selected,
            "X_main_group_min": X_main_group_min,
            "X_main_group_max": X_main_group_max,
            "X_secondary_actual": X_secondary_actual,
            "X_secondary_paired_actual": X_secondary_paired_actual,
            "X_secondary_selected": X_secondary_selected,
            "X_secondary_group_min": X_secondary_group_min,
            "X_secondary_group_max": X_secondary_group_max,
            "selection_method": (
                "single_row_interpolation_with_reference_pool"
                if use_reference_pool
                else "balanced_row_interpolation_within_group"
            ),
            "alpha_min": singleton_alpha_min if use_reference_pool else alpha_min,
            "alpha_max": singleton_alpha_max if use_reference_pool else alpha_max,
            "main_replaced_copied_feature_count": main_replaced_count,
            "secondary_replaced_copied_feature_count": secondary_replaced_count,
            "main_unchangeable_copied_feature_count": main_unchangeable_count,
            "secondary_unchangeable_copied_feature_count": secondary_unchangeable_count,
        }
