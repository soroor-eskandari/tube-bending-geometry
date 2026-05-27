import numpy as np


class RFGroupFeatureSelector:
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
    def interpolate_within_group(
        *,
        X_main: np.ndarray,
        X_secondary: np.ndarray,
        n_new_samples: int,
        rng: np.random.Generator,
    ) -> dict:
        n_original = X_main.shape[0]

        if n_original == 0:
            raise ValueError("Cannot select synthetic features from an empty group.")

        left_indices = rng.choice(n_original, size=n_new_samples, replace=True)

        if n_original == 1:
            right_indices = left_indices.copy()
            weights = np.zeros(n_new_samples, dtype=np.float64)
        else:
            right_indices = rng.choice(n_original, size=n_new_samples, replace=True)
            same_pair_mask = right_indices == left_indices
            right_indices[same_pair_mask] = (
                right_indices[same_pair_mask] + 1
            ) % n_original
            weights = rng.uniform(0.0, 1.0, size=n_new_samples)

        weight_col = weights.reshape(-1, 1)

        X_main_actual = X_main[left_indices]
        X_secondary_actual = X_secondary[left_indices]

        X_main_selected = (
            (1.0 - weight_col) * X_main[left_indices]
            + weight_col * X_main[right_indices]
        )
        X_secondary_selected = (
            (1.0 - weight_col) * X_secondary[left_indices]
            + weight_col * X_secondary[right_indices]
        )

        return {
            "sampled_indices": left_indices,
            "paired_indices": right_indices,
            "interpolation_weight": weights,
            "X_main_actual": X_main_actual,
            "X_main_selected": X_main_selected,
            "X_main_group_min": np.min(X_main, axis=0),
            "X_main_group_max": np.max(X_main, axis=0),
            "X_secondary_actual": X_secondary_actual,
            "X_secondary_selected": X_secondary_selected,
            "X_secondary_group_min": np.min(X_secondary, axis=0),
            "X_secondary_group_max": np.max(X_secondary, axis=0),
            "selection_method": "row_interpolation_within_group",
        }
