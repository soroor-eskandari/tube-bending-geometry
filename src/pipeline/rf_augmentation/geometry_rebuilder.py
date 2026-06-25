import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GeometryRebuilder:
    """
    Converts model outputs (wide format) into geometry DataFrame (long format).
    """

    @staticmethod
    def build(
        y_main: np.ndarray,
        y_secondary: np.ndarray,
        angle_values: np.ndarray = None,
        n_original: int | None = None,
        experiment_ids: list | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        y_main : np.ndarray (n_experiments, n_points)
        y_secondary : np.ndarray (n_experiments, n_points)
        angle_values : np.ndarray (n_points,)
            Optional: predefined angle/distance values
        n_original : int | None
            Number of original experiments. If provided, adds a boolean
            "Synthetic" column where rows from original experiments are False
            and augmented experiments are True.

        Returns
        -------
        geometry_df : pd.DataFrame
        """
        n_experiments, n_points = y_main.shape
        if experiment_ids is None:
            experiment_ids = np.arange(1, n_experiments + 1)
        else:
            experiment_ids = list(experiment_ids)

        if len(experiment_ids) != n_experiments:
            raise ValueError(
                "experiment_ids length must match number of experiments "
                f"({n_experiments}), got {len(experiment_ids)}"
            )

        # -------------------------
        # Default angle values
        # -------------------------
        if angle_values is None:
            # You MUST adjust this if your real data uses different angles
            angle_values = np.linspace(0, 180, n_points)

        if len(angle_values) != n_points:
            raise ValueError(
                f"angle_values length must match number of points ({n_points}), "
                f"got {len(angle_values)}"
            )

        # -------------------------
        # Build DataFrame
        # -------------------------
        records = []

        for exp_id in range(n_experiments):
            is_synthetic = False
            if n_original is not None:
                is_synthetic = exp_id >= n_original
            for i in range(n_points):
                records.append({
                    "Experiment_ID": experiment_ids[exp_id],
                    "Angle[degree]ORDistance[mm]": angle_values[i],
                    "Secondary-axis [mm]": y_secondary[exp_id, i],
                    "Main-axis [mm]": y_main[exp_id, i],
                    "Synthetic": is_synthetic,
                })

        geometry_df = pd.DataFrame(records)

        return geometry_df
