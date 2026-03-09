import logging
import numpy as np
import pandas as pd

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class SyntheticExperimentGenerator:
    """
    Generates synthetic experiments for groups with fewer experiments
    than the expected number.
    """

    def __init__(
        self,
        angle_col: str,
        feature_cols: list[str],
        expected_per_group: int,
        random_state: int = 42,
    ):
        self.angle_col = angle_col
        self.feature_cols = feature_cols
        self.expected_per_group = expected_per_group

        np.random.seed(random_state)

    @log_function
    def generate(self, df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:

        synthetic_rows = []
        max_exp_id = df["Experiment_ID"].max()

        logger.info("Starting synthetic generation")
        logger.info("Expected experiments per group: %s", self.expected_per_group)

        for group_id, group in df.groupby("Group_ID"):

            existing_experiments = group["Experiment_ID"].nunique()
            needed = max(0, self.expected_per_group - existing_experiments)

            logger.info(
                "Group %s | existing=%s | generating=%s",
                group_id,
                existing_experiments,
                needed,
            )

            if needed == 0:
                continue

            group_stats = stats[stats["Group_ID"] == group_id].copy()
            group_stats = group_stats.sort_values(self.angle_col)

            angles = group_stats[self.angle_col].values
            theta = 2 * np.pi * (angles - angles.min()) / (angles.max() - angles.min())

            for _ in range(needed):

                max_exp_id += 1
                new_exp = max_exp_id

                phi1 = np.random.uniform(0, 2 * np.pi)
                phi2 = np.random.uniform(0, 2 * np.pi)

                for idx, row in group_stats.iterrows():

                    new_row = {
                        "Experiment_ID": new_exp,
                        "Group_ID": group_id,
                        self.angle_col: row[self.angle_col],
                    }

                    t = theta[idx - group_stats.index.min()]

                    for feature in self.feature_cols:

                        mean_val = row[f"{feature}_mean"]
                        std_val = row[f"{feature}_std"]

                        if pd.isna(std_val) or std_val == 0:
                            residual = 0
                        else:
                            residual = std_val * (
                                0.6 * np.sin(t + phi1) +
                                0.4 * np.cos(t + phi2)
                            )

                        new_row[feature] = mean_val + residual

                    synthetic_rows.append(new_row)

        synthetic_df = pd.DataFrame(synthetic_rows)

        logger.info("Synthetic samples generated: %s", len(synthetic_df))

        return synthetic_df