import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from src.logging.log_utils import log_function, logger


class DataLoader:
    def __init__(self, output_dir: str) -> None:
        """
        Initialize the DataLoaderCSV with a directory path to store CSV files.
        Creates the directory if it does not exist.

        Args:
            output_dir (str): Path to directory where CSV files will be stored.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CSV output directory set to: {self.output_dir}")

    @log_function
    def store_to_csv(
        self, 
        dataframes: Optional[Dict[str, pd.DataFrame]] = None
    ) -> None:
        """
        Save multiple DataFrames as CSV files in the output directory.

        Args:
            dataframes (dict, optional): Dictionary where keys are filenames
                                         and values are DataFrames to save.
        """
        if not dataframes:
            logger.warning("No dataframes provided. Nothing to store.")
            return

        for file_name, df in dataframes.items():
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Skipped {file_name}: not a valid DataFrame.")
                continue

            file_path = self.output_dir / f"{file_name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved CSV '{file_path}' with shape {df.shape}")

    @log_function
    def load_all_csv(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the output directory into a dictionary of DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filenames (without .csv) to DataFrames.
        """
        dataframes = {}
        for csv_file in self.output_dir.glob("*.csv"):
            df_name = csv_file.stem
            df = pd.read_csv(csv_file)
            dataframes[df_name] = df
            logger.info(f"Loaded CSV '{csv_file}' with shape {df.shape}")

        return dataframes

    @log_function
    def load_csv_by_experiment(self, experiment_id: int) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files and filter rows by Experiment_ID.

        Args:
            experiment_id (int): Experiment_ID to filter.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of filtered DataFrames.
        """
        dataframes = {}
        for csv_file in self.output_dir.glob("*.csv"):
            df_name = csv_file.stem
            df = pd.read_csv(csv_file)
            if "Experiment_ID" in df.columns:
                df = df[df["Experiment_ID"] == experiment_id]
            dataframes[df_name] = df
            logger.info(f"Loaded and filtered CSV '{csv_file}' for Experiment_ID={experiment_id} (shape={df.shape})")
        return dataframes

    @log_function
    def load_experiment_ids(self) -> List[int]:
        """
        Load unique Experiment_IDs from all CSV files that contain this column.

        Returns:
            List[int]: Sorted list of unique experiment IDs.
        """
        experiment_ids = set()
        for csv_file in self.output_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            if "Experiment_ID" in df.columns:
                experiment_ids.update(df["Experiment_ID"].unique())
        experiment_ids = sorted(list(experiment_ids))
        logger.info(f"Found {len(experiment_ids)} unique Experiment_IDs across CSV files.")
        return experiment_ids
