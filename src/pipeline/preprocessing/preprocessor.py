from pathlib import Path

import pandas as pd

from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.preprocessing.loader import DataLoader

from src.logging.log_utils import log_function, logger


class DataPreprocessPipeline:
    """
    Pipeline for extracting and saving STL-suitable geometry data to CSV.

    Steps:
    1. Extract raw data
    2. Transform data (quality check, filtering, normalization, NaN handling)
    3. Save processed datasets via DataLoader
    """

    @classmethod
    @log_function
    def run(
        cls,
        failed_experiment: list[int] | None,
        output_dir: Path | str,
        eliminated_columns: dict[str, list[str]] | None = None,
        normalized_tables: list[str] | None = None,
        nan_handler: bool = True,
        bending_setups_path: Path | str = Path("data/raw/unique_bending_setups.csv"),
    ) -> None:
        """
        Execute preprocessing pipeline and persist results.

        Args:
            failed_experiment: Experiment IDs to exclude
            output_dir: Destination directory
            eliminated_columns: {df_name: [col1, col2]}
            normalized_tables: list of df names to normalize
            nan_handler: Whether to drop NaN columns
            bending_setups_path: CSV of unique bending setup groups
        """

        # --- 1. Extract ---
        extractor = DataExtractor()
        dfs = extractor.get_all_bending_setups()

        # --- 2. Transform ---
        transformer = DataTransformer(**dfs)

        # 2.1 Quality check
        transformer.check_quality()

        # 2.2 Remove failed experiments
        if failed_experiment:
            transformer.delete_failed_experiment(failed_experiment)

        # 2.3 Eliminate columns (NEW)
        if eliminated_columns:
            for df_name, cols in eliminated_columns.items():
                for col in cols:
                    transformer.eliminate_column(df_name, col)

        # 2.4 Normalize (NEW: configurable)
        if normalized_tables:
            transformer.normalize_data(normalized_tables)

        # 2.5 Handle NaNs
        if nan_handler:
            transformer.nan_handler()

        # --- 3. Load ---
        DataLoader(
            output_dir=output_dir,
            df_arc=transformer.df_arc,
            df_lin1=transformer.df_lin1,
            df_lin2=transformer.df_lin2,
            df_stl_arc=transformer.df_stl_arc,
            df_stl_lin1=transformer.df_stl_lin1,
            df_stl_lin2=transformer.df_stl_lin2,
            df_machine=transformer.df_machine,
            df_sensor=transformer.df_sensor,
            df_movement=transformer.df_movement,
            df_bending=transformer.df_bending,
            # Derived tables read by the RF augmentation, QRF, HGP and dashboard steps
            df_geometry=transformer.df_arc,
            df_machine_and_movement=cls._build_machine_and_movement(
                transformer.df_machine, transformer.df_movement
            ),
            df_processed_bending_setup=cls._build_processed_bending_setup(
                bending_setups_path
            ),
        )

    @staticmethod
    def _build_machine_and_movement(
        df_machine: pd.DataFrame | None,
        df_movement: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """
        Combine machine and movement signals side by side.

        Both tables share the same Experiment_ID / Time_[s] rows, so the
        movement keys are dropped to keep a single copy of each.
        """
        if df_machine is None or df_movement is None:
            return None

        df_machine = df_machine.reset_index(drop=True)
        df_movement = df_movement.reset_index(drop=True)

        key_columns = ["Experiment_ID", "Time_[s]"]
        if not df_machine[key_columns].equals(df_movement[key_columns]):
            raise ValueError(
                "Machine and movement rows are not aligned on "
                "Experiment_ID and Time_[s]."
            )

        return pd.concat(
            [df_machine, df_movement.drop(columns=key_columns)], axis=1
        )

    @staticmethod
    def _build_processed_bending_setup(
        bending_setups_path: Path | str,
    ) -> pd.DataFrame | None:
        """
        Load the unique bending setup groups with Group_ID as first column.
        """
        bending_setups_path = Path(bending_setups_path)
        if not bending_setups_path.exists():
            logger.warning(
                f"Bending setup groups not found at '{bending_setups_path}'; "
                "skipping processed_bending_setup.csv."
            )
            return None

        df_setups = pd.read_csv(bending_setups_path)
        columns = ["Group_ID"] + [
            col for col in df_setups.columns if col != "Group_ID"
        ]
        return df_setups[columns]
          