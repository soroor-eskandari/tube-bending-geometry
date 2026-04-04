from pathlib import Path

from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.preprocessing.loader import DataLoader

from src.logging.log_utils import log_function


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
    ) -> None:
        """
        Execute preprocessing pipeline and persist results.

        Args:
            failed_experiment: Experiment IDs to exclude
            output_dir: Destination directory
            eliminated_columns: {df_name: [col1, col2]}
            normalized_tables: list of df names to normalize
            nan_handler: Whether to drop NaN columns
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
        )
          