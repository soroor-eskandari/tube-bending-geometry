from pathlib import Path

from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.preprocessing.loader import DataLoader

from src.logging.log_utils import log_function


class DataPreprocessPipeline:
    """
    Pipeline for extracting, transforming, and loading bending setup and process data.

    This class orchestrates the full ETL process:
    1. Extracts bending setup and machine/process data using DataExtractor.
    2. Transforms the data with DataTransformer (quality check, remove failed experiments, normalization, NaN handling).
    3. Loads the final DataFrames into a SQLite database using DataLoader.
    """

    @classmethod
    @log_function
    def run(
        cls,
        failed_experiment: list[int] | None,
        eliminated_columns: dict[str, list[str]] | None,
        normalized_tables: list[str] | None,
        correlation_matrices: list[str] | None,
        nan_handler: bool = True,
    ):
        """
        Execute the full preprocessing pipeline: extraction, transformation, and loading.

        Steps:
        ------
        1. Extraction:
            - Use DataExtractor to load all bending setups into DataFrames.
        2. Transformation:
            - Create a DataTransformer instance with the extracted DataFrames.
            - Check data quality and convert types.
            - Remove failed experiments (hardcoded IDs [1, 48, 166]).
            - Normalize numeric data.
            - Drop all-NaN columns.
            - Retrieve processed process and geometry DataFrames.
            - Eliminate columns that are always constant(PRESSURE-DIE_LEFT_AXIAL_Movement_[mm], COLLET_ROTATING_Movement_[mm])
        3. Loading:
            - Collect selected DataFrames into a dictionary.
            - Use DataLoader to save them to SQLite.
            - Certain tables store the index as a column for query convenience.

        Args:
            None

        Returns:
            None: The pipeline updates and saves the DataFrames to the SQLite database.
        """
        extractor = DataExtractor()
        dfs = extractor.get_all_bending_setups()

        transformer = DataTransformer(**dfs)
        transformer.check_quality()
        if failed_experiment:
            transformer.delete_failed_experiment(failed_experiment=failed_experiment)

        if eliminated_columns:
            pairs = [
                (key, item)
                for key, values in eliminated_columns.items()
                for item in values
            ]
            for tabel_name, column_name in pairs:
                transformer.eliminate_column(
                    df_name=tabel_name, column_name=column_name
                )

        if normalized_tables:  
            transformer.normalize_data(normalized_table=normalized_tables)

        if nan_handler:
            transformer.nan_handler()

        if correlation_matrices:
            transformer.save_correlation_matrices(tables=correlation_matrices)

        df_machine_and_movement, df_sensor, df_machine, df_movement = (
            transformer.get_process_data()
        )
        df_arc, df_lin1, df_lin2, linear_df, all_geometry_data = (
            transformer.get_geometry_data()
        )
        transformer.get_bending_setup()

        loader = DataLoader("data/processed/tube_geometry.db")
        dataframes = {
            "machine_and_movement": df_machine_and_movement,
            "linear1": df_lin1,
            "linear2": df_lin2,
            "arc": df_arc,
            "movement": df_movement,
        }
        loader.store_to_sqlite(
            dataframes=dataframes,
            store_index_tables=[
                "machine_and_movement",
                "movement",
            ],
        )


class STLGeometryPreprocessPipeline:
    """
    Pipeline for extracting and saving STL-suitable geometry data to CSV.

    Steps:
    1. Extract STL-suitable geometry data using DataExtractor.
    2. Transform data with DataTransformer (type coercion, remove failed experiments, NaN handling).
    3. Save arc/lin1/lin2 STL geometry CSVs plus combined datasets.
    """

    @classmethod
    @log_function
    def run(
        cls,
        failed_experiment: list[int] | None,
        output_dir: Path | str,
        nan_handler: bool = True,
    ) -> None:
        """
        Execute the STL geometry preprocessing pipeline and write CSV files.

        Args:
            failed_experiment (list[int] | None): Experiment IDs to exclude.
            output_dir (Path | str): Destination directory for CSV files.
            nan_handler (bool): Whether to drop all-NaN columns. Defaults to True.
        """
        extractor = DataExtractor()
        dfs = extractor.get_all_bending_setups()

        transformer = DataTransformer(**dfs)
        transformer.check_quality()

        if failed_experiment:
            transformer.delete_failed_experiment(failed_experiment=failed_experiment)

        if nan_handler:
            transformer.nan_handler()

        df_arc = transformer.df_arc
        df_bending = transformer.df_bending

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df_arc.to_csv(output_path / "geometry_data.csv", index=False)
        df_bending.to_csv(output_path / "bending_data.csv", index=False)
 

