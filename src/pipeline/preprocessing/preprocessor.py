from pathlib import Path

from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.preprocessing.loader import DataLoader
from src.logging.log_utils import log_function


class STLGeometryPreprocessPipeline:
    """
    Pipeline that extracts, quality-checks, filters, transforms,
    but stores ONLY STL geometry-related tables.
    """

    @classmethod
    @log_function
    def run(
        cls,
        failed_experiment: list[int] | None = None,
        output_dir: Path | None = None,
    ):
        # ----------------------------
        # Paths (Path-composed)
        # ----------------------------
        pkl_file_path = Path("data") / "raw" / "experiments_process_and_results.pkl"
        if output_dir is None:
            output_dir = Path("data") / "processed" / "stl"

        # ----------------------------
        # Step 1: Extract all sections
        # ----------------------------
        extractor = DataExtractor(pkl_file_path=pkl_file_path)
        all_sections = extractor.get_all_sections()

        # Map sections to DataFrames
        section_names = [
            "arc", "lin1", "lin2",
            "stl_arc", "stl_lin1", "stl_lin2",
            "machine", "sensor", "movement", "bending"
        ]
        dfs = {}
        for section in section_names:
            df = all_sections[all_sections["Section"] == section].copy()
            df.drop(columns=["Section"], inplace=True)
            dfs[section] = df

        # ----------------------------
        # Step 2: Initialize transformer
        # ----------------------------
        transformer = DataTransformer(
            df_arc=dfs.get("arc"),
            df_lin1=dfs.get("lin1"),
            df_lin2=dfs.get("lin2"),
            df_stl_arc=dfs.get("stl_arc"),
            df_stl_lin1=dfs.get("stl_lin1"),
            df_stl_lin2=dfs.get("stl_lin2"),
            df_machine=dfs.get("machine"),
            df_sensor=dfs.get("sensor"),
            df_movement=dfs.get("movement"),
            df_bending=dfs.get("bending"),
        )

        # ----------------------------
        # Step 3: Full quality check on all tables
        # ----------------------------
        transformer.check_quality()

        # ----------------------------
        # Step 4: Remove failed experiments (all tables)
        # ----------------------------
        if failed_experiment:
            transformer.delete_failed_experiment(failed_experiment)

        # ----------------------------
        # Step 5: Transform STL tables only
        # ----------------------------
        stl_tables = [
            "df_stl_arc",
            "df_stl_lin1",
            "df_stl_lin2",
        ]
        transformer.normalize_data(normalized_table=stl_tables)
        transformer.nan_handler()

        # ----------------------------
        # Step 6: Retrieve STL geometry tables
        # ----------------------------
        (
            df_stl_arc,
            df_stl_lin1,
            df_stl_lin2,
            stl_linear_combined,
            all_geometry_stl,
        ) = transformer.get_stl_geometry_data()

        # ----------------------------
        # Step 7: Store STL CSVs ONLY
        # ----------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(output_dir=output_dir)

        stl_dataframes_to_save = {
            "stl_arc": df_stl_arc,
            "stl_lin1": df_stl_lin1,
            "stl_lin2": df_stl_lin2,
            "stl_linear_combined": stl_linear_combined,
            "all_geometry_stl": all_geometry_stl,
        }

        loader.store_to_csv(dataframes=stl_dataframes_to_save)


if __name__ == "__main__":
    STLGeometryPreprocessPipeline.run(
        failed_experiment=[1, 48, 166],
        output_dir=Path("data") / "processed" / "stl",
    )

    print("STL geometry preprocessing finished. CSVs saved.")
