import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.logging.log_utils import log_function, logger


class DataLoader:
    def __init__(
        self,
        output_dir: Path | str,
        df_arc: Optional[pd.DataFrame] = None,
        df_lin1: Optional[pd.DataFrame] = None,
        df_lin2: Optional[pd.DataFrame] = None,
        df_stl_arc: Optional[pd.DataFrame] = None,
        df_stl_lin1: Optional[pd.DataFrame] = None,
        df_stl_lin2: Optional[pd.DataFrame] = None,
        df_machine: Optional[pd.DataFrame] = None,
        df_sensor: Optional[pd.DataFrame] = None,
        df_movement: Optional[pd.DataFrame] = None,
        df_bending: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize DataLoader with optional DataFrames.
        Only non-None DataFrames will be saved.
        """

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store all DataFrames (including None)
        self.dataframes: Dict[str, Optional[pd.DataFrame]] = {
            "df_arc": df_arc,
            "df_lin1": df_lin1,
            "df_lin2": df_lin2,
            "df_stl_arc": df_stl_arc,
            "df_stl_lin1": df_stl_lin1,
            "df_stl_lin2": df_stl_lin2,
            "df_machine": df_machine,
            "df_sensor": df_sensor,
            "df_movement": df_movement,
            "df_bending": df_bending,
        }

        self._save_available()

    @log_function
    def _save_available(self) -> None:
        """
        Save only DataFrames that are not None.
        """

        for name, df in self.dataframes.items():

            if df is None:
                logger.info(f"Skipping '{name}' (None).")
                continue

            if df.empty:
                logger.warning(f"Skipping '{name}' (empty DataFrame).")
                continue

            file_name = name.replace("df_", "") + ".csv"
            file_path = self.output_dir / file_name

            df.to_csv(file_path, index=False)
            logger.info(f"Saved '{name}' to '{file_path}'.")