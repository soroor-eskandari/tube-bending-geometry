from src.logging.log_utils import log_function
from pathlib import Path
import pickle
import pandas as pd


class DataExtractor:
    """
    Class to load and extract experiment data from a pickled dictionary.

    Attributes:
        loaded_dict (dict): Dictionary loaded from
        'experiments_process_and_results.pkl' containing all experiment data.
    """

    def __init__(self, pkl_file_path) -> None:
        """Load serialized experiment data into memory."""
        
        with open(pkl_file_path, 'rb') as f:
            self._store = pickle.load(f)

    def _build_frame(self, source: dict, section: str) -> pd.DataFrame:
        """
        Construct a unified DataFrame for a given experiment section.
        """
        tables = []

        if section == "bending_setups":
            for exp_key in source:
                tables.append(source[exp_key][section])

            combined = pd.concat(tables, ignore_index=True)
            combined.columns = combined.columns.str.replace(
                "Experiment", "Experiment_ID"
            )
            return combined

        for exp_key, exp_blob in source.items():
            df = exp_blob[section].copy()
            df.insert(0, "Experiment_ID", exp_key.replace("Exp_", ""))
            tables.append(df)

        return pd.concat(tables, ignore_index=True)

    @log_function
    def get_section(self, section_name: str) -> pd.DataFrame:
        """Return a DataFrame for a single experiment section."""
        return self._build_frame(self._store, section_name)

    @log_function
    def get_all_sections(self) -> pd.DataFrame:
        """
        Return all experiment sections stacked into one DataFrame,
        with an additional 'Section' column.
        """
        section_map = {
            "arc": "geometry_data_key_characteristics_arc",
            "lin1": "geometry_data_key_characteristics_linear_1",
            "lin2": "geometry_data_key_characteristics_linear_2",
            "stl_arc": "geometry_data_stl_suitable_arc",
            "stl_lin1": "geometry_data_stl_suitable_linear_1",
            "stl_lin2": "geometry_data_stl_suitable_linear_2",
            "machine": "process_parameters_loads_machine",
            "sensor": "process_parameters_loads_sensor",
            "movement": "process_parameters_movements",
            "bending": "bending_setups",
        }

        frames = []
        for label, key in section_map.items():
            df = self._build_frame(self._store, key)
            df.insert(0, "Section", label)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    @log_function
    def get_experiment(self, experiment_id: int) -> pd.DataFrame:
        """
        Return all data for a single experiment as a normalized DataFrame.
        """
        exp_key = f"Exp_{experiment_id}"
        payload = self._store.get(exp_key, {})

        records = []
        for block_name, block_value in payload.items():
            frame = pd.DataFrame(block_value)
            frame.insert(0, "Block", block_name)
            frame.insert(0, "Experiment_ID", experiment_id)
            records.append(frame)

        return pd.concat(records, ignore_index=True)
