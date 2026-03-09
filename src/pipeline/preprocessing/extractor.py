import pickle
import pandas as pd

from src.logging.log_utils import log_function


class DataExtractor:
    """
    Class to load and extract experiment data from a pickled dictionary.

    Attributes:
        loaded_dict (dict): Dictionary loaded from 'experiments_process_and_results.pkl' containing all experiment data.
    """

    def __init__(self) -> None:
        """Load the experiments dictionary from the pickle file into self.loaded_dict."""
        with open("data/raw/experiments_process_and_results.pkl", "rb") as f:
            self.loaded_dict = pickle.load(f)

    def _take_bending_setups(
        self, experiments_process_and_results, process_part
    ) -> pd.DataFrame:
        """
        Extract all bending setups for a specific machine part as a DataFrame.

        Adds an 'Experiment_ID' column if not already present and ensures it is the first column.

        Args:
            experiments_process_and_results (dict): The loaded experiments dictionary.
            process_part (str): Key for the machine part or bending setup to extract.

        Returns:
            pd.DataFrame: Concatenated DataFrame containing all setups for the specified machine part.
        """
        if process_part != "bending_setups":
            df_list = []
            for (
                experiment_name,
                experiment_data,
            ) in experiments_process_and_results.items():
                df = experiment_data[process_part].copy()
                if "Experiment_ID" in df.columns:
                    cols = ["Experiment_ID"] + [
                        col for col in df.columns if col != "Experiment_ID"
                    ]
                    df = df[cols]
                else:
                    df.insert(0, "Experiment_ID", experiment_name)

                df_list.append(df)

            result = pd.concat(df_list, ignore_index=True)
            result["Experiment_ID"] = result["Experiment_ID"].str.replace("Exp_", "")
            return result

        result = pd.concat(
            [
                experiments_process_and_results[experiment_name]["bending_setups"]
                for experiment_name in experiments_process_and_results.keys()
            ],
            ignore_index=True,
        )
        result.columns = result.columns.str.replace("Experiment", "Experiment_ID")
        return result

    @log_function
    def get_part_bending_setups(self, process_part):
        """
        Return bending setups for a specific machine part from the loaded dictionary.

        Args:
            process_part (str): Key for the machine part or bending setup to extract.

        Returns:
            pd.DataFrame: DataFrame containing all setups for the specified machine part.
        """
        return self._take_bending_setups(self.loaded_dict, process_part)

    @log_function
    def get_all_bending_setups(self):
        """
        Return all bending setups from the loaded dictionary as a dictionary of DataFrames.

        Returns:
            dict[str, pd.DataFrame]: Dictionary where keys are DataFrame names (e.g., 'df_arc', 'df_lin1')
                                     and values are the corresponding bending setup DataFrames.
        """
        return {
            "df_arc": self._take_bending_setups(
                self.loaded_dict, "geometry_data_key_characteristics_arc"
            ),
            "df_lin1": self._take_bending_setups(
                self.loaded_dict, "geometry_data_key_characteristics_linear_1"
            ),
            "df_lin2": self._take_bending_setups(
                self.loaded_dict, "geometry_data_key_characteristics_linear_2"
            ),
            "df_stl_arc": self._take_bending_setups(
                self.loaded_dict, "geometry_data_stl_suitable_arc"
            ),
            "df_stl_lin1": self._take_bending_setups(
                self.loaded_dict, "geometry_data_stl_suitable_linear_1"
            ),
            "df_stl_lin2": self._take_bending_setups(
                self.loaded_dict, "geometry_data_stl_suitable_linear_2"
            ),
            "df_machine": self._take_bending_setups(
                self.loaded_dict, "process_parameters_loads_machine"
            ),
            "df_sensor": self._take_bending_setups(
                self.loaded_dict, "process_parameters_loads_sensor"
            ),
            "df_movement": self._take_bending_setups(
                self.loaded_dict, "process_parameters_movements"
            ),
            "df_bending": self._take_bending_setups(self.loaded_dict, "bending_setups"),
        }

    @log_function
    def get_experiment_by_number(self, experiment_number):
        """
        Retrieve a specific experiment from the loaded dictionary by its number.

        Args:
            experiment_number (int): Number of the experiment to retrieve.

        Returns:
            dict: Dictionary containing data for the specified experiment, or None if not found.
        """
        return self.loaded_dict.get(f"Exp_{experiment_number}")

    @log_function
    def get_experiment_keys(self):
        """
        Print the keys of a sample experiment (hardcoded experiment number 2)
        to inspect the structure of the dictionary.
        """
        for key in list(self.loaded_dict.get(f"Exp_{2}")):
            print(key)