import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from src.logging.log_utils import log_function


class DataLoader:
    def __init__(self, db_path: str) -> None:
        """
        Initialize the DataLoader with a path to the SQLite database.
        If the database file or its parent directories do not exist, they will be created.

        Args:
            db_path (str): Path to SQLite database file.
        """
        self.db_path = db_path
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @log_function
    def store_to_sqlite(
        self, 
        dataframes: Optional[Dict[str, pd.DataFrame]] = None, 
        store_index_tables: Optional[List[str]] = None
    ) -> None:
        """
        Save multiple DataFrames to a SQLite database and create indexes on all columns.

        Args:
            dataframes (dict, optional): Dictionary where keys are table names 
                                         and values are DataFrames to save.
            store_index_tables (list, optional): List of table names for which the 
                                                 DataFrame index should be stored as a column.
        """
        if not dataframes:
            print("No dataframes provided. Nothing to store.")
            return

        if store_index_tables is None:
            store_index_tables = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for table_name, df in dataframes.items():
            if not isinstance(df, pd.DataFrame):
                print(f"Skipped {table_name}: not a valid DataFrame.")
                continue

            index = table_name in store_index_tables

            df.to_sql(table_name, conn, index=index, if_exists="replace")
            print(f"Saved table '{table_name}' with shape {df.shape} to SQLite.")

            for col in df.columns:
                index_name = f"idx_{table_name}_{col}"
                try:
                    cursor.execute(
                        f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}"("{col}")'
                    )
                    print(f"Created index on {table_name}({col})")
                except sqlite3.OperationalError as e:
                    print(f"Failed to create index on {table_name}({col}): {e}")

        conn.commit()
        conn.close()

    @log_function
    def load_all_data_from_sqlite(self, store_index_tables=[
            "df_machine",
            "df_sensor",
            "df_machine_and_movement",
            "df_movement",
        ]) -> Dict[str, pd.DataFrame]:
        """
        Load all tables from the SQLite database into a dictionary of DataFrames.

        Tables listed in `store_index_tables` will have 'Time_[s]' set as the index
        if that column exists. Each table name is used as the key in the returned dictionary.

        Args:
            store_index_tables (list): List of table names for which 'Time_[s]' should be used as index.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping table names to their respective DataFrames.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        dataframes = {}
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

            if table in store_index_tables and "Time_[s]" in df.columns:
                df.set_index("Time_[s]", inplace=True)

            dataframes[table] = df

        conn.close()
        return dataframes

    @log_function
    def load_data_by_experiment_from_sqlite(
        self,
        experiment_id,
        store_index_tables=[
            "df_machine",
            "df_sensor",
            "df_machine_and_movement",
            "df_movement",
        ],
    ) -> Dict[str, pd.DataFrame]:
        """
        Load tables from the SQLite database filtered by a specific Experiment_ID.

        Only rows with the specified `experiment_id` are loaded. For tables listed
        in `store_index_tables`, 'Time_[s]' is set as the index if present.
        Returns a dictionary mapping table names to their filtered DataFrames.

        Args:
            experiment_id (int): The Experiment_ID to filter rows by.
            store_index_tables (list): List of table names for which 'Time_[s]' should be used as index.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping table names to their filtered DataFrames.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        dataframes = {}
        for table in tables:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE Experiment_ID = ?",
                conn,
                params=(experiment_id,),
            )
            if table in store_index_tables and "Time_[s]" in df.columns:
                df.set_index("Time_[s]", inplace=True)
            dataframes[table] = df

        conn.close()
        return dataframes
    
    @log_function
    def load_experiment_ids_from_sqlite(
        self):
        """
        Load Experiment_ID from the SQLite database.

        Returns:
            pd.DataFrame: panda dataframe of experiment ids.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        df = pd.read_sql_query(f"SELECT * FROM machine_and_movement", conn)
        conn.close()
        experiment_ids = df["Experiment_ID"].unique()
        return experiment_ids