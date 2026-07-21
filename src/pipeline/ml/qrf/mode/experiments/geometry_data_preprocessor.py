from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file was not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        "Only CSV and Parquet are supported."
    )


def parse_experiment_numbers(value: object) -> list[int]:
    """Convert Experiment_Number values to a list of integers."""
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]

    if pd.isna(value):
        return []

    return [int(value)]


def load_bending_setups(
    path: str | Path,
    excluded_experiments: list[int] | set[int] | None = None,
) -> pd.DataFrame:
    """Load bending setups, remove excluded experiments and add Group_ID."""
    excluded = set(excluded_experiments or [])

    bending_df = read_table(path).copy()

    if "Experiment_Number" not in bending_df.columns:
        raise KeyError("Missing required column: Experiment_Number")

    bending_df["Experiment_Number"] = (
        bending_df["Experiment_Number"]
        .apply(parse_experiment_numbers)
        .apply(
            lambda values: [
                experiment_id
                for experiment_id in values
                if experiment_id not in excluded
            ]
        )
    )

    bending_df = (
        bending_df[
            bending_df["Experiment_Number"].apply(bool)
        ]
        .reset_index(drop=True)
    )

    if "Group_ID" not in bending_df.columns:
        bending_df.insert(
            0,
            "Group_ID",
            range(1, len(bending_df) + 1),
        )

    return bending_df


REQUIRED_GEOMETRY_COLUMNS = {
    "Angle[degree]ORDistance[mm]",
    "Main-axis [mm]",
    "Secondary-axis [mm]",
}


def attach_group_id(
    geometry_df: pd.DataFrame,
    bending_setups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add Group_ID to geometry rows using Experiment_Number.

    bending_setups_df must contain:
        Group_ID
        Experiment_Number

    Experiment_Number in bending_setups_df may contain a list of experiments.
    """
    if "Group_ID" in geometry_df.columns:
        return geometry_df

    if "Experiment_Number" not in geometry_df.columns:
        raise KeyError(
            "Geometry data has neither 'Group_ID' nor "
            "'Experiment_Number', so Group_ID cannot be assigned."
        )

    required_bending_columns = {
        "Group_ID",
        "Experiment_Number",
    }
    missing = required_bending_columns.difference(
        bending_setups_df.columns
    )

    if missing:
        raise KeyError(
            "Bending setup data is missing columns: "
            f"{sorted(missing)}"
        )

    experiment_group_map = (
        bending_setups_df[
            ["Group_ID", "Experiment_Number"]
        ]
        .explode("Experiment_Number")
        .dropna(subset=["Experiment_Number"])
        .copy()
    )

    experiment_group_map["Experiment_Number"] = pd.to_numeric(
        experiment_group_map["Experiment_Number"],
        errors="raise",
    ).astype(int)

    geometry_df = geometry_df.copy()

    geometry_df["Experiment_Number"] = pd.to_numeric(
        geometry_df["Experiment_Number"],
        errors="raise",
    ).astype(int)

    geometry_df = geometry_df.merge(
        experiment_group_map,
        on="Experiment_Number",
        how="left",
        validate="many_to_one",
    )

    unmatched = geometry_df["Group_ID"].isna()

    if unmatched.any():
        missing_experiments = sorted(
            geometry_df.loc[
                unmatched,
                "Experiment_Number",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Some geometry experiments could not be mapped "
            "to a Group_ID. Missing Experiment_Number values: "
            f"{missing_experiments[:20]}"
        )

    geometry_df["Group_ID"] = (
        geometry_df["Group_ID"].astype(int)
    )

    return geometry_df


def load_geometry_data(
    path: str | Path,
    bending_setups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load geometry data and ensure it contains Group_ID.
    """
    geometry_df = read_table(path).copy()

    missing_geometry_columns = (
        REQUIRED_GEOMETRY_COLUMNS.difference(
            geometry_df.columns
        )
    )

    if missing_geometry_columns:
        raise KeyError(
            "Geometry data is missing required columns: "
            f"{sorted(missing_geometry_columns)}"
        )

    geometry_df = attach_group_id(
        geometry_df=geometry_df,
        bending_setups_df=bending_setups_df,
    )

    return geometry_df

def load_selected_geometry_source(
    project_root: str | Path,
    paths_config: dict,
    data_config: dict,
    bending_setups_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, Path]:
    """
    Load the selected geometry source and attach Group_ID.
    """
    project_root = Path(project_root).resolve()

    source_name = data_config.get("geometry_source")

    if not source_name:
        raise KeyError(
            "Missing config value: data.geometry_source"
        )

    geometry_sources = paths_config.get(
        "geometry_sources"
    )

    if not geometry_sources:
        raise KeyError(
            "Missing config section: paths.geometry_sources"
        )

    if source_name not in geometry_sources:
        raise ValueError(
            f"Unknown geometry source {source_name!r}. "
            f"Available sources: "
            f"{sorted(geometry_sources.keys())}"
        )

    geometry_path = Path(
        geometry_sources[source_name]
    )

    if not geometry_path.is_absolute():
        geometry_path = project_root / geometry_path

    geometry_path = geometry_path.resolve()

    geometry_df = load_geometry_data(
        path=geometry_path,
        bending_setups_df=bending_setups_df,
    )

    return geometry_df, source_name, geometry_path