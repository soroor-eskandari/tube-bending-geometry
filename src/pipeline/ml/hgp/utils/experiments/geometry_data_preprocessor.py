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
    Add Group_ID to geometry rows by mapping:

        geometry_df["Experiment_ID"]
            ->
        bending_setups_df["Experiment_Number"]

    Experiment_Number in bending_setups_df may contain a list.
    """
    if "Group_ID" in geometry_df.columns:
        return geometry_df

    if "group_id" in geometry_df.columns:
        geometry_df = geometry_df.copy()
        geometry_df["Group_ID"] = pd.to_numeric(
            geometry_df["group_id"],
            errors="raise",
        ).astype(int)

        return geometry_df

    geometry_experiment_column = "Experiment_ID"
    setup_experiment_column = "Experiment_Number"

    if geometry_experiment_column not in geometry_df.columns:
        raise KeyError(
            "Geometry data is missing the required column "
            f"{geometry_experiment_column!r}. "
            f"Available columns: {geometry_df.columns.tolist()}"
        )

    required_bending_columns = {
        "Group_ID",
        setup_experiment_column,
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
            [
                "Group_ID",
                setup_experiment_column,
            ]
        ]
        .explode(setup_experiment_column)
        .dropna(subset=[setup_experiment_column])
        .copy()
    )

    experiment_group_map[setup_experiment_column] = (
        pd.to_numeric(
            experiment_group_map[setup_experiment_column],
            errors="raise",
        )
        .astype(int)
    )

    geometry_df = geometry_df.copy()

    geometry_df[geometry_experiment_column] = (
        pd.to_numeric(
            geometry_df[geometry_experiment_column],
            errors="raise",
        )
        .astype(int)
    )

    experiment_group_map = experiment_group_map.rename(
        columns={
            setup_experiment_column: geometry_experiment_column
        }
    )

    geometry_df = geometry_df.merge(
        experiment_group_map,
        on=geometry_experiment_column,
        how="left",
        validate="many_to_one",
    )

    unmatched = geometry_df["Group_ID"].isna()

    if unmatched.any():
        missing_experiments = sorted(
            geometry_df.loc[
                unmatched,
                geometry_experiment_column,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Some geometry experiments could not be mapped "
            "to a Group_ID. Missing Experiment_ID values: "
            f"{missing_experiments[:20]}"
        )

    geometry_df["Group_ID"] = (
        geometry_df["Group_ID"].astype(int)
    )

    return geometry_df


def merge_group_setup_features(
    geometry_df: pd.DataFrame,
    bending_setups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach bending-setup features to every geometry row by Group_ID.

    Identification columns are not copied as model features. Every retained
    setup feature must be constant inside each Group_ID. Existing geometry
    columns with the same names are replaced by the setup-table values to
    avoid pandas _x/_y suffixes.
    """
    group_column = "Group_ID"

    if group_column not in geometry_df.columns:
        raise KeyError(
            "Geometry data is missing required column 'Group_ID'."
        )

    if group_column not in bending_setups_df.columns:
        raise KeyError(
            "Bending setup data is missing required column 'Group_ID'."
        )

    excluded_setup_columns = {
        group_column,
        "Experiment_ID",
        "Experiment_Number",
        "Tube_numbers",
    }

    setup_feature_columns = [
        column
        for column in bending_setups_df.columns
        if column not in excluded_setup_columns
    ]

    if not setup_feature_columns:
        raise ValueError(
            "No bending setup feature columns are available for merging."
        )

    setup_working_df = bending_setups_df[
        [
            group_column,
            *setup_feature_columns,
        ]
    ].copy()

    setup_working_df[group_column] = pd.to_numeric(
        setup_working_df[group_column],
        errors="raise",
    ).astype(int)

    geometry_working_df = geometry_df.copy()
    geometry_working_df[group_column] = pd.to_numeric(
        geometry_working_df[group_column],
        errors="raise",
    ).astype(int)

    feature_unique_counts = (
        setup_working_df
        .groupby(
            group_column,
            sort=False,
        )[setup_feature_columns]
        .nunique(dropna=False)
    )

    inconsistent_mask = feature_unique_counts.gt(1)

    if inconsistent_mask.any().any():
        inconsistent_columns = (
            inconsistent_mask
            .any(axis=0)
            .loc[lambda values: values]
            .index
            .tolist()
        )

        inconsistent_groups = sorted(
            feature_unique_counts.index[
                inconsistent_mask.any(axis=1)
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "Bending setup features must be constant within each "
            "Group_ID. "
            f"Inconsistent columns: {inconsistent_columns}. "
            f"Affected groups: {inconsistent_groups[:20]}"
        )

    setup_group_df = (
        setup_working_df
        .drop_duplicates(
            subset=[group_column]
        )
        .reset_index(drop=True)
    )

    # Replace already-existing versions of setup features and prevent
    # automatic pandas suffixes such as _x and _y.
    overlapping_columns = [
        column
        for column in setup_feature_columns
        if column in geometry_working_df.columns
    ]

    if overlapping_columns:
        geometry_working_df = geometry_working_df.drop(
            columns=overlapping_columns
        )

    merged_df = geometry_working_df.merge(
        setup_group_df,
        on=group_column,
        how="left",
        validate="many_to_one",
    )

    matched_groups = set(
        setup_group_df[group_column]
        .astype(int)
        .tolist()
    )
    geometry_groups = set(
        merged_df[group_column]
        .astype(int)
        .tolist()
    )
    missing_groups = sorted(
        geometry_groups.difference(matched_groups)
    )

    if missing_groups:
        raise ValueError(
            "No bending setup row was found for geometry Group_ID "
            f"values: {missing_groups[:20]}"
        )

    return merged_df

def load_geometry_data(
    path: str | Path,
    bending_setups_df: pd.DataFrame,
    excluded_experiments: list[int] | set[int] | None = None,
) -> pd.DataFrame:
    geometry_df = read_table(path).copy()

    geometry_df.columns = geometry_df.columns.str.strip()

    missing_geometry_columns = (
        REQUIRED_GEOMETRY_COLUMNS.difference(
            geometry_df.columns
        )
    )

    if missing_geometry_columns:
        raise KeyError(
            "Geometry data is missing required columns: "
            f"{sorted(missing_geometry_columns)}. "
            f"Available columns: {geometry_df.columns.tolist()}"
        )

    excluded = set(excluded_experiments or [])

    # Delete excluded experiments before assigning Group_ID.
    if excluded:
        if "Experiment_ID" not in geometry_df.columns:
            raise KeyError(
                "Cannot remove excluded experiments because "
                "'Experiment_ID' is missing from geometry data."
            )

        geometry_df["Experiment_ID"] = pd.to_numeric(
            geometry_df["Experiment_ID"],
            errors="raise",
        ).astype(int)

        geometry_df = geometry_df[
            ~geometry_df["Experiment_ID"].isin(excluded)
        ].copy()

    geometry_df = attach_group_id(
        geometry_df=geometry_df,
        bending_setups_df=bending_setups_df,
    )

    geometry_df = merge_group_setup_features(
        geometry_df=geometry_df,
        bending_setups_df=bending_setups_df,
    )

    return geometry_df.reset_index(drop=True)

def load_selected_geometry_source(
    project_root: str | Path,
    paths_config: dict,
    data_config: dict,
    bending_setups_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, Path]:
    """
    Load the selected geometry source, attach Group_ID and merge setup features.
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
        excluded_experiments=data_config.get(
            "excluded_experiments",
            [],
        ),
    )

    return geometry_df, source_name, geometry_path
