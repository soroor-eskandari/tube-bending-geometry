from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Split-generation configuration
# ============================================================

SPLIT_FEATURES = (
    "Mandrel retraction timing",
    "Collet boost",
)

MIN_COMBINATION_SIZE = 1
MAX_COMBINATION_SIZE = 3

DEFAULT_CONFIG_NAME = "qrf_split_ranking.yaml"

DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "src/pipeline/ml/qrf/data/various_splits.parquet"
)


# ============================================================
# Dynamic project paths
# ============================================================

def find_repository_root(start: Path) -> Path:
    """
    Find the repository root without using an absolute path.

    The repository root is identified using either:
      1. the directory name ``tube-bending-geometry``;
      2. a directory containing ``src/pipeline/ml/qrf``.
    """
    current = start.resolve()

    while True:
        expected_qrf_directory = (
            current
            / "src"
            / "pipeline"
            / "ml"
            / "qrf"
        )

        if (
            current.name == "tube-bending-geometry"
            or expected_qrf_directory.exists()
        ):
            return current

        if current == current.parent:
            raise RuntimeError(
                "Could not locate the repository root. "
                "Expected a parent directory containing "
                "'src/pipeline/ml/qrf'."
            )

        current = current.parent


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = find_repository_root(
    SCRIPT_PATH.parent
)

QRF_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "pipeline"
    / "ml"
    / "qrf"
)

for import_path in (
    REPOSITORY_ROOT,
    QRF_ROOT,
):
    import_value = str(import_path)

    if import_value not in sys.path:
        sys.path.insert(
            0,
            import_value,
        )


from src.pipeline.ml.qrf.utils.experiments.geometry_data_preprocessor import (
    load_bending_setups,
)


# ============================================================
# Generic helpers
# ============================================================

def resolve_path(
    repository_root: Path,
    value: str | Path,
) -> Path:
    """
    Resolve a path relative to the repository root.
    """
    path = Path(value)

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def load_yaml_config(
    config_path: Path,
) -> dict:
    """
    Load the QRF YAML configuration.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"QRF config was not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            "The QRF config must contain a YAML dictionary."
        )

    return config


def decode_experiment_ids(
    value: object,
) -> list[int]:
    """
    Convert Experiment_Number content to a clean list of integers.
    """
    if value is None:
        return []

    if isinstance(value, str):
        stripped_value = value.strip()

        if not stripped_value:
            return []

        try:
            value = json.loads(
                stripped_value
            )
        except json.JSONDecodeError:
            value = ast.literal_eval(
                stripped_value
            )

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            pd.Series,
        ),
    ):
        output: list[int] = []

        for item in value:
            if pd.notna(item):
                output.append(
                    int(item)
                )

        return sorted(
            set(output)
        )

    if pd.isna(value):
        return []

    return [int(value)]


def clean_scalar_for_name(
    value: object,
) -> str:
    """
    Format feature values consistently inside split names.
    """
    if pd.isna(value):
        return "NaN"

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        float_value = float(value)

        if float_value.is_integer():
            return str(
                int(float_value)
            )

        return format(
            float_value,
            ".15g",
        )

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return str(
            int(value)
        )

    return str(value).strip()


def calculate_test_percentage(
    train_count: int,
    test_count: int,
) -> float:
    """
    Calculate only the test-set percentage.

    Example:
        train_count=80
        test_count=20

    returns:
        20.0
    """
    total_count = (
        int(train_count)
        + int(test_count)
    )

    if total_count <= 0:
        raise ValueError(
            "Cannot calculate test percentage "
            "for an empty dataset."
        )

    return round(
        100.0
        * int(test_count)
        / total_count,
        2,
    )

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataframe_name: str,
) -> None:
    """
    Validate required DataFrame columns.
    """
    missing_columns = set(
        required_columns
    ).difference(
        dataframe.columns
    )

    if missing_columns:
        raise KeyError(
            f"{dataframe_name} is missing columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )


# ============================================================
# Atomic split conditions
# ============================================================

def build_atomic_conditions(
    bending_setups_df: pd.DataFrame,
    split_features: Iterable[str],
) -> list[dict]:
    """
    Build one atomic condition for every unique feature value.

    Example atomic conditions:

        Mandrel retraction timing=5
        Collet boost=0.9
    """
    conditions: list[dict] = []

    for feature in split_features:
        feature_values = (
            bending_setups_df[feature]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        try:
            feature_values = sorted(
                feature_values
            )
        except TypeError:
            feature_values = sorted(
                feature_values,
                key=lambda value: str(value),
            )

        for value in feature_values:
            matching_group_ids = sorted(
                bending_setups_df.loc[
                    bending_setups_df[
                        feature
                    ].eq(value),
                    "Group_ID",
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            if not matching_group_ids:
                continue

            condition_name = (
                f"{feature}="
                f"{clean_scalar_for_name(value)}"
            )

            conditions.append(
                {
                    "feature": feature,
                    "value": value,
                    "condition_name": (
                        condition_name
                    ),
                    "group_ids": (
                        matching_group_ids
                    ),
                }
            )

    if not conditions:
        raise ValueError(
            "No split conditions could be generated from "
            f"features {list(split_features)}."
        )

    return conditions


# ============================================================
# Split generation
# ============================================================

def build_split_name(
    selected_conditions: tuple[dict, ...],
) -> str:
    """
    Build a readable name for one held-out feature value.
    """
    if len(selected_conditions) != 1:
        raise ValueError(
            "Exactly one condition is required for each split."
        )

    condition = selected_conditions[0]

    return (
        f"{condition['feature']}="
        f"{clean_scalar_for_name(condition['value'])}"
    )


def collect_group_experiment_ids(
    bending_setups_df: pd.DataFrame,
    group_ids: Iterable[int],
) -> list[int]:
    """
    Return all unique experiment IDs belonging to selected groups.
    """
    selected_group_ids = {
        int(group_id)
        for group_id in group_ids
    }

    experiment_values = (
        bending_setups_df.loc[
            bending_setups_df[
                "Group_ID"
            ].isin(selected_group_ids),
            "Experiment_Number",
        ]
        .tolist()
    )

    experiment_ids: set[int] = set()

    for value in experiment_values:
        experiment_ids.update(
            decode_experiment_ids(value)
        )

    return sorted(experiment_ids)


def generate_split_metadata(
    bending_setups_df: pd.DataFrame,
    *,
    split_features: tuple[str, ...],
    minimum_combination_size: int = 1,
    maximum_combination_size: int = 3,
) -> pd.DataFrame:
    """
    Generate one leave-one-feature-value-out split per unique value.

    For every unique value of every feature in ``split_features``:
      - groups having that value are assigned to test;
      - all remaining groups are assigned to train.

    The combination-size parameters are retained only for backward
    compatibility with existing callers and do not affect the result.
    """
    validate_required_columns(
        dataframe=bending_setups_df,
        required_columns={
            "Group_ID",
            "Experiment_Number",
            *split_features,
        },
        dataframe_name="Bending setups",
    )

    bending_setups_df = bending_setups_df.copy()
    bending_setups_df["Group_ID"] = pd.to_numeric(
        bending_setups_df["Group_ID"],
        errors="raise",
    ).astype(int)
    bending_setups_df["Experiment_Number"] = (
        bending_setups_df["Experiment_Number"].apply(
            decode_experiment_ids
        )
    )

    all_group_ids = sorted(
        bending_setups_df["Group_ID"]
        .astype(int)
        .unique()
        .tolist()
    )
    all_experiment_ids = collect_group_experiment_ids(
        bending_setups_df=bending_setups_df,
        group_ids=all_group_ids,
    )

    if len(all_group_ids) < 2:
        raise ValueError(
            "At least two groups are required to create "
            "non-empty train and test sets."
        )

    if len(all_experiment_ids) < 2:
        raise ValueError(
            "At least two experiments are required to create "
            "non-empty train and test sets."
        )

    atomic_conditions = build_atomic_conditions(
        bending_setups_df=bending_setups_df,
        split_features=split_features,
    )

    candidate_rows: list[dict] = []
    observed_split_keys: set[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = set()

    for condition in atomic_conditions:
        selected_conditions = (condition,)

        test_group_ids = sorted(
            int(group_id)
            for group_id in condition["group_ids"]
        )
        train_group_ids = sorted(
            set(all_group_ids).difference(test_group_ids)
        )

        if not train_group_ids or not test_group_ids:
            continue

        train_experiment_ids = collect_group_experiment_ids(
            bending_setups_df=bending_setups_df,
            group_ids=train_group_ids,
        )
        test_experiment_ids = collect_group_experiment_ids(
            bending_setups_df=bending_setups_df,
            group_ids=test_group_ids,
        )

        if not train_experiment_ids or not test_experiment_ids:
            continue

        experiment_overlap = set(train_experiment_ids).intersection(
            test_experiment_ids
        )
        if experiment_overlap:
            raise ValueError(
                "Experiment leakage detected between train and "
                "test groups. Overlapping experiments: "
                f"{sorted(experiment_overlap)}"
            )

        split_key = (
            tuple(train_group_ids),
            tuple(test_group_ids),
        )
        if split_key in observed_split_keys:
            continue
        observed_split_keys.add(split_key)

        split_name = build_split_name(selected_conditions)

        candidate_rows.append(
            {
                "split_name": split_name,
                "split_type": "single",
                "n_conditions": 1,
                "split_features": [condition["feature"]],
                "split_conditions": [
                    condition["condition_name"]
                ],
                "train_group_ids": train_group_ids,
                "test_group_ids": test_group_ids,
                "train_experiment_ids": train_experiment_ids,
                "test_experiment_ids": test_experiment_ids,
                "train_exp": train_experiment_ids,
                "test_exp": test_experiment_ids,
                "train_group_count": len(train_group_ids),
                "test_group_count": len(test_group_ids),
                "train_experiment_count": len(
                    train_experiment_ids
                ),
                "test_experiment_count": len(
                    test_experiment_ids
                ),
                "test_experiment_percentage": (
                    calculate_test_percentage(
                        train_count=len(train_experiment_ids),
                        test_count=len(test_experiment_ids),
                    )
                ),
                "test_group_percentage": (
                    calculate_test_percentage(
                        train_count=len(train_group_ids),
                        test_count=len(test_group_ids),
                    )
                ),
            }
        )

    if not candidate_rows:
        raise ValueError(
            "No valid leave-one-feature-value-out splits "
            "were generated."
        )

    split_metadata_df = pd.DataFrame(candidate_rows)
    split_metadata_df = (
        split_metadata_df
        .sort_values(
            ["split_features", "split_name"],
            ascending=True,
            key=lambda series: series.astype(str),
        )
        .reset_index(drop=True)
    )

    split_metadata_df.insert(
        0,
        "split_index",
        range(1, len(split_metadata_df) + 1),
    )
    split_metadata_df.insert(
        1,
        "summary_df_index",
        range(len(split_metadata_df)),
    )

    preferred_column_order = [
        "split_index",
        "summary_df_index",
        "split_name",
        "split_type",
        "n_conditions",
        "split_features",
        "split_conditions",
        "train_group_ids",
        "test_group_ids",
        "train_experiment_ids",
        "test_experiment_ids",
        "train_exp",
        "test_exp",
        "train_group_count",
        "test_group_count",
        "train_experiment_count",
        "test_experiment_count",
        "test_experiment_percentage",
        "test_group_percentage",
    ]

    split_metadata_df = split_metadata_df[preferred_column_order]
    validate_generated_splits(split_metadata_df)

    return split_metadata_df


# ============================================================
# Validation and storage
# ============================================================

def validate_generated_splits(
    split_metadata_df: pd.DataFrame,
) -> None:
    """
    Validate that generated metadata contains exactly one row per split.
    """
    required_columns = {
        "split_index",
        "split_name",
        "train_group_ids",
        "test_group_ids",
        "train_experiment_ids",
        "test_experiment_ids",
        "test_experiment_percentage",
        "test_group_percentage",
    }

    validate_required_columns(
        dataframe=split_metadata_df,
        required_columns=required_columns,
        dataframe_name="Generated split metadata",
    )

    if split_metadata_df.empty:
        raise ValueError(
            "Generated split metadata is empty."
        )

    if split_metadata_df[
        "split_index"
    ].duplicated().any():
        duplicated_indices = (
            split_metadata_df.loc[
                split_metadata_df[
                    "split_index"
                ].duplicated(
                    keep=False
                ),
                "split_index",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate split_index values were generated: "
            f"{duplicated_indices}"
        )

    if split_metadata_df[
        "split_name"
    ].duplicated().any():
        duplicated_names = (
            split_metadata_df.loc[
                split_metadata_df[
                    "split_name"
                ].duplicated(
                    keep=False
                ),
                "split_name",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate split names were generated: "
            f"{duplicated_names}"
        )

    for row in split_metadata_df.itertuples(
        index=False
    ):
        train_groups = set(
            int(value)
            for value in row.train_group_ids
        )

        test_groups = set(
            int(value)
            for value in row.test_group_ids
        )

        overlapping_groups = (
            train_groups.intersection(
                test_groups
            )
        )

        if overlapping_groups:
            raise ValueError(
                f"Group leakage in split_index="
                f"{row.split_index}: "
                f"{sorted(overlapping_groups)}"
            )

        train_experiments = set(
            int(value)
            for value
            in row.train_experiment_ids
        )

        test_experiments = set(
            int(value)
            for value
            in row.test_experiment_ids
        )

        overlapping_experiments = (
            train_experiments.intersection(
                test_experiments
            )
        )

        if overlapping_experiments:
            raise ValueError(
                f"Experiment leakage in split_index="
                f"{row.split_index}: "
                f"{sorted(overlapping_experiments)}"
            )


def overwrite_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Replace the previous Parquet file with the new file.

    The new table is first written to a temporary file and then moved
    over the old file. This prevents leaving a partially written output.
    """
    output_path = output_path.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f".{output_path.stem}"
        f"__temporary"
        f"{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    dataframe.to_parquet(
        temporary_path,
        index=False,
    )

    # Path.replace() overwrites the existing destination atomically
    # on the same filesystem.
    temporary_path.replace(
        output_path
    )

    return output_path


# ============================================================
# Main pipeline
# ============================================================

def run_qrf_split_creating(
    *,
    config_path: str | Path,
    output_path: str | Path | None = None,
    split_features: tuple[str, ...] = SPLIT_FEATURES,
    minimum_combination_size: int = (
        MIN_COMBINATION_SIZE
    ),
    maximum_combination_size: int = (
        MAX_COMBINATION_SIZE
    ),
) -> tuple[pd.DataFrame, Path]:
    """
    Generate and overwrite ``various_splits.parquet``.
    """
    resolved_config_path = resolve_path(
        REPOSITORY_ROOT,
        config_path,
    )

    config = load_yaml_config(
        resolved_config_path
    )

    paths_config = config.get(
        "paths",
        {},
    )

    data_config = config.get(
        "data",
        {},
    )

    if "bending_setups" not in paths_config:
        raise KeyError(
            "Config is missing "
            "paths.bending_setups."
        )

    bending_setups_path = resolve_path(
        REPOSITORY_ROOT,
        paths_config[
            "bending_setups"
        ],
    )

    if output_path is None:
        configured_split_path = (
            paths_config.get(
                "split_metadata"
            )
        )

        if configured_split_path:
            resolved_output_path = (
                resolve_path(
                    REPOSITORY_ROOT,
                    configured_split_path,
                )
            )
        else:
            resolved_output_path = (
                REPOSITORY_ROOT
                / DEFAULT_OUTPUT_RELATIVE_PATH
            ).resolve()
    else:
        resolved_output_path = resolve_path(
            REPOSITORY_ROOT,
            output_path,
        )

    excluded_experiments = (
        data_config.get(
            "excluded_experiments",
            [],
        )
    )

    bending_setups_df = load_bending_setups(
        path=bending_setups_path,
        excluded_experiments=(
            excluded_experiments
        ),
    )

    split_metadata_df = (
        generate_split_metadata(
            bending_setups_df=(
                bending_setups_df
            ),
            split_features=tuple(
                split_features
            ),
            minimum_combination_size=(
                minimum_combination_size
            ),
            maximum_combination_size=(
                maximum_combination_size
            ),
        )
    )

    stored_path = overwrite_parquet(
        dataframe=split_metadata_df,
        output_path=resolved_output_path,
    )

    print()
    print("=" * 72)
    print("QRF split metadata created successfully")
    print("=" * 72)
    print(
        f"Bending setups: {bending_setups_path}"
    )
    print(
        f"Output:         {stored_path}"
    )
    print(
        f"Split features: {list(split_features)}"
    )
    print(
        "Split rule:      selected feature value -> test; "
        "all other groups -> train"
    )
    print(
        f"Number of splits: {len(split_metadata_df)}"
    )
    print()

    print("Split counts by type:")
    print(
        split_metadata_df[
            "split_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Stored columns:")
    print(
        split_metadata_df.columns.tolist()
    )

    print()
    print("First generated splits:")
    print(
        split_metadata_df[
            [
                "split_index",
                "split_name",
                "test_experiment_percentage",
                "test_group_percentage",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    return (
        split_metadata_df,
        stored_path,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate leave-one-feature-value-out QRF train/test "
            "splits and overwrite various_splits.parquet."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            QRF_ROOT
            / "configs"
            / DEFAULT_CONFIG_NAME
        ),
        help=(
            "QRF YAML config containing paths.bending_setups "
            "and preferably paths.split_metadata."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path. When omitted, "
            "config['paths']['split_metadata'] is used."
        ),
    )

    parser.add_argument(
        "--max-combination-size",
        type=int,
        default=MAX_COMBINATION_SIZE,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    run_qrf_split_creating(
        config_path=arguments.config,
        output_path=arguments.output,
        split_features=SPLIT_FEATURES,
        minimum_combination_size=(
            MIN_COMBINATION_SIZE
        ),
        maximum_combination_size=(
            arguments.max_combination_size
        ),
    )


if __name__ == "__main__":
    main()
