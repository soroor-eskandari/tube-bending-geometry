from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _decode_int_list(value: object) -> list[int]:
    """Decode a list stored as JSON, Python literal or array."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple, set)):
        raise TypeError(
            "Expected split group IDs to be list-like, "
            f"but received {type(value).__name__}."
        )

    return [int(item) for item in value]

def get_split_row(
    split_metadata_df: pd.DataFrame,
    split_index: int,
) -> pd.Series:
    split_indices = pd.to_numeric(
        split_metadata_df["split_index"],
        errors="coerce",
    )

    matches = split_metadata_df[
        split_indices.eq(int(split_index))
    ]

    if matches.empty:
        raise ValueError(
            f"No split metadata found for split_index={split_index}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple rows found for split_index={split_index}."
        )

    return matches.iloc[0]

def make_train_test_split(
    geometry_df: pd.DataFrame,
    split_metadata_df: pd.DataFrame,
    split_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    split_row = get_split_row(
        split_metadata_df=split_metadata_df,
        split_index=split_index,
    )

    train_group_ids = _decode_int_list(
        split_row["train_group_ids"]
    )
    test_group_ids = _decode_int_list(
        split_row["test_group_ids"]
    )

    overlap = set(train_group_ids).intersection(test_group_ids)
    if overlap:
        raise ValueError(
            f"Data leakage in split {split_index}; "
            f"overlapping groups: {sorted(overlap)}"
        )

    train_df = geometry_df[
        geometry_df["Group_ID"].isin(train_group_ids)
    ].copy()

    test_df = geometry_df[
        geometry_df["Group_ID"].isin(test_group_ids)
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"Training data is empty for split_index={split_index}."
        )

    if test_df.empty:
        raise ValueError(
            f"Test data is empty for split_index={split_index}."
        )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_row,
    )

def _normalise_group_ids_for_comparison(
    value: object,
) -> tuple[int, ...]:
    return tuple(sorted(_decode_int_list(value)))

def load_split_metadata(
    path: str | Path,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Split metadata was not found: {path}"
        )

    split_df = pd.read_parquet(path).copy()

    required = {
        "split_index",
        "train_group_ids",
        "test_group_ids",
    }

    missing = required.difference(split_df.columns)

    if missing:
        raise KeyError(
            "Split metadata is missing columns: "
            f"{sorted(missing)}"
        )

    split_df["_train_key"] = (
        split_df["train_group_ids"]
        .apply(_normalise_group_ids_for_comparison)
    )

    split_df["_test_key"] = (
        split_df["test_group_ids"]
        .apply(_normalise_group_ids_for_comparison)
    )

    split_df = (
        split_df.drop_duplicates(
            subset=[
                "split_index",
                "_train_key",
                "_test_key",
            ]
        )
        .drop(
            columns=[
                "_train_key",
                "_test_key",
            ]
        )
        .reset_index(drop=True)
    )

    return split_df