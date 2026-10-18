from __future__ import annotations

import pandas as pd

from src.dashboard_core import (
    ANGLE_COL,
    EXPERIMENT_COL,
    GROUP_COL,
    MAIN_AXIS,
    experiment_group_mapping,
    mean_pairwise_curve_distance,
    parse_int_list,
)


def test_parse_int_list_from_string() -> None:
    assert parse_int_list("[1, 2, 3]") == [1, 2, 3]


def test_experiment_group_mapping_expands_setup_rows() -> None:
    setup = pd.DataFrame(
        {
            GROUP_COL: [10, 11],
            "Experiment_Number": ["[1, 2]", "[3]"],
        }
    )
    mapped = experiment_group_mapping(setup)
    assert mapped.to_dict("records") == [
        {EXPERIMENT_COL: 1, GROUP_COL: 10},
        {EXPERIMENT_COL: 2, GROUP_COL: 10},
        {EXPERIMENT_COL: 3, GROUP_COL: 11},
    ]


def test_mean_pairwise_curve_distance_requires_two_experiments() -> None:
    frame = pd.DataFrame(
        {
            EXPERIMENT_COL: [1, 1],
            ANGLE_COL: [0, 1],
            MAIN_AXIS: [0.0, 1.0],
        }
    )
    assert mean_pairwise_curve_distance(frame, MAIN_AXIS) is None


def test_mean_pairwise_curve_distance_for_two_curves() -> None:
    frame = pd.DataFrame(
        {
            EXPERIMENT_COL: [1, 1, 2, 2],
            ANGLE_COL: [0, 1, 0, 1],
            MAIN_AXIS: [0.0, 1.0, 3.0, 5.0],
        }
    )
    assert round(mean_pairwise_curve_distance(frame, MAIN_AXIS), 6) == 5.0
