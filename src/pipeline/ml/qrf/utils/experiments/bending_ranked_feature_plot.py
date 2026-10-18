from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PAIRWISE_DISTANCE_COLUMN = "mean_pairwise_distance"
STD_DISTANCE_COLUMN = "std_distance"
MANDREL_POSITION_FEATURE = "mandrel_position"
MANDREL_POSITION_CLUSTER_COUNT = 5
METRIC_AXIS_LIMITS = (0.0, 2.2)


def format_metric_label(metric_name: str) -> str:
    """Create a readable axis label for a metric column."""
    metric_labels = {
        PAIRWISE_DISTANCE_COLUMN: "Mean pairwise distance within group",
        STD_DISTANCE_COLUMN: "Within-group standard deviation",
    }

    return metric_labels.get(
        metric_name,
        metric_name,
    )


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name for tolerant matching."""
    return (
        str(column_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def is_mandrel_position_feature(feature_name: str) -> bool:
    """Return whether a feature should use mandrel-position clustering."""
    return normalize_column_name(feature_name) == MANDREL_POSITION_FEATURE


def cluster_feature_values_for_plotting(
    dataframe: pd.DataFrame,
    feature_name: str,
    value_column: str = "feature_value",
    cluster_count: int = MANDREL_POSITION_CLUSTER_COUNT,
) -> pd.DataFrame:
    """
    Collapse Mandrel position values into a small number of sorted clusters.

    Other features are returned unchanged. The clustered value is the mean
    Mandrel position inside each quantile-like cluster, so plots keep a numeric
    x/y axis while avoiding many visually indistinguishable tiny values.
    """
    if not is_mandrel_position_feature(feature_name):
        return dataframe

    clustered_data = dataframe.copy()
    numeric_values = pd.to_numeric(
        clustered_data[value_column],
        errors="coerce",
    )
    valid_values = numeric_values.dropna()
    unique_count = int(valid_values.nunique())

    if unique_count <= cluster_count:
        return clustered_data

    valid_mask = numeric_values.notna()
    rank_values = numeric_values.loc[valid_mask].rank(method="first")
    cluster_codes = pd.qcut(
        rank_values,
        q=min(cluster_count, unique_count),
        labels=False,
        duplicates="drop",
    )
    cluster_centers = (
        numeric_values
        .loc[valid_mask]
        .groupby(cluster_codes)
        .mean()
    )

    clustered_data.loc[valid_mask, value_column] = (
        cluster_codes
        .map(cluster_centers)
        .to_numpy(dtype=float)
    )

    return clustered_data


def set_shared_y_axis_limits(
    axes,
) -> tuple[float, float]:
    """
    Apply one y-axis scale to a collection of existing Matplotlib axes.

    This is intended for main/secondary comparison plots where each row should
    use the same metric range on both sides.
    """
    axis_list = [
        axis
        for axis in np.asarray(axes, dtype=object).ravel()
        if axis is not None
    ]

    if not axis_list:
        raise ValueError("At least one axis is required.")

    y_limits = np.array(
        [
            axis.get_ylim()
            for axis in axis_list
        ],
        dtype=float,
    )

    finite_limits = y_limits[np.isfinite(y_limits).all(axis=1)]

    if finite_limits.size == 0:
        raise ValueError("No finite y-axis limits are available.")

    y_min = float(finite_limits[:, 0].min())
    y_max = float(finite_limits[:, 1].max())

    if y_min == y_max:
        padding = max(
            abs(y_min) * 0.05,
            1.0,
        )
        y_min -= padding
        y_max += padding

    for axis in axis_list:
        axis.set_ylim(
            y_min,
            y_max,
        )

    return y_min, y_max


def set_metric_axis_limits(
    ax,
    limits: tuple[float, float] = METRIC_AXIS_LIMITS,
) -> None:
    """Use the same metric scale on every ranked-feature plot."""
    ax.set_ylim(limits)


def resolve_column_name(
    dataframe: pd.DataFrame,
    requested_name: str,
    fallback_names: list[str] | None = None,
) -> str:
    """Find a dataframe column using tolerant, case-insensitive matching."""
    normalized_to_original = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    candidates = [requested_name]
    if fallback_names:
        candidates.extend(fallback_names)

    for candidate in candidates:
        normalized_requested = normalize_column_name(candidate)

        if normalized_requested in normalized_to_original:
            return normalized_to_original[normalized_requested]

        partial_matches = [
            original
            for normalized, original in normalized_to_original.items()
            if normalized_requested in normalized
            or normalized in normalized_requested
        ]

        if len(partial_matches) == 1:
            return partial_matches[0]

    raise KeyError(
        f"Could not resolve column {requested_name!r}.\n"
        f"Available columns are:\n{list(dataframe.columns)}"
    )


def convert_group_id(series: pd.Series) -> pd.Series:
    """Convert group identifiers to nullable integers."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def select_ranked_features(
    ranking_data: pd.DataFrame,
    geometry_type: str,
    top_n_features: int | None = None,
    feature_column: str | None = None,
    rank_column: str | None = None,
    geometry_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str | None]:
    """Filter ranking data and return ranked feature rows."""
    feature_column = feature_column or resolve_column_name(
        ranking_data,
        "feature",
    )
    rank_column = rank_column or resolve_column_name(
        ranking_data,
        "rank",
    )

    if geometry_column is None:
        for candidate in ["geometry_type", "geometry", "curve_type"]:
            try:
                geometry_column = resolve_column_name(
                    ranking_data,
                    candidate,
                )
                break
            except KeyError:
                continue

    filtered_ranking_data = ranking_data.copy()

    if geometry_column is not None:
        filtered_ranking_data = filtered_ranking_data.loc[
            filtered_ranking_data[geometry_column]
            .astype(str)
            .str.lower()
            .eq(geometry_type.lower())
        ].copy()

    ranked_features = (
        filtered_ranking_data[
            [
                feature_column,
                rank_column,
            ]
        ]
        .dropna(
            subset=[
                feature_column,
                rank_column,
            ]
        )
        .drop_duplicates(subset=[feature_column])
        .sort_values(rank_column)
        .reset_index(drop=True)
    )

    if top_n_features is not None:
        ranked_features = ranked_features.head(top_n_features)

    return (
        ranked_features,
        filtered_ranking_data,
        feature_column,
        rank_column,
        geometry_column,
    )


def get_available_ranked_features(
    ranked_features: pd.DataFrame,
    bending_data: pd.DataFrame,
    feature_column: str,
) -> tuple[list[str], list[str]]:
    """Split ranked features into available and missing bending-data columns."""
    available_features = []
    missing_features = []

    for feature_name in ranked_features[feature_column]:
        if feature_name in bending_data.columns:
            available_features.append(feature_name)
        else:
            missing_features.append(feature_name)

    return available_features, missing_features


def infer_pairwise_distance_columns(
    bending_data: pd.DataFrame,
    group_column: str,
    exclude_columns: Iterable[str] | None = None,
) -> list[str]:
    """Choose numeric bending columns for within-group pairwise distance."""
    excluded = {group_column}

    if exclude_columns is not None:
        excluded.update(exclude_columns)

    return [
        column
        for column in bending_data.select_dtypes(include="number").columns
        if column not in excluded
    ]


def compute_group_mean_pairwise_distance(
    dataframe: pd.DataFrame,
    group_column: str,
    distance_columns: list[str],
    standardize: bool = True,
    output_column: str = PAIRWISE_DISTANCE_COLUMN,
) -> pd.DataFrame:
    """Compute mean pairwise Euclidean distance between rows inside each group."""
    if not distance_columns:
        raise ValueError("At least one distance column is required.")

    distance_data = dataframe[
        [
            group_column,
            *distance_columns,
        ]
    ].copy()

    for column in distance_columns:
        distance_data[column] = pd.to_numeric(
            distance_data[column],
            errors="coerce",
        )

    if standardize:
        for column in distance_columns:
            column_std = distance_data[column].std()

            if pd.notna(column_std) and column_std != 0:
                distance_data[column] = (
                    distance_data[column] - distance_data[column].mean()
                ) / column_std

    rows = []

    for group_id, group_data in distance_data.groupby(
        group_column,
        dropna=True,
    ):
        values = group_data[distance_columns].dropna().to_numpy()

        if len(values) < 2:
            mean_distance = 0.0
        else:
            deltas = values[:, None, :] - values[None, :, :]
            distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=2))
            upper_triangle_indices = np.triu_indices(
                len(values),
                k=1,
            )
            mean_distance = float(
                distance_matrix[upper_triangle_indices].mean()
            )

        rows.append(
            {
                group_column: group_id,
                output_column: mean_distance,
            }
        )

    return pd.DataFrame(rows)


def compute_group_std_distance(
    dataframe: pd.DataFrame,
    group_column: str,
    distance_columns: list[str],
    standardize: bool = True,
    output_column: str = STD_DISTANCE_COLUMN,
) -> pd.DataFrame:
    """
    Compute one within-group standard-deviation spread value per group.

    The result is the Euclidean norm of the per-column standard deviations.
    With standardize=True, columns are standardized globally before group stds
    are computed, so columns with larger units do not dominate the metric.
    """
    if not distance_columns:
        raise ValueError("At least one distance column is required.")

    distance_data = dataframe[
        [
            group_column,
            *distance_columns,
        ]
    ].copy()

    for column in distance_columns:
        distance_data[column] = pd.to_numeric(
            distance_data[column],
            errors="coerce",
        )

    if standardize:
        for column in distance_columns:
            column_std = distance_data[column].std()

            if pd.notna(column_std) and column_std != 0:
                distance_data[column] = (
                    distance_data[column] - distance_data[column].mean()
                ) / column_std

    rows = []

    for group_id, group_data in distance_data.groupby(
        group_column,
        dropna=True,
    ):
        values = group_data[distance_columns].dropna()

        if len(values) < 2:
            std_distance = 0.0
        else:
            column_stds = values.std(ddof=1).to_numpy(dtype=float)
            std_distance = float(
                np.linalg.norm(column_stds)
            )

        rows.append(
            {
                group_column: group_id,
                output_column: std_distance,
            }
        )

    return pd.DataFrame(rows)


def build_feature_plot_dataframe(
    feature_name: str,
    bending_data: pd.DataFrame,
    group_distance_data: pd.DataFrame,
    bending_group_name: str,
    metric_name: str = PAIRWISE_DISTANCE_COLUMN,
) -> pd.DataFrame:
    """Build plotting data for one ranked bending feature."""
    feature_value_column = resolve_column_name(
        bending_data,
        feature_name,
    )

    plot_data = bending_data[
        [
            bending_group_name,
            feature_value_column,
        ]
    ].copy()

    plot_data = plot_data.rename(
        columns={
            feature_value_column: "feature_value",
        }
    )

    plot_data = plot_data.drop_duplicates(
        subset=[
            bending_group_name,
            "feature_value",
        ]
    )

    plot_data = plot_data.merge(
        group_distance_data[
            [
                bending_group_name,
                metric_name,
            ]
        ],
        on=bending_group_name,
        how="left",
    )

    return plot_data


def _feature_band_half_height(
    feature_values: np.ndarray,
    minimum: float = 0.01,
) -> float:
    if len(feature_values) <= 1:
        return max(
            abs(float(feature_values[0])) * 0.03,
            minimum,
        )

    value_differences = np.diff(np.sort(feature_values))
    positive_differences = value_differences[value_differences > 0]

    if len(positive_differences) == 0:
        return minimum

    return max(
        float(positive_differences.min()) * 0.35,
        minimum,
    )


def plot_ranked_feature_group_spread(
    plot_data: pd.DataFrame,
    feature_name: str,
    feature_rank: int,
    metric_name: str,
    group_column: str,
    geometry_type: str,
    show_values: bool = False,
    group_label_mode: str = "right_margin",
    ax=None,
):
    """
    Plot group spread by bending feature value.

    x-axis: selected group-spread metric
    y-axis: selected bending feature value
    colored lines: group IDs, vertically separated inside each feature interval

    group_label_mode:
        "right_margin" places group labels in one aligned column.
        "line_end" places group labels at the end of each line.
        "legend" places group labels and metric ranges in an outside legend.
        "none" hides group labels.
    """
    required_columns = [
        group_column,
        "feature_value",
        metric_name,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in plot_data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in plot_data: {missing_columns}. "
            f"Available columns: {list(plot_data.columns)}"
        )

    clean_data = plot_data[required_columns].copy()
    clean_data[group_column] = pd.to_numeric(
        clean_data[group_column],
        errors="coerce",
    )
    clean_data["feature_value"] = pd.to_numeric(
        clean_data["feature_value"],
        errors="coerce",
    )
    clean_data[metric_name] = pd.to_numeric(
        clean_data[metric_name],
        errors="coerce",
    )
    clean_data = clean_data.dropna(
        subset=required_columns,
    )
    clean_data = cluster_feature_values_for_plotting(
        dataframe=clean_data,
        feature_name=feature_name,
    )

    if clean_data.empty:
        raise ValueError(
            f"No valid data available for plotting feature {feature_name!r}."
        )

    grouped_data = (
        clean_data
        .groupby(
            [
                "feature_value",
                group_column,
            ],
            as_index=False,
        )
        .agg(metric_value=(metric_name, "mean"))
        .sort_values(
            [
                "feature_value",
                group_column,
            ]
        )
    )

    unique_feature_values = np.sort(
        grouped_data["feature_value"].unique()
    )
    band_half_height = _feature_band_half_height(
        unique_feature_values,
    )

    max_groups_per_feature_value = int(
        grouped_data
        .groupby("feature_value")[group_column]
        .count()
        .max()
    )
    fig_height = max(
        5,
        0.9 * len(unique_feature_values)
        + 0.08 * max_groups_per_feature_value,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, fig_height),
        )
    else:
        fig = ax.figure

    for feature_value in unique_feature_values:
        ax.axhspan(
            feature_value - band_half_height,
            feature_value + band_half_height,
            color="0.82",
            alpha=0.7,
            zorder=0,
        )

    unique_groups = sorted(grouped_data[group_column].unique())
    color_map = {
        group: plt.cm.tab20(index % 20)
        for index, group in enumerate(unique_groups)
    }

    if group_label_mode not in {
        "right_margin",
        "line_end",
        "legend",
        "none",
    }:
        raise ValueError(
            "group_label_mode must be one of: "
            "'right_margin', 'line_end', 'legend', 'none'."
        )

    metric_max = grouped_data["metric_value"].max()
    label_x_offset = max(
        metric_max * 0.008,
        0.01,
    )
    right_label_x = metric_max * 1.04
    legend_handles = []

    for feature_value, value_data in grouped_data.groupby("feature_value"):
        value_data = value_data.sort_values(group_column)

        if len(value_data) == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(
                -band_half_height * 0.62,
                band_half_height * 0.62,
                len(value_data),
            )

        for offset, (_, row) in zip(offsets, value_data.iterrows()):
            group_id = row[group_column]
            distance_value = row["metric_value"]
            y_position = feature_value + offset

            ax.hlines(
                y=y_position,
                xmin=0,
                xmax=distance_value,
                color=color_map[group_id],
                linewidth=3.2,
                alpha=0.95,
                zorder=3,
            )

            if group_label_mode == "legend":
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color_map[group_id],
                        linewidth=3.2,
                        label=(
                            f"G{int(group_id)} | "
                            f"{feature_value:.4g} | "
                            f"0-{distance_value:.3g}"
                        ),
                    )
                )
            elif group_label_mode != "none":
                if group_label_mode == "right_margin":
                    label_x = right_label_x
                    ax.plot(
                        [
                            distance_value,
                            label_x - label_x_offset,
                        ],
                        [
                            y_position,
                            y_position,
                        ],
                        color=color_map[group_id],
                        linewidth=0.7,
                        alpha=0.45,
                        zorder=2,
                    )
                else:
                    label_x = distance_value + label_x_offset

                ax.text(
                    label_x,
                    y_position,
                    f"G{int(group_id)}",
                    color=color_map[group_id],
                    fontsize=7,
                    va="center",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.65,
                        "pad": 0.4,
                    },
                )

            if show_values:
                ax.text(
                    distance_value,
                    y_position,
                    f" {distance_value:.3g}",
                    color=color_map[group_id],
                    fontsize=8,
                    va="center",
                    ha="right",
                )

    x_multiplier = 1.28 if group_label_mode == "right_margin" else 1.18
    if group_label_mode == "legend":
        x_multiplier = 1.06

    x_right = max(
        metric_max * x_multiplier,
        1.0,
    )

    ax.set_xlim(
        left=0,
        right=x_right,
    )
    ax.set_yticks(unique_feature_values)
    ax.set_xlabel(format_metric_label(metric_name))
    ax.set_ylabel(feature_name)
    ax.set_title(
        f"Rank {int(feature_rank)} | {feature_name} | {geometry_type}"
    )
    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.25,
    )
    ax.grid(
        axis="y",
        visible=False,
    )

    if group_label_mode == "legend" and legend_handles:
        legend_column_count = max(
            1,
            int(np.ceil(len(legend_handles) / 28)),
        )
        ax.legend(
            handles=legend_handles,
            title=f"{group_column} | {feature_name} | range",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=6,
            title_fontsize=7,
            frameon=True,
            ncol=legend_column_count,
            handlelength=1.8,
            columnspacing=0.9,
        )

    if ax is None:
        fig.tight_layout()

    return fig, ax


def plot_ranked_feature_group_bars(*args, **kwargs):
    """Backward-compatible name for the new group-spread line plot."""
    return plot_ranked_feature_group_spread(*args, **kwargs)


def generate_ranked_feature_group_spread_plots(
    ranked_features: pd.DataFrame,
    bending_data: pd.DataFrame,
    group_distance_data: pd.DataFrame,
    feature_column: str,
    rank_column: str,
    group_column: str,
    geometry_type: str,
    metric_name: str = PAIRWISE_DISTANCE_COLUMN,
    save_figures: bool = False,
    figure_output_dir: Path | None = None,
    show_values: bool = False,
) -> dict[str, plt.Figure]:
    """Generate one group-spread figure for each ranked bending feature."""
    generated_figures = {}

    for _, row in ranked_features.iterrows():
        feature_name = row[feature_column]
        feature_rank = row[rank_column]

        if feature_name not in bending_data.columns:
            print(
                f"Skipping {feature_name!r}: "
                "column not found in bending setup data."
            )
            continue

        feature_plot_data = build_feature_plot_dataframe(
            feature_name=feature_name,
            bending_data=bending_data,
            group_distance_data=group_distance_data,
            bending_group_name=group_column,
            metric_name=metric_name,
        )

        if feature_plot_data.empty:
            print(
                f"Skipping {feature_name!r}: "
                "no valid rows after merging."
            )
            continue

        fig, _ = plot_ranked_feature_group_spread(
            plot_data=feature_plot_data,
            feature_name=feature_name,
            feature_rank=feature_rank,
            metric_name=metric_name,
            group_column=group_column,
            geometry_type=geometry_type,
            show_values=show_values,
        )

        generated_figures[feature_name] = fig

        if save_figures:
            if figure_output_dir is None:
                raise ValueError(
                    "figure_output_dir is required when save_figures=True."
                )

            figure_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            safe_feature_name = (
                str(feature_name)
                .lower()
                .replace(" ", "_")
                .replace("/", "_")
            )
            output_path = (
                figure_output_dir
                / (
                    f"rank_{int(feature_rank):02d}_"
                    f"{safe_feature_name}_"
                    f"{normalize_column_name(metric_name)}.png"
                )
            )

            fig.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight",
            )
            print("Saved:", output_path)

        plt.show()

    return generated_figures


def _prepare_metric_feature_data(
    plot_data: pd.DataFrame,
    metric_name: str,
    feature_name: str | None = None,
) -> pd.DataFrame:
    required_columns = [
        "feature_value",
        metric_name,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in plot_data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in plot_data: {missing_columns}. "
            f"Available columns: {list(plot_data.columns)}"
        )

    clean_data = plot_data[required_columns].copy()
    clean_data["feature_value"] = pd.to_numeric(
        clean_data["feature_value"],
        errors="coerce",
    )
    clean_data[metric_name] = pd.to_numeric(
        clean_data[metric_name],
        errors="coerce",
    )
    clean_data = clean_data.dropna(
        subset=required_columns,
    )
    if feature_name is not None:
        clean_data = cluster_feature_values_for_plotting(
            dataframe=clean_data,
            feature_name=feature_name,
        )

    if clean_data.empty:
        raise ValueError("No valid feature/metric rows are available.")

    return clean_data.sort_values("feature_value")


def plot_metric_distribution_by_feature_value(
    plot_data: pd.DataFrame,
    feature_name: str,
    metric_name: str,
    geometry_type: str,
    ax=None,
):
    """Plot metric distribution for each bending feature value."""
    clean_data = _prepare_metric_feature_data(
        plot_data=plot_data,
        metric_name=metric_name,
        feature_name=feature_name,
    )

    unique_feature_values = np.sort(
        clean_data["feature_value"].unique()
    )

    grouped_values = [
        clean_data.loc[
            clean_data["feature_value"].eq(feature_value),
            metric_name,
        ].to_numpy()
        for feature_value in unique_feature_values
    ]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, max(4, 0.45 * len(unique_feature_values))),
        )
    else:
        fig = ax.figure

    positions = np.arange(len(unique_feature_values))

    ax.boxplot(
        grouped_values,
        positions=positions,
        vert=True,
        widths=0.55,
        whis=(0, 100),
        patch_artist=True,
        boxprops={
            "facecolor": "0.82",
            "edgecolor": "0.35",
        },
        medianprops={
            "color": "black",
            "linewidth": 1.4,
        },
        whiskerprops={
            "color": "0.4",
        },
        capprops={
            "color": "0.4",
        },
        flierprops={
            "marker": "",
        },
    )

    rng = np.random.default_rng(42)

    for position, values in zip(positions, grouped_values):
        jitter = rng.uniform(
            low=-0.16,
            high=0.16,
            size=len(values),
        )
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=24,
            alpha=0.72,
            color="#2f6fbb",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{value:.4g}" for value in unique_feature_values],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Distribution | {feature_name} | {geometry_type}"
    )
    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.25,
    )
    set_metric_axis_limits(ax)

    if ax is None:
        fig.tight_layout()

    return fig, ax


def plot_metric_mean_median_trend(
    plot_data: pd.DataFrame,
    feature_name: str,
    metric_name: str,
    geometry_type: str,
    ax=None,
):
    """Plot mean and median metric trends across bending feature values."""
    clean_data = _prepare_metric_feature_data(
        plot_data=plot_data,
        metric_name=metric_name,
        feature_name=feature_name,
    )

    trend_data = (
        clean_data
        .groupby("feature_value", as_index=False)
        .agg(
            mean_metric=(metric_name, "mean"),
            median_metric=(metric_name, "median"),
            count=(metric_name, "count"),
        )
        .sort_values("feature_value")
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 5),
        )
    else:
        fig = ax.figure

    ax.plot(
        trend_data["feature_value"],
        trend_data["mean_metric"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="Mean",
    )
    ax.plot(
        trend_data["feature_value"],
        trend_data["median_metric"],
        marker="s",
        linewidth=2.0,
        color="#d62728",
        label="Median",
    )

    ax.set_xlabel(feature_name)
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Mean/median trend | {feature_name} | {geometry_type}"
    )
    ax.grid(
        axis="both",
        linestyle=":",
        alpha=0.25,
    )
    ax.legend()
    set_metric_axis_limits(ax)

    if ax is None:
        fig.tight_layout()

    return fig, ax


def _default_control_columns(
    bending_data: pd.DataFrame,
    group_column: str,
    feature_name: str,
) -> list[str]:
    feature_column = resolve_column_name(
        bending_data,
        feature_name,
    )

    excluded = {
        group_column,
        feature_column,
    }

    return [
        column
        for column in bending_data.select_dtypes(include="number").columns
        if column not in excluded
    ]


def build_adjusted_effect_dataframe(
    plot_data: pd.DataFrame,
    bending_data: pd.DataFrame,
    feature_name: str,
    metric_name: str,
    group_column: str,
    control_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate feature effect after controlling for other bending parameters."""
    if group_column not in plot_data.columns:
        raise KeyError(
            f"{group_column!r} is missing from plot_data."
        )

    feature_column = resolve_column_name(
        bending_data,
        feature_name,
    )

    if control_columns is None:
        control_columns = _default_control_columns(
            bending_data=bending_data,
            group_column=group_column,
            feature_name=feature_name,
        )

    model_columns = [
        group_column,
        feature_column,
        *control_columns,
    ]

    model_data = plot_data[
        [
            group_column,
            metric_name,
        ]
    ].merge(
        bending_data[model_columns],
        on=group_column,
        how="left",
    )

    numeric_columns = [
        metric_name,
        feature_column,
        *control_columns,
    ]

    for column in numeric_columns:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    model_data = model_data.dropna(
        subset=numeric_columns,
    )

    if model_data.empty:
        raise ValueError("No valid rows are available for adjusted effect.")

    x_columns = [
        feature_column,
        *control_columns,
    ]
    x_matrix = model_data[x_columns].to_numpy(dtype=float)
    y_values = model_data[metric_name].to_numpy(dtype=float)

    x_mean = x_matrix.mean(axis=0)
    x_std = x_matrix.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_scaled = (x_matrix - x_mean) / x_std
    design_matrix = np.column_stack(
        [
            np.ones(len(x_scaled)),
            x_scaled,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        y_values,
        rcond=None,
    )

    feature_axis_data = cluster_feature_values_for_plotting(
        dataframe=model_data.rename(
            columns={
                feature_column: "feature_value",
            }
        ),
        feature_name=feature_name,
    )
    feature_values = np.sort(
        feature_axis_data["feature_value"].unique()
    )
    control_medians = (
        model_data[control_columns].median().to_numpy(dtype=float)
        if control_columns
        else np.array([])
    )

    prediction_rows = []

    for feature_value in feature_values:
        raw_row = np.array(
            [
                feature_value,
                *control_medians,
            ],
            dtype=float,
        )
        scaled_row = (raw_row - x_mean) / x_std
        prediction = float(
            np.dot(
                np.r_[
                    1.0,
                    scaled_row,
                ],
                coefficients,
            )
        )
        prediction_rows.append(
            {
                "feature_value": feature_value,
                "adjusted_metric": prediction,
            }
        )

    return pd.DataFrame(prediction_rows)


def plot_adjusted_effect_by_feature_value(
    plot_data: pd.DataFrame,
    bending_data: pd.DataFrame,
    feature_name: str,
    metric_name: str,
    group_column: str,
    geometry_type: str,
    control_columns: list[str] | None = None,
    ax=None,
):
    """Plot adjusted metric effect of one feature, controlling other parameters."""
    adjusted_data = build_adjusted_effect_dataframe(
        plot_data=plot_data,
        bending_data=bending_data,
        feature_name=feature_name,
        metric_name=metric_name,
        group_column=group_column,
        control_columns=control_columns,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 5),
        )
    else:
        fig = ax.figure

    ax.plot(
        adjusted_data["feature_value"],
        adjusted_data["adjusted_metric"],
        marker="o",
        linewidth=2.2,
        color="#2ca02c",
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"Adjusted {metric_name}")
    ax.set_title(
        f"Adjusted effect | {feature_name} | {geometry_type}"
    )
    ax.grid(
        axis="both",
        linestyle=":",
        alpha=0.25,
    )
    set_metric_axis_limits(ax)

    if ax is None:
        fig.tight_layout()

    return fig, ax
