from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


def _shorten_text(
    value: object,
    max_length: int = 75,
) -> str:
    text = str(value)

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def _validate_metric_columns(
    metrics_df: pd.DataFrame,
    metric: str,
) -> None:
    required_columns = {
        "split_index",
        "split_name",
        "split_rank",
        "axis",
        "n_estimators",
        metric,
    }

    missing = required_columns.difference(
        metrics_df.columns
    )

    if missing:
        raise KeyError(
            "Metrics DataFrame is missing required columns: "
            f"{sorted(missing)}"
        )


def _get_ranked_split_catalog(
    metrics_df: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    """
    Return one row per ranked split.
    """
    catalog_df = (
        metrics_df[
            metrics_df["axis"].eq(axis)
        ][
            [
                "split_index",
                "split_rank",
                "split_name",
            ]
        ]
        .drop_duplicates()
        .dropna(subset=["split_rank"])
        .copy()
    )

    catalog_df["split_rank"] = (
        pd.to_numeric(
            catalog_df["split_rank"],
            errors="raise",
        ).astype(int)
    )

    return catalog_df.sort_values(
        "split_rank",
        ascending=True,
    )


def _select_top_and_worst_splits(
    metrics_df: pd.DataFrame,
    axis: str,
    top_n: int = 3,
    worst_n: int = 3,
) -> pd.DataFrame:
    """
    Select the first top_n and final worst_n ranked splits.
    """
    catalog_df = _get_ranked_split_catalog(
        metrics_df=metrics_df,
        axis=axis,
    )

    if catalog_df.empty:
        raise ValueError(
            f"No ranked splits are available for axis={axis!r}."
        )

    top_df = catalog_df.head(top_n).copy()
    top_df["rank_group"] = "Top"

    worst_df = catalog_df.tail(worst_n).copy()
    worst_df["rank_group"] = "Worst"

    selected_df = (
        pd.concat(
            [top_df, worst_df],
            ignore_index=True,
        )
        .drop_duplicates(subset=["split_index"])
        .sort_values(
            ["rank_group", "split_rank"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    return selected_df


def plot_metric_by_tree_count(
    metrics_df: pd.DataFrame,
    metric: str,
    axis: str,
    minimum_coverage: float | None = None,
    top_n: int = 3,
    worst_n: int = 3,
    max_label_length: int = 75,
    figsize: tuple[int, int] = (14, 7),
) -> pd.DataFrame:
    """
    Plot tree-count convergence for the top and worst ranked splits.

    Returns the selected split catalog so it can optionally be
    inspected in the notebook.
    """
    _validate_metric_columns(
        metrics_df=metrics_df,
        metric=metric,
    )

    selected_splits_df = (
        _select_top_and_worst_splits(
            metrics_df=metrics_df,
            axis=axis,
            top_n=top_n,
            worst_n=worst_n,
        )
    )

    selected_indices = set(
        selected_splits_df["split_index"]
    )

    plot_df = metrics_df[
        metrics_df["axis"].eq(axis)
        & metrics_df["split_index"].isin(
            selected_indices
        )
    ].copy()

    plot_df = plot_df.merge(
        selected_splits_df[
            [
                "split_index",
                "rank_group",
            ]
        ],
        on="split_index",
        how="left",
        validate="many_to_one",
    )

    fig, ax = plt.subplots(figsize=figsize)

    group_columns = [
        "split_index",
        "split_rank",
        "split_name",
        "rank_group",
    ]

    grouped = plot_df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for (
        split_index,
        split_rank,
        split_name,
        rank_group,
    ), split_df in grouped:
        split_df = split_df.sort_values(
            "n_estimators"
        )

        short_name = _shorten_text(
            split_name,
            max_length=max_label_length,
        )

        label = (
            f"{rank_group} rank {int(split_rank)} — "
            f"{short_name}"
        )

        linestyle = (
            "-"
            if rank_group == "Top"
            else "--"
        )

        ax.plot(
            split_df["n_estimators"],
            split_df[metric],
            marker="o",
            linestyle=linestyle,
            label=label,
        )

    readable_metric = metric.replace(
        "_",
        " ",
    ).title()

    ax.set_title(
        f"{axis.capitalize()} axis: "
        f"{readable_metric} by tree count"
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel(readable_metric)
    ax.grid(alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )

    fig.tight_layout()
    plt.show()

    return selected_splits_df


def plot_prediction_interval(
    predictions_df: pd.DataFrame,
    run_id: str,
) -> None:
    plot_df = predictions_df[
        predictions_df["run_id"].eq(run_id)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            f"No predictions found for run_id={run_id!r}."
        )

    required_columns = {
        "Angle[degree]ORDistance[mm]",
        "y_true",
        "y_lower",
        "y_median",
        "y_upper",
    }

    missing = required_columns.difference(
        plot_df.columns
    )

    if missing:
        raise KeyError(
            "Predictions DataFrame is missing columns: "
            f"{sorted(missing)}"
        )

    plot_df = plot_df.sort_values(
        "Angle[degree]ORDistance[mm]"
    )

    x = plot_df["Angle[degree]ORDistance[mm]"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        x,
        plot_df["y_lower"],
        plot_df["y_upper"],
        alpha=0.25,
        label="Prediction interval",
    )

    ax.plot(
        x,
        plot_df["y_true"],
        label="Observed",
    )

    ax.plot(
        x,
        plot_df["y_median"],
        label="Median prediction",
    )

    ax.set_title(run_id)
    ax.set_xlabel("Angle or distance")
    ax.set_ylabel("Axis value [mm]")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()


def _shorten_text(
    value: object,
    max_length: int = 58,
) -> str:
    text = str(value)

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def _load_qrf_ranking_catalog(
    split_metadata: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """
    Load one unique ranking row per split.

    The various_splits.parquet file contains many rows per split,
    so ranking information must be deduplicated by split_index.
    """
    if isinstance(split_metadata, pd.DataFrame):
        split_df = split_metadata.copy()
    else:
        split_path = Path(split_metadata)

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split metadata was not found: {split_path}"
            )

        split_df = pd.read_parquet(split_path)

    required_columns = {
        "split_index",
        "split_name",
        "qrf_rank_main",
        "qrf_score_main",
        "qrf_coverage_main",
        "qrf_coverage_error_main",
        "qrf_rmse_main",
        "qrf_mae_main",
        "qrf_mean_interval_width_main",
        "qrf_rank_secondary",
        "qrf_score_secondary",
        "qrf_coverage_secondary",
        "qrf_coverage_error_secondary",
        "qrf_rmse_secondary",
        "qrf_mae_secondary",
        "qrf_mean_interval_width_secondary",
    }

    missing = required_columns.difference(split_df.columns)

    if missing:
        raise KeyError(
            "Split metadata does not contain the QRF ranking "
            f"columns: {sorted(missing)}. "
            "Run run_qrf_split_ranking.py first."
        )

    catalog_df = (
        split_df[
            sorted(required_columns)
        ]
        .drop_duplicates(subset=["split_index"])
        .sort_values("split_index")
        .reset_index(drop=True)
    )

    return catalog_df


def _build_axis_ranking_table(
    ranking_catalog_df: pd.DataFrame,
    selected_split_indices: set[int],
    axis: str,
) -> pd.DataFrame:
    """
    Return ranking metrics for one target axis.
    """
    if axis not in {"main", "secondary"}:
        raise ValueError(
            "axis must be 'main' or 'secondary'."
        )

    axis_df = ranking_catalog_df[
        ranking_catalog_df["split_index"].isin(
            selected_split_indices
        )
    ].copy()

    axis_df = axis_df.rename(
        columns={
            f"qrf_rank_{axis}": "qrf_rank",
            f"qrf_score_{axis}": "qrf_score",
            f"qrf_coverage_{axis}": "coverage_percent",
            f"qrf_coverage_error_{axis}": "coverage_error",
            f"qrf_rmse_{axis}": "rmse",
            f"qrf_mae_{axis}": "mae",
            f"qrf_mean_interval_width_{axis}": (
                "mean_interval_width"
            ),
        }
    )

    axis_df["axis"] = axis.capitalize()

    return axis_df[
        [
            "axis",
            "split_index",
            "split_name",
            "qrf_rank",
            "qrf_score",
            "coverage_percent",
            "coverage_error",
            "rmse",
            "mae",
            "mean_interval_width",
        ]
    ].sort_values("qrf_rank")


def plot_main_secondary_convergence_with_ranking(
    metrics_df: pd.DataFrame,
    split_metadata: pd.DataFrame | str | Path,
    metric: str = "coverage_percent",
    top_n: int = 3,
    worst_n: int = 3,
    max_label_length: int = 62,
    figsize: tuple[int, int] = (18, 12),
) -> pd.DataFrame:
    """
    Plot main and secondary convergence in one figure and show
    the QRF-ranking metrics underneath.

    Ranking is read from the updated various_splits.parquet file.
    Main and secondary rankings remain separate.

    Returns
    -------
    pd.DataFrame
        Table containing the ranking explanation displayed below
        the plots.
    """
    required_metrics_columns = {
        "split_index",
        "axis",
        "n_estimators",
        metric,
    }

    missing = required_metrics_columns.difference(
        metrics_df.columns
    )

    if missing:
        raise KeyError(
            "Metrics DataFrame is missing columns: "
            f"{sorted(missing)}"
        )

    ranking_catalog_df = _load_qrf_ranking_catalog(
        split_metadata
    )

    available_indices = set(
        pd.to_numeric(
            metrics_df["split_index"],
            errors="raise",
        ).astype(int)
    )

    main_ranking_df = _build_axis_ranking_table(
        ranking_catalog_df=ranking_catalog_df,
        selected_split_indices=available_indices,
        axis="main",
    )

    secondary_ranking_df = _build_axis_ranking_table(
        ranking_catalog_df=ranking_catalog_df,
        selected_split_indices=available_indices,
        axis="secondary",
    )

    def select_extremes(
        axis_ranking_df: pd.DataFrame,
    ) -> pd.DataFrame:
        top_df = axis_ranking_df.head(top_n).copy()
        top_df["rank_group"] = "Top"

        worst_df = axis_ranking_df.tail(worst_n).copy()
        worst_df["rank_group"] = "Worst"

        return (
            pd.concat(
                [top_df, worst_df],
                ignore_index=True,
            )
            .drop_duplicates(subset=["split_index"])
            .sort_values("qrf_rank")
            .reset_index(drop=True)
        )

    selected_main_df = select_extremes(
        main_ranking_df
    )
    selected_secondary_df = select_extremes(
        secondary_ranking_df
    )

    explanation_df = pd.concat(
        [
            selected_main_df,
            selected_secondary_df,
        ],
        ignore_index=True,
    )

    figure = plt.figure(
        figsize=(20, 16)
    )

    grid = figure.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[
            2.8,   # plots
            0.95,  # legends
            2.0,   # table
        ],
        hspace=0.22,
        wspace=0.28,
    )

    main_ax = figure.add_subplot(
        grid[0, 0]
    )

    secondary_ax = figure.add_subplot(
        grid[0, 1]
    )

    main_legend_ax = figure.add_subplot(
        grid[1, 0]
    )

    secondary_legend_ax = figure.add_subplot(
        grid[1, 1]
    )

    table_ax = figure.add_subplot(
        grid[2, :]
    )

    def draw_axis(
        ax: plt.Axes,
        axis: str,
        selected_df: pd.DataFrame,
    ) -> None:
        selected_indices = set(
            selected_df["split_index"].astype(int)
        )

        plot_df = metrics_df[
            metrics_df["axis"].eq(axis)
            & metrics_df["split_index"].isin(
                selected_indices
            )
        ].copy()

        plot_df = plot_df.merge(
            selected_df[
                [
                    "split_index",
                    "split_name",
                    "qrf_rank",
                    "rank_group",
                ]
            ],
            on="split_index",
            how="left",
            validate="many_to_one",
            suffixes=("", "_ranking"),
        )

        split_name_column = (
            "split_name_ranking"
            if "split_name_ranking" in plot_df.columns
            else "split_name"
        )

        grouped = plot_df.groupby(
            [
                "split_index",
                split_name_column,
                "qrf_rank",
                "rank_group",
            ],
            dropna=False,
            sort=False,
        )

        for (
            split_index,
            split_name,
            qrf_rank,
            rank_group,
        ), split_df in grouped:
            split_df = split_df.sort_values(
                "n_estimators"
            )

            label = (
                f"{rank_group} rank {int(qrf_rank)} — "
                f"{split_name}"
            )

            linestyle = (
                "-"
                if rank_group == "Top"
                else "--"
            )

            ax.plot(
                split_df["n_estimators"],
                split_df[metric],
                marker="o",
                linestyle=linestyle,
                label=label,
            )

        readable_metric = metric.replace(
            "_",
            " ",
        ).title()

        ax.set_title(
            f"{axis.capitalize()} axis — {readable_metric}"
        )
        ax.set_xlabel("Number of trees")
        ax.set_ylabel(readable_metric)
        ax.grid(alpha=0.3)

    draw_axis(
        ax=main_ax,
        axis="main",
        selected_df=selected_main_df,
    )

    draw_axis(
        ax=secondary_ax,
        axis="secondary",
        selected_df=selected_secondary_df,
    )

    main_handles, main_labels = (
        main_ax.get_legend_handles_labels()
    )

    secondary_handles, secondary_labels = (
        secondary_ax.get_legend_handles_labels()
    )

    main_legend_ax.axis("off")
    secondary_legend_ax.axis("off")

    main_legend_ax.legend(
        handles=main_handles,
        labels=main_labels,
        title="Main-axis split ranking",
        loc="center",
        ncol=1,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.45,
        handlelength=2.5,
    )

    secondary_legend_ax.legend(
        handles=secondary_handles,
        labels=secondary_labels,
        title="Secondary-axis split ranking",
        loc="center",
        ncol=1,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.45,
        handlelength=2.5,
    )

    table_ax.axis("off")

    table_display_df = explanation_df[
        [
            "axis",
            "rank_group",
            "qrf_rank",
            "split_name",
            "qrf_score",
            "coverage_percent",
            "coverage_error",
            "rmse",
            "mae",
            "mean_interval_width",
        ]
    ].copy()

    table_display_df["split_name"] = (
        table_display_df["split_name"]
        .apply(
            lambda value: _shorten_text(
                value,
                max_length=48,
            )
        )
    )

    numeric_columns = [
        "qrf_score",
        "coverage_percent",
        "coverage_error",
        "rmse",
        "mae",
        "mean_interval_width",
    ]

    for column in numeric_columns:
        table_display_df[column] = (
            pd.to_numeric(
                table_display_df[column],
                errors="coerce",
            ).map(
                lambda value: (
                    f"{value:.4f}"
                    if pd.notna(value)
                    else ""
                )
            )
        )

    table_display_df = table_display_df.rename(
        columns={
            "axis": "Axis",
            "rank_group": "Group",
            "qrf_rank": "Rank",
            "split_name": "Split",
            "qrf_score": "QRF score",
            "coverage_percent": "Coverage %",
            "coverage_error": "|Coverage − 90|",
            "rmse": "RMSE",
            "mae": "MAE",
            "mean_interval_width": "Interval width",
        }
    )

    table = table_ax.table(
        cellText=table_display_df.values,
        colLabels=table_display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)

    table_ax.set_title(
        (
            "Why a split can have higher coverage but a worse rank: "
            "the QRF score also includes RMSE, MAE and interval width. "
            "Lower QRF score is better."
        ),
        pad=16,
    )

    figure.suptitle(
        (
            "QRF tree convergence and axis-specific "
            "split-ranking explanation"
        ),
        fontsize=15,
    )

    figure.subplots_adjust(
        top=0.93,
        bottom=0.04,
        left=0.05,
        right=0.98,
    )
    plt.show()

    return explanation_df


def _infer_split_type(split_name: str) -> str:
    """
    Infer whether a split is single, pair or a larger combination.
    """
    name = str(split_name)
    lower_name = name.lower()

    if "single" in lower_name:
        return "Single"

    if "pair" in lower_name:
        return "Pair"

    condition_count = name.count("__AND__") + 1

    if condition_count <= 1:
        return "Single"

    if condition_count == 2:
        return "Pair"

    return "Combination"


def _wrap_split_name(
    split_name: str,
    width: int = 58,
) -> str:
    """
    Wrap a split name without truncating it.
    """
    readable_name = str(split_name)

    readable_name = readable_name.replace(
        "split = ",
        "",
    )

    readable_name = readable_name.replace(
        "__AND__",
        " AND ",
    )

    readable_name = readable_name.replace(
        "__",
        " ",
    )

    return textwrap.fill(
        readable_name,
        width=width,
    )


def _load_unique_qrf_splits(
    split_metadata: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """
    Load one row per unique split from various_splits.parquet.
    """
    if isinstance(split_metadata, pd.DataFrame):
        split_df = split_metadata.copy()
    else:
        split_path = Path(split_metadata)

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split metadata was not found: {split_path}"
            )

        split_df = pd.read_parquet(split_path)

    required_columns = {
        "split_index",
        "split_name",
        "qrf_rank_main",
        "qrf_score_main",
        "qrf_coverage_main",
        "qrf_coverage_error_main",
        "qrf_rmse_main",
        "qrf_mae_main",
        "qrf_mean_interval_width_main",
        "qrf_rank_secondary",
        "qrf_score_secondary",
        "qrf_coverage_secondary",
        "qrf_coverage_error_secondary",
        "qrf_rmse_secondary",
        "qrf_mae_secondary",
        "qrf_mean_interval_width_secondary",
    }

    missing = required_columns.difference(
        split_df.columns
    )

    if missing:
        raise KeyError(
            "Split metadata is missing QRF ranking columns: "
            f"{sorted(missing)}. "
            "Run run_qrf_split_ranking.py first."
        )

    unique_df = (
        split_df[
            sorted(required_columns)
        ]
        .drop_duplicates(subset=["split_index"])
        .reset_index(drop=True)
    )

    unique_df["split_type"] = (
        unique_df["split_name"]
        .apply(_infer_split_type)
    )

    return unique_df


def _prepare_axis_split_effects(
    split_df: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    """
    Convert axis-specific QRF columns to a common schema.
    """
    if axis not in {"main", "secondary"}:
        raise ValueError(
            "axis must be either 'main' or 'secondary'."
        )

    axis_df = split_df[
        [
            "split_index",
            "split_name",
            "split_type",
            f"qrf_rank_{axis}",
            f"qrf_score_{axis}",
            f"qrf_coverage_{axis}",
            f"qrf_coverage_error_{axis}",
            f"qrf_rmse_{axis}",
            f"qrf_mae_{axis}",
            f"qrf_mean_interval_width_{axis}",
        ]
    ].copy()

    axis_df = axis_df.rename(
        columns={
            f"qrf_rank_{axis}": "qrf_rank",
            f"qrf_score_{axis}": "qrf_score",
            f"qrf_coverage_{axis}": "coverage_percent",
            f"qrf_coverage_error_{axis}": "coverage_error",
            f"qrf_rmse_{axis}": "rmse",
            f"qrf_mae_{axis}": "mae",
            f"qrf_mean_interval_width_{axis}": (
                "mean_interval_width"
            ),
        }
    )

    numeric_columns = [
        "qrf_rank",
        "qrf_score",
        "coverage_percent",
        "coverage_error",
        "rmse",
        "mae",
        "mean_interval_width",
    ]

    for column in numeric_columns:
        axis_df[column] = pd.to_numeric(
            axis_df[column],
            errors="raise",
        )

    axis_df = (
        axis_df
        .sort_values(
            "qrf_rank",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    axis_df["display_name"] = (
        axis_df["split_name"]
        .apply(_wrap_split_name)
    )

    return axis_df


def _format_full_split_name(
    split_name: object,
    wrap_width: int = 68,
) -> str:
    """
    Convert a stored split name into a readable multi-line label.

    The complete name is preserved; it is wrapped, not truncated.
    """
    text = str(split_name).strip()

    if text.startswith("split = "):
        text = text.removeprefix("split = ")

    text = text.replace(
        "__AND__",
        " AND ",
    )

    # Keep underscores inside real variable names, but clean the
    # repeated separators used by the generated split names.
    text = text.replace(
        "pair__",
        "Pair: ",
    )

    text = text.replace(
        "single__",
        "Single: ",
    )

    return textwrap.fill(
        text,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _load_unique_axis_rankings(
    split_metadata: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """
    Load one unique row per split from various_splits.parquet.

    The source parquet contains many prediction rows per split.
    This function keeps one ranking record per split_index.
    """
    if isinstance(split_metadata, pd.DataFrame):
        split_df = split_metadata.copy()
    else:
        split_path = Path(split_metadata)

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split metadata was not found: {split_path}"
            )

        split_df = pd.read_parquet(
            split_path
        )

    required_columns = {
        "split_index",
        "split_name",
        "qrf_rank_main",
        "qrf_score_main",
        "qrf_coverage_main",
        "qrf_coverage_error_main",
        "qrf_rmse_main",
        "qrf_mae_main",
        "qrf_mean_interval_width_main",
        "qrf_rank_secondary",
        "qrf_score_secondary",
        "qrf_coverage_secondary",
        "qrf_coverage_error_secondary",
        "qrf_rmse_secondary",
        "qrf_mae_secondary",
        "qrf_mean_interval_width_secondary",
    }

    missing = required_columns.difference(
        split_df.columns
    )

    if missing:
        raise KeyError(
            "Split metadata does not contain all QRF ranking "
            f"columns: {sorted(missing)}. "
            "Run run_qrf_split_ranking.py first."
        )

    unique_df = (
        split_df[
            list(required_columns)
        ]
        .drop_duplicates(
            subset=["split_index"]
        )
        .reset_index(drop=True)
    )

    return unique_df


def _prepare_split_metric_row(
    ranking_df: pd.DataFrame,
    axis: str,
    wrap_width: int,
) -> pd.DataFrame:
    """
    Prepare one axis-specific table for plotting.
    """
    if axis not in {
        "main",
        "secondary",
    }:
        raise ValueError(
            "axis must be 'main' or 'secondary'."
        )

    axis_df = ranking_df[
        [
            "split_index",
            "split_name",
            f"qrf_rank_{axis}",
            f"qrf_score_{axis}",
            f"qrf_coverage_{axis}",
            f"qrf_coverage_error_{axis}",
            f"qrf_rmse_{axis}",
            f"qrf_mae_{axis}",
            f"qrf_mean_interval_width_{axis}",
        ]
    ].copy()

    axis_df = axis_df.rename(
        columns={
            f"qrf_rank_{axis}": "qrf_rank",
            f"qrf_score_{axis}": "qrf_score",
            f"qrf_coverage_{axis}": (
                "coverage_percent"
            ),
            f"qrf_coverage_error_{axis}": (
                "coverage_error"
            ),
            f"qrf_rmse_{axis}": "rmse",
            f"qrf_mae_{axis}": "mae",
            f"qrf_mean_interval_width_{axis}": (
                "mean_interval_width"
            ),
        }
    )

    numeric_columns = [
        "qrf_rank",
        "qrf_score",
        "coverage_percent",
        "coverage_error",
        "rmse",
        "mae",
        "mean_interval_width",
    ]

    for column in numeric_columns:
        axis_df[column] = pd.to_numeric(
            axis_df[column],
            errors="raise",
        )

    # Rank 1 appears at the top of the horizontal plots.
    axis_df = (
        axis_df
        .sort_values(
            "qrf_rank",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    axis_df["display_name"] = axis_df.apply(
        lambda row: (
            f"Rank {int(row['qrf_rank'])} | "
            f"Score {row['qrf_score']:.3f}\n"
            f"{_format_full_split_name(
                row['split_name'],
                wrap_width=wrap_width,
            )}"
        ),
        axis=1,
    )

    return axis_df


def plot_split_metric_comparison(
    split_metadata: pd.DataFrame | str | Path,
    *,
    wrap_width: int = 68,
    figsize: tuple[int, int] = (32, 24),
    bar_height: float = 0.72,
) -> pd.DataFrame:
    """
    Compare all split definitions using the four QRF-ranking metrics.

    The output is one figure:

        Main axis:
            coverage error | RMSE | MAE | interval width

        Secondary axis:
            coverage error | RMSE | MAE | interval width

    Main and secondary rankings are independent. Split names are shown
    in full using wrapped multi-line labels.

    Lower values are better for all four displayed metrics.

    Returns
    -------
    pd.DataFrame
        Long-form table containing the plotted main and secondary
        ranking metrics.
    """
    ranking_df = _load_unique_axis_rankings(
        split_metadata
    )

    main_df = _prepare_split_metric_row(
        ranking_df=ranking_df,
        axis="main",
        wrap_width=wrap_width,
    )

    secondary_df = _prepare_split_metric_row(
        ranking_df=ranking_df,
        axis="secondary",
        wrap_width=wrap_width,
    )

    metric_specs = [
        (
            "coverage_error",
            "Absolute coverage error from 90%",
            "Percentage points",
        ),
        (
            "rmse",
            "RMSE",
            "Error",
        ),
        (
            "mae",
            "MAE",
            "Error",
        ),
        (
            "mean_interval_width",
            "Mean prediction interval width",
            "Interval width",
        ),
    ]

    figure, axes = plt.subplots(
        nrows=2,
        ncols=len(metric_specs),
        figsize=figsize,
        sharey="row",
    )

    def draw_metric_row(
        row_axes,
        axis_df: pd.DataFrame,
        axis_name: str,
    ) -> None:
        y_positions = list(
            range(len(axis_df))
        )

        for column_index, (
            metric_column,
            metric_title,
            x_label,
        ) in enumerate(metric_specs):
            ax = row_axes[column_index]

            bars = ax.barh(
                y_positions,
                axis_df[metric_column],
                height=bar_height,
            )

            ax.set_title(
                metric_title,
                fontsize=12,
            )

            ax.set_xlabel(
                x_label,
            )

            ax.grid(
                axis="x",
                alpha=0.3,
            )

            ax.set_yticks(
                y_positions
            )

            if column_index == 0:
                ax.set_yticklabels(
                    axis_df["display_name"],
                    fontsize=7.5,
                )

                ax.set_ylabel(
                    f"{axis_name} split definitions",
                    labelpad=12,
                )
            else:
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

            maximum = float(
                axis_df[metric_column].max()
            )

            offset = (
                maximum * 0.015
                if maximum > 0
                else 0.001
            )

            for bar, value in zip(
                bars,
                axis_df[metric_column],
            ):
                ax.text(
                    bar.get_width() + offset,
                    (
                        bar.get_y()
                        + bar.get_height() / 2
                    ),
                    f"{value:.3f}",
                    va="center",
                    fontsize=7,
                )

            ax.set_xlim(
                left=0,
                right=(
                    maximum * 1.18
                    if maximum > 0
                    else 1
                ),
            )

        row_axes[0].text(
            -0.02,
            1.08,
            axis_name,
            transform=row_axes[0].transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
        )

    draw_metric_row(
        row_axes=axes[0],
        axis_df=main_df,
        axis_name="Main axis",
    )

    draw_metric_row(
        row_axes=axes[1],
        axis_df=secondary_df,
        axis_name="Secondary axis",
    )

    figure.suptitle(
        (
            "Effect of train/test split definition on QRF performance\n"
            "Main and secondary axes are ranked independently; "
            "lower metric values are better"
        ),
        fontsize=16,
    )

    figure.subplots_adjust(
        left=0.32,
        right=0.98,
        top=0.91,
        bottom=0.05,
        hspace=0.22,
        wspace=0.20,
    )

    plt.show()

    main_output = main_df.copy()
    main_output["axis"] = "main"

    secondary_output = secondary_df.copy()
    secondary_output["axis"] = (
        "secondary"
    )

    return pd.concat(
        [
            main_output,
            secondary_output,
        ],
        ignore_index=True,
    )


def plot_all_split_effects(
    split_metadata: pd.DataFrame | str | Path,
    figsize: tuple[int, int] = (24, 18),
) -> pd.DataFrame:
    """
    Show the effect of all split definitions on QRF performance.

    Main and secondary rankings are displayed separately in one figure.
    Lower QRF score is better.

    Returns a long DataFrame containing all plotted values.
    """
    unique_split_df = _load_unique_qrf_splits(
        split_metadata
    )

    main_df = _prepare_axis_split_effects(
        split_df=unique_split_df,
        axis="main",
    )

    secondary_df = _prepare_axis_split_effects(
        split_df=unique_split_df,
        axis="secondary",
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
    )

    def draw_axis(
        ax: plt.Axes,
        axis_df: pd.DataFrame,
        axis_title: str,
    ) -> None:
        y_positions = range(len(axis_df))

        bars = ax.barh(
            y_positions,
            axis_df["qrf_score"],
        )

        ax.set_yticks(
            list(y_positions)
        )

        ax.set_yticklabels(
            axis_df["display_name"],
            fontsize=8,
        )

        ax.set_xlabel(
            "QRF composite score — lower is better"
        )

        ax.set_title(axis_title)

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        max_score = axis_df["qrf_score"].max()

        annotation_offset = (
            max_score * 0.015
            if max_score > 0
            else 0.01
        )

        for bar, row in zip(
            bars,
            axis_df.itertuples(),
        ):
            annotation = (
                f"Rank {int(row.qrf_rank)} | "
                f"{row.split_type} | "
                f"Cov {row.coverage_percent:.1f}% | "
                f"RMSE {row.rmse:.3f}"
            )

            ax.text(
                bar.get_width() + annotation_offset,
                bar.get_y() + bar.get_height() / 2,
                annotation,
                va="center",
                fontsize=7,
            )

        ax.set_xlim(
            0,
            max_score * 1.55,
        )

    draw_axis(
        ax=axes[0],
        axis_df=main_df,
        axis_title=(
            "Main axis — effect of train/test split"
        ),
    )

    draw_axis(
        ax=axes[1],
        axis_df=secondary_df,
        axis_title=(
            "Secondary axis — effect of train/test split"
        ),
    )

    figure.suptitle(
        (
            "QRF performance across all split definitions\n"
            "Ranking combines coverage calibration, RMSE, "
            "MAE and prediction-interval width"
        ),
        fontsize=15,
    )

    figure.subplots_adjust(
        left=0.25,
        right=0.97,
        top=0.91,
        bottom=0.06,
        wspace=0.85,
    )

    plt.show()

    main_output = main_df.copy()
    main_output["axis"] = "main"

    secondary_output = secondary_df.copy()
    secondary_output["axis"] = "secondary"

    return pd.concat(
        [
            main_output,
            secondary_output,
        ],
        ignore_index=True,
    )