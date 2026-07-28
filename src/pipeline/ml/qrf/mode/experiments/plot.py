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
    ylim: tuple[float, float] | None = None,
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

    if ylim is not None:
        ax.set_ylim(ylim)

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
        grid[0, 1],
        sharex=main_ax,
        sharey=main_ax,
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

    combined_metric_values = pd.concat(
        [
            metrics_df.loc[
                metrics_df["axis"].eq("main")
                & metrics_df["split_index"].isin(
                    selected_main_df["split_index"]
                ),
                metric,
            ],
            metrics_df.loc[
                metrics_df["axis"].eq("secondary")
                & metrics_df["split_index"].isin(
                    selected_secondary_df["split_index"]
                ),
                metric,
            ],
        ],
        ignore_index=True,
    )

    combined_metric_values = pd.to_numeric(
        combined_metric_values,
        errors="coerce",
    ).dropna()

    if not combined_metric_values.empty:
        metric_min = float(
            combined_metric_values.min()
        )

        metric_max = float(
            combined_metric_values.max()
        )

        metric_range = (
            metric_max
            - metric_min
        )

        padding = (
            metric_range * 0.05
            if metric_range > 0
            else max(
                abs(metric_min),
                1.0,
            ) * 0.05
        )

        shared_ylim = (
            metric_min - padding,
            metric_max + padding,
        )

        main_ax.set_ylim(shared_ylim)
        secondary_ax.set_ylim(shared_ylim)

    

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
    selected_splits_by_axis: (
        dict[str, list[int]] | None
    ) = None,
    figsize: tuple[int, int] = (24, 18),
) -> pd.DataFrame:
    """
    Show the effect of selected split definitions on QRF performance.

    Main and secondary rankings are displayed separately.

    When selected_splits_by_axis is provided, each axis is filtered
    independently. This is required because the best and worst splits
    can be different for main and secondary axes.

    Lower QRF score is better.

    Parameters
    ----------
    split_metadata:
        DataFrame or path to various_splits.parquet.

    selected_splits_by_axis:
        Optional axis-specific split selection:

        {
            "main": [1, 2, 3, ...],
            "secondary": [4, 5, 6, ...],
        }

    figsize:
        Matplotlib figure size.

    Returns
    -------
    pd.DataFrame
        Long DataFrame containing the plotted main and secondary
        values.
    """
    unique_split_df = _load_unique_qrf_splits(
        split_metadata
    )

    if selected_splits_by_axis is not None:
        required_axes = {
            "main",
            "secondary",
        }

        missing_axes = (
            required_axes
            - set(selected_splits_by_axis)
        )

        if missing_axes:
            raise KeyError(
                "selected_splits_by_axis is missing axes: "
                f"{sorted(missing_axes)}"
            )

        main_selected_indices = {
            int(split_index)
            for split_index
            in selected_splits_by_axis["main"]
        }

        secondary_selected_indices = {
            int(split_index)
            for split_index
            in selected_splits_by_axis["secondary"]
        }

        main_split_df = unique_split_df[
            unique_split_df["split_index"]
            .astype(int)
            .isin(main_selected_indices)
        ].copy()

        secondary_split_df = unique_split_df[
            unique_split_df["split_index"]
            .astype(int)
            .isin(secondary_selected_indices)
        ].copy()

        if main_split_df.empty:
            raise ValueError(
                "No main-axis split rows matched "
                "selected_splits_by_axis['main']."
            )

        if secondary_split_df.empty:
            raise ValueError(
                "No secondary-axis split rows matched "
                "selected_splits_by_axis['secondary']."
            )

    else:
        main_split_df = unique_split_df.copy()
        secondary_split_df = unique_split_df.copy()

    main_df = _prepare_axis_split_effects(
        split_df=main_split_df,
        axis="main",
    )

    secondary_df = _prepare_axis_split_effects(
        split_df=secondary_split_df,
        axis="secondary",
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,
    )

    combined_scores = pd.concat(
        [
            main_df["qrf_score"],
            secondary_df["qrf_score"],
        ],
        ignore_index=True,
    )

    combined_scores = pd.to_numeric(
        combined_scores,
        errors="coerce",
    ).dropna()

    if combined_scores.empty:
        raise ValueError(
            "No valid QRF scores are available for plotting."
        )

    shared_max_score = float(
        combined_scores.max()
    )

    shared_xlim = (
        0.0,
        (
            shared_max_score * 1.55
            if shared_max_score > 0
            else 1.0
        ),
    )

    def draw_axis(
        ax: plt.Axes,
        axis_df: pd.DataFrame,
        axis_title: str,
    ) -> None:
        y_positions = list(
            range(len(axis_df))
        )

        bars = ax.barh(
            y_positions,
            axis_df["qrf_score"],
        )

        ax.set_yticks(
            y_positions
        )

        ax.set_yticklabels(
            axis_df["display_name"],
            fontsize=8,
        )

        ax.set_xlabel(
            "QRF composite score — lower is better"
        )

        ax.set_title(
            axis_title
        )

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        annotation_offset = (
            shared_max_score * 0.015
            if shared_max_score > 0
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
                (
                    bar.get_y()
                    + bar.get_height() / 2
                ),
                annotation,
                va="center",
                fontsize=7,
            )

        ax.set_xlim(
            shared_xlim
        )

    draw_axis(
        ax=axes[0],
        axis_df=main_df,
        axis_title=(
            "Main axis — selected train/test splits"
        ),
    )

    draw_axis(
        ax=axes[1],
        axis_df=secondary_df,
        axis_title=(
            "Secondary axis — selected train/test splits"
        ),
    )

    figure.suptitle(
        (
            "QRF performance for axis-specific top and worst splits\n"
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


def plot_split_coverage_by_split_name(
    split_metadata: pd.DataFrame | str | Path,
    *,
    sort_by: str = "coverage",
    ascending: bool = False,
    figsize: tuple[int, int] = (18, 6),
    marker_size: float = 44.0,
    show_split_names: bool = False,
    show: bool = True,
) -> tuple[
    plt.Figure,
    pd.DataFrame,
]:
    """
    Plot main and secondary QRF coverage across split definitions.

    x-axis positions represent sorted split ranks. The returned DataFrame
    preserves the exact split names.
    """
    if isinstance(
        split_metadata,
        pd.DataFrame,
    ):
        split_df = split_metadata.copy()
    else:
        split_df = pd.read_parquet(
            Path(split_metadata)
        )

    required_columns = {
        "split_index",
        "split_name",
        "qrf_coverage_main",
        "qrf_coverage_secondary",
    }

    missing_columns = required_columns.difference(
        split_df.columns
    )

    if missing_columns:
        raise KeyError(
            "split_metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    split_df = (
        split_df
        .drop_duplicates(
            subset=[
                "split_index",
            ]
        )
        .copy()
    )

    axis_frames = []

    for axis in (
        "main",
        "secondary",
    ):
        axis_df = split_df[
            [
                "split_index",
                "split_name",
                f"qrf_coverage_{axis}",
            ]
        ].copy()

        axis_df = axis_df.rename(
            columns={
                f"qrf_coverage_{axis}": "coverage",
            }
        )
        axis_df["axis"] = axis
        axis_df["coverage"] = pd.to_numeric(
            axis_df["coverage"],
            errors="coerce",
        )
        axis_df = axis_df.dropna(
            subset=[
                "coverage",
            ]
        )
        axis_frames.append(axis_df)

    plot_df = pd.concat(
        axis_frames,
        ignore_index=True,
    )

    if plot_df.empty:
        raise ValueError(
            "No valid coverage values are available for plotting."
        )

    if sort_by not in {
        "coverage",
        "split_name",
        "split_index",
    }:
        raise ValueError(
            "sort_by must be one of: "
            "'coverage', 'split_name', 'split_index'."
        )

    plot_df = (
        plot_df
        .sort_values(
            [
                "axis",
                sort_by,
            ],
            ascending=[
                True,
                ascending,
            ],
        )
        .reset_index(drop=True)
    )

    plot_df["x_position"] = (
        plot_df
        .groupby("axis")
        .cumcount()
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
    )

    coverage_min = float(
        plot_df["coverage"].min()
    )
    coverage_max = float(
        plot_df["coverage"].max()
    )
    coverage_padding = max(
        (coverage_max - coverage_min) * 0.08,
        1.0,
    )
    y_limits = (
        max(
            0.0,
            coverage_min - coverage_padding,
        ),
        min(
            100.0,
            coverage_max + coverage_padding,
        ),
    )

    for ax, axis in zip(
        axes,
        (
            "main",
            "secondary",
        ),
        strict=True,
    ):
        axis_df = plot_df[
            plot_df["axis"].eq(axis)
        ].copy()

        ax.plot(
            axis_df["x_position"],
            axis_df["coverage"],
            marker="o",
            markersize=marker_size ** 0.5,
            linewidth=1.8,
        )

        ax.scatter(
            axis_df["x_position"],
            axis_df["coverage"],
            s=marker_size,
            zorder=3,
        )

        ax.set_title(
            f"{axis.capitalize()} axis"
        )
        ax.set_xlabel("")
        ax.set_ylim(
            y_limits
        )
        ax.grid(
            axis="y",
            linestyle=":",
            alpha=0.35,
        )

        if show_split_names:
            ax.set_xticks(
                axis_df["x_position"]
            )
            ax.set_xticklabels(
                [
                    _shorten_text(
                        split_name,
                        max_length=35,
                    )
                    for split_name
                    in axis_df["split_name"]
                ],
                rotation=75,
                ha="right",
                fontsize=7,
            )
        else:
            last_position = int(
                axis_df["x_position"].max()
            )
            middle_position = int(
                round(last_position / 2)
            )
            tick_positions = [
                0,
                middle_position,
                last_position,
            ]
            position_to_split_name = (
                axis_df
                .set_index("x_position")["split_name"]
                .to_dict()
            )
            tick_labels = [
                _shorten_text(
                    position_to_split_name[
                        position
                    ],
                    max_length=45,
                )
                for position
                in tick_positions
            ]

            ax.set_xticks(
                tick_positions
            )
            ax.set_xticklabels(
                tick_labels,
                rotation=90,
                va="top",
                ha="center",
            )

    axes[0].set_ylabel(
        "Coverage (%)"
    )

    figure.suptitle(
        "QRF coverage by split definition",
        fontsize=14,
    )
    figure.tight_layout()

    if show:
        plt.show()

    return (
        figure,
        plot_df,
    )
# -----------------------------------------------------------------------------
# Top-3 / worst-3 bending-setup train-value maps
# -----------------------------------------------------------------------------

import ast
from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.lines import Line2D


def _coerce_collection(value: object) -> list[object]:
    """
    Convert parquet/list-like values to a Python list.

    This supports real lists, NumPy arrays, tuples, sets and stringified
    list values stored by parquet/csv pipelines.
    """
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]

        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            return list(parsed)

        return [parsed]

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    return [value]


def _values_equal(left: object, right: object) -> bool:
    """Compare numeric and string representations safely."""
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass

    try:
        return bool(
            np.isclose(
                float(left),
                float(right),
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        return (
            str(left).strip()
            == str(right).strip()
        )


def _sorted_unique_values(
    series: pd.Series,
) -> list[object]:
    """Return stable unique values, numerically sorted when possible."""
    values = (
        series
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    try:
        return sorted(
            values,
            key=float,
        )
    except (TypeError, ValueError):
        return sorted(
            values,
            key=lambda item: str(item).lower(),
        )


def _normalise_feature_values(
    values: Sequence[object],
) -> list[tuple[object, float]]:
    """
    Map the actual numeric range of one feature to [-1, 1].

    Numeric values preserve their relative distances. Categorical values
    receive equally spaced positions.
    """
    clean_values = list(values)

    if not clean_values:
        return []

    if len(clean_values) == 1:
        return [
            (
                clean_values[0],
                0.0,
            )
        ]

    try:
        numeric_values = np.asarray(
            [
                float(value)
                for value in clean_values
            ],
            dtype=float,
        )

        minimum = float(
            np.nanmin(numeric_values)
        )
        maximum = float(
            np.nanmax(numeric_values)
        )

        if np.isclose(
            minimum,
            maximum,
        ):
            positions = np.zeros(
                len(clean_values),
                dtype=float,
            )
        else:
            positions = (
                2.0
                * (
                    numeric_values
                    - minimum
                )
                / (
                    maximum
                    - minimum
                )
                - 1.0
            )
    except (TypeError, ValueError):
        positions = np.linspace(
            -1.0,
            1.0,
            num=len(clean_values),
        )

    return list(
        zip(
            clean_values,
            positions.tolist(),
            strict=True,
        )
    )


def _extract_feature_values(
    bending_setup_df: pd.DataFrame,
    feature_names: Sequence[str],
) -> dict[str, list[object]]:
    """Extract every observed value of each requested bending feature."""
    missing = [
        feature
        for feature in feature_names
        if feature not in bending_setup_df.columns
    ]

    if missing:
        raise KeyError(
            "Bending-setup data is missing feature columns: "
            f"{missing}"
        )

    return {
        feature: _sorted_unique_values(
            bending_setup_df[feature]
        )
        for feature in feature_names
    }


def _resolve_group_column(
    bending_setup_df: pd.DataFrame,
    requested_column: str | None,
) -> str:
    """Resolve the bending-setup group identifier column."""
    if requested_column is not None:
        if requested_column not in bending_setup_df.columns:
            raise KeyError(
                "The requested group column was not found in "
                f"bending_setup_df: {requested_column!r}"
            )

        return requested_column

    candidates = (
        "Group_ID",
        "group_id",
        "Group ID",
        "group",
    )

    for candidate in candidates:
        if candidate in bending_setup_df.columns:
            return candidate

    raise KeyError(
        "Could not infer the bending-setup group column. "
        "Pass bending_group_column explicitly."
    )


def _build_train_values_by_split(
    ranking_df: pd.DataFrame,
    bending_setup_df: pd.DataFrame,
    feature_names: Sequence[str],
    bending_group_column: str,
    train_group_ids_column: str,
) -> dict[int, dict[str, list[object]]]:
    """
    Derive the exact feature values represented in each split's train set.

    The split metadata already stores train_group_ids. Using those IDs is
    safer than interpreting split_name, because split_name normally
    describes the held-out/test condition.
    """
    if train_group_ids_column not in ranking_df.columns:
        raise KeyError(
            "split_metadata does not contain "
            f"{train_group_ids_column!r}."
        )

    output: dict[int, dict[str, list[object]]] = {}

    for row in ranking_df[
        [
            "split_index",
            train_group_ids_column,
        ]
    ].itertuples(index=False):
        split_index = int(
            row.split_index
        )

        train_group_ids = _coerce_collection(
            getattr(
                row,
                train_group_ids_column,
            )
        )

        train_mask = bending_setup_df[
            bending_group_column
        ].apply(
            lambda group_id: any(
                _values_equal(
                    group_id,
                    train_group_id,
                )
                for train_group_id
                in train_group_ids
            )
        )

        train_bending_df = bending_setup_df.loc[
            train_mask
        ]

        output[split_index] = {
            feature: _sorted_unique_values(
                train_bending_df[feature]
            )
            for feature in feature_names
        }

    return output


def _calculate_train_feature_stds(
    feature_values: Mapping[
        str,
        Sequence[object],
    ],
    train_values: Mapping[
        str,
        Sequence[object],
    ],
) -> dict[str, float]:
    """
    Calculate per-feature STD of represented train values.

    Values are measured on the same normalised [-1, 1] scale used by the
    value-map plot, so the STDs are comparable across features with different
    physical units.
    """
    output: dict[str, float] = {}

    for feature, possible_values in feature_values.items():
        normalised_values = _normalise_feature_values(
            possible_values
        )

        selected_positions = []

        for selected_value in train_values.get(
            feature,
            [],
        ):
            for value, x_position in normalised_values:
                if _values_equal(
                    value,
                    selected_value,
                ):
                    selected_positions.append(
                        float(x_position)
                    )
                    break

        if len(selected_positions) < 2:
            output[feature] = 0.0
        else:
            output[feature] = float(
                np.std(
                    selected_positions,
                    ddof=0,
                )
            )

    return output


def _mean_train_feature_std(
    feature_stds: Mapping[
        str,
        float,
    ],
) -> float:
    """Return the mean normalised train-value STD across features."""
    values = np.asarray(
        list(
            feature_stds.values()
        ),
        dtype=float,
    )

    if values.size == 0:
        return float("nan")

    return float(
        np.nanmean(
            values
        )
    )


def _prepare_ranked_split_pairs(
    ranking_df: pd.DataFrame,
    top_n: int,
    worst_n: int,
) -> list[dict[str, object]]:
    """
    Pair main and secondary splits by ranking position.

    Worst rank 1 means the split with the largest numerical QRF rank.
    """
    selected_by_axis: dict[
        str,
        list[pd.Series],
    ] = {}

    for axis in (
        "main",
        "secondary",
    ):
        rank_column = (
            f"qrf_rank_{axis}"
        )

        axis_df = (
            ranking_df
            .dropna(
                subset=[
                    rank_column,
                ]
            )
            .copy()
        )

        axis_df[rank_column] = (
            pd.to_numeric(
                axis_df[rank_column],
                errors="raise",
            )
        )

        axis_df = axis_df.sort_values(
            rank_column,
            ascending=True,
        )

        top_rows = [
            row
            for _, row
            in axis_df.head(
                top_n
            ).iterrows()
        ]

        worst_rows = [
            row
            for _, row
            in axis_df.tail(
                worst_n
            )
            .sort_values(
                rank_column,
                ascending=False,
            )
            .iterrows()
        ]

        selected_by_axis[axis] = (
            top_rows
            + worst_rows
        )

    labels = (
        [
            f"Top rank {position}"
            for position
            in range(
                1,
                top_n + 1,
            )
        ]
        + [
            f"Worst rank {position}"
            for position
            in range(
                1,
                worst_n + 1,
            )
        ]
    )

    return [
        {
            "label": label,
            "main": selected_by_axis[
                "main"
            ][index],
            "secondary": selected_by_axis[
                "secondary"
            ][index],
        }
        for index, label
        in enumerate(labels)
    ]


def _format_metric_value(
    label: str,
    value: object,
) -> str:
    numeric_value = float(
        value
    )

    if "coverage" in label.lower():
        return (
            f"{numeric_value:.2f}%"
        )

    return (
        f"{numeric_value:.4f}"
    )


def _draw_split_value_matrix(
    ax: plt.Axes,
    *,
    split_row: pd.Series,
    axis: str,
    feature_values: Mapping[
        str,
        Sequence[object],
    ],
    train_values_by_split: Mapping[
        int,
        Mapping[
            str,
            Sequence[object],
        ],
    ],
    marker_size: float,
) -> None:
    """Draw one feature-value matrix and its axis-specific metrics."""
    split_index = int(
        split_row[
            "split_index"
        ]
    )

    split_name = str(
        split_row[
            "split_name"
        ]
    )

    train_values = (
        train_values_by_split[
            split_index
        ]
    )

    feature_names = list(
        feature_values
    )

    y_positions = (
        np.arange(
            len(feature_names)
        )[::-1]
    )

    for (
        y_position,
        feature,
    ) in zip(
        y_positions,
        feature_names,
        strict=True,
    ):
        possible_values = list(
            feature_values[
                feature
            ]
        )

        normalised_values = (
            _normalise_feature_values(
                possible_values
            )
        )

        selected_values = (
            train_values.get(
                feature,
                [],
            )
        )

        for (
            value,
            x_position,
        ) in normalised_values:
            is_train = any(
                _values_equal(
                    value,
                    train_value,
                )
                for train_value
                in selected_values
            )

            ax.scatter(
                x_position,
                y_position,
                s=marker_size,
                facecolors=(
                    "tab:blue"
                    if is_train
                    else "none"
                ),
                edgecolors=(
                    "tab:blue"
                    if is_train
                    else "black"
                ),
                linewidths=1.25,
                zorder=3,
            )

    ax.axvline(
        0.0,
        linewidth=0.8,
        alpha=0.35,
    )

    ax.set_xlim(
        -1.08,
        1.08,
    )

    ax.set_ylim(
        -0.65,
        len(feature_names) - 0.35,
    )

    ax.set_xticks(
        [
            -1.0,
            0.0,
            1.0,
        ]
    )

    ax.set_xlabel(
        "Normalised feature value"
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        feature_names
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    rank = int(
        split_row[
            f"qrf_rank_{axis}"
        ]
    )

    ax.set_title(
        (
            f"{axis.capitalize()} axis | "
            f"rank {rank}\n"
            f"{_shorten_text(split_name, 78)}"
        )
    )

    metric_candidates = [
        (
            "Train coverage",
            f"qrf_train_coverage_{axis}",
        ),
        (
            "MAE",
            f"qrf_mae_{axis}",
        ),
        (
            "RMSE",
            f"qrf_rmse_{axis}",
        ),
        (
            "PINAW",
            f"qrf_pinaw_{axis}",
        ),
        (
            "Mean interval width",
            (
                f"qrf_mean_interval_width_"
                f"{axis}"
            ),
        ),
        (
            "Test coverage",
            f"qrf_coverage_{axis}",
        ),
    ]

    metric_text = []

    train_feature_stds = (
        _calculate_train_feature_stds(
            feature_values=(
                feature_values
            ),
            train_values=(
                train_values
            ),
        )
    )
    mean_train_feature_std = (
        _mean_train_feature_std(
            train_feature_stds
        )
    )

    if pd.notna(
        mean_train_feature_std
    ):
        metric_text.append(
            (
                f"{'Norm train STD':<20} "
                f"{mean_train_feature_std:.4f}"
            )
        )

    for (
        label,
        column,
    ) in metric_candidates:
        if (
            column
            not in split_row.index
            or pd.isna(
                split_row[
                    column
                ]
            )
        ):
            continue

        metric_text.append(
            (
                f"{label:<20} "
                f"{_format_metric_value(
                    label,
                    split_row[column],
                )}"
            )
        )

    if metric_text:
        ax.text(
            1.03,
            0.5,
            "\n".join(
                metric_text
            ),
            transform=ax.transAxes,
            va="center",
            ha="left",
            family="monospace",
            fontsize=9,
            bbox={
                "boxstyle": (
                    "round,pad=0.45"
                ),
                "facecolor": "white",
                "alpha": 0.95,
            },
        )


def plot_top_worst_split_value_maps(
    split_metadata: (
        pd.DataFrame
        | str
        | Path
    ),
    bending_setup_df: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    bending_group_column: (
        str
        | None
    ) = None,
    train_group_ids_column: str = (
        "train_group_ids"
    ),
    top_n: int = 3,
    worst_n: int = 3,
    figsize: tuple[
        float,
        float,
    ] = (
        22,
        8,
    ),
    marker_size: float = 58.0,
    show: bool = True,
) -> tuple[
    list[plt.Figure],
    pd.DataFrame,
]:
    """
    Plot top-3 and worst-3 split train-value maps.

    Six figures are produced by default. Each figure contains:
        left: one main-axis split
        right: the corresponding secondary-axis split

    For every bending feature:
        blue filled circle = the value occurs in the train groups
        black open circle = the value does not occur in the train groups

    Every numeric feature is min-max normalised to [-1, 1] using its
    actual values. Categorical features use equally spaced positions.

    Train membership is derived from train_group_ids and the bending
    setup Group_ID column. split_name is used only as a title.

    Returns
    -------
    figures:
        The generated Matplotlib Figure objects.

    selected_splits_df:
        One record per displayed axis/split.
    """
    if isinstance(
        split_metadata,
        pd.DataFrame,
    ):
        raw_df = (
            split_metadata.copy()
        )
    else:
        raw_df = (
            pd.read_parquet(
                Path(
                    split_metadata
                )
            )
        )

    required_columns = {
        "split_index",
        "split_name",
        train_group_ids_column,
        "qrf_rank_main",
        "qrf_rank_secondary",
        "qrf_mae_main",
        "qrf_mae_secondary",
        "qrf_rmse_main",
        "qrf_rmse_secondary",
        "qrf_coverage_main",
        "qrf_coverage_secondary",
        (
            "qrf_mean_interval_width_"
            "main"
        ),
        (
            "qrf_mean_interval_width_"
            "secondary"
        ),
    }

    missing = (
        required_columns
        .difference(
            raw_df.columns
        )
    )

    if missing:
        raise KeyError(
            "split_metadata is missing columns: "
            f"{sorted(missing)}"
        )

    ranking_df = (
        raw_df
        .drop_duplicates(
            subset=[
                "split_index",
            ]
        )
        .reset_index(
            drop=True,
        )
    )

    resolved_group_column = (
        _resolve_group_column(
            bending_setup_df=(
                bending_setup_df
            ),
            requested_column=(
                bending_group_column
            ),
        )
    )

    feature_value_map = (
        _extract_feature_values(
            bending_setup_df=(
                bending_setup_df
            ),
            feature_names=(
                feature_names
            ),
        )
    )

    train_values_by_split = (
        _build_train_values_by_split(
            ranking_df=ranking_df,
            bending_setup_df=(
                bending_setup_df
            ),
            feature_names=(
                feature_names
            ),
            bending_group_column=(
                resolved_group_column
            ),
            train_group_ids_column=(
                train_group_ids_column
            ),
        )
    )

    ranked_pairs = (
        _prepare_ranked_split_pairs(
            ranking_df=ranking_df,
            top_n=top_n,
            worst_n=worst_n,
        )
    )

    figures: list[
        plt.Figure
    ] = []

    selected_records: list[
        dict[str, object]
    ] = []

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=(
                "tab:blue"
            ),
            markeredgecolor=(
                "tab:blue"
            ),
            label=(
                "Value present in train set"
            ),
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            label=(
                "Value absent from train set"
            ),
        ),
    ]

    for pair in ranked_pairs:
        figure, axes = (
            plt.subplots(
                nrows=1,
                ncols=2,
                figsize=figsize,
                sharex=True,
                sharey=True,
            )
        )

        for (
            ax,
            axis,
        ) in zip(
            axes,
            (
                "main",
                "secondary",
            ),
            strict=True,
        ):
            split_row = pair[
                axis
            ]

            _draw_split_value_matrix(
                ax=ax,
                split_row=(
                    split_row
                ),
                axis=axis,
                feature_values=(
                    feature_value_map
                ),
                train_values_by_split=(
                    train_values_by_split
                ),
                marker_size=(
                    marker_size
                ),
            )

            split_index = int(
                split_row[
                    "split_index"
                ]
            )
            train_feature_stds = (
                _calculate_train_feature_stds(
                    feature_values=(
                        feature_value_map
                    ),
                    train_values=(
                        train_values_by_split[
                            split_index
                        ]
                    ),
                )
            )

            selected_records.append(
                {
                    "figure_group": (
                        pair[
                            "label"
                        ]
                    ),
                    "axis": axis,
                    "split_index": split_index,
                    "split_name": (
                        split_row[
                            "split_name"
                        ]
                    ),
                    "rank": int(
                        split_row[
                            f"qrf_rank_{axis}"
                        ]
                    ),
                    "mean_train_feature_std": (
                        _mean_train_feature_std(
                            train_feature_stds
                        )
                    ),
                    "train_feature_std_by_feature": (
                        train_feature_stds
                    ),
                }
            )

        figure.suptitle(
            (
                f"{pair['label']}: "
                "bending-setup values "
                "represented in training"
            ),
            fontsize=15,
        )

        figure.legend(
            handles=(
                legend_handles
            ),
            loc="lower center",
            ncol=2,
            frameon=True,
        )

        figure.subplots_adjust(
            left=0.19,
            right=0.84,
            top=0.83,
            bottom=0.16,
            wspace=0.38,
        )

        figures.append(
            figure
        )

        if show:
            plt.show()

    return (
        figures,
        pd.DataFrame(
            selected_records
        ),
    )
