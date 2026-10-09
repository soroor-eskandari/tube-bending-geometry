from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.pipeline.ml.hgp.utils.experiments.hyper_parameter_tuning import (
    HyperParameterTuning,
)

from src.pipeline.ml.hgp.utils.experiments.hgp_model_trainer import (
    train_and_predict,
)


class SecondaryHyperParameterTuning(
    HyperParameterTuning
):
    """
    HGP tuner dedicated to Secondary-axis.

    All configured objective strategies use the same
    evaluated CV candidate pool.

    One winner is selected for every objective strategy.

    Every winner is then:
        1. evaluated on the real held-out test set
        2. stored in the results DataFrame
        3. plotted separately
        4. saved under HGP/results/tuning_plots/secondary
    """

    def run(
        self,
    ) -> pd.DataFrame:

        results_df = (
            self.run_axis(
                axis="secondary"
            )
        )

        project_root = Path(
            self.config[
                "_project_root"
            ]
        ).resolve()

        output_path = (
            project_root
            / "src"
            / "pipeline"
            / "ml"
            / "hgp"
            / "data"
            / "hgp_tuning_secondary_results.parquet"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists():
            output_path.unlink()

        results_df.to_parquet(
            output_path,
            index=False,
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "SECONDARY tuning finished."
        )

        print(
            f"Saved results:\n"
            f"{output_path}"
        )

        print(
            "=" * 80
        )

        self.plot_objective_winners(
            results_df=results_df
        )

        return results_df

    # =========================================================
    # Plot every objective winner
    # =========================================================

    def plot_objective_winners(
        self,
        *,
        results_df: pd.DataFrame,
    ) -> None:

        project_root = Path(
            self.config[
                "_project_root"
            ]
        ).resolve()

        (
            geometry_df,
            geometry_source,
            _,
        ) = self._load_geometry(
            project_root
        )

        if geometry_source != "real":
            raise ValueError(
                "Secondary tuning plots "
                "must use real geometry."
            )

        split_metadata_df = (
            pd.read_parquet(
                self._resolve_path(
                    project_root,
                    self.config[
                        "paths"
                    ][
                        "split_metadata"
                    ],
                )
            )
        )

        split_row = (
            self._best_split_row(
                split_metadata_df=(
                    split_metadata_df
                ),
                axis="secondary",
            )
        )

        train_df, test_df = (
            self._split_geometry(
                geometry_df=geometry_df,
                split_row=split_row,
            )
        )

        feature_columns = list(
            self.config[
                "features"
            ][
                "input_columns"
            ]
        )

        aggregation_columns = list(
            self.config[
                "features"
            ][
                "aggregation_columns"
            ]
        )

        target_column = (
            self.config[
                "features"
            ][
                "target_columns"
            ][
                "secondary"
            ]
        )

        output_directory = (
            project_root
            / "src"
            / "pipeline"
            / "ml"
            / "hgp"
            / "results"
            / "tuning_plots"
            / "secondary"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for (
            _,
            result_row,
        ) in results_df.iterrows():

            strategy = str(
                result_row[
                    "objective_strategy"
                ]
            )

            model_config = (
                result_row[
                    "best_model_config"
                ]
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                "Plotting Secondary "
                f"winner: {strategy}"
            )

            (
                _,
                predictions,
            ) = train_and_predict(
                train_df=train_df,
                test_df=test_df,
                feature_columns=(
                    feature_columns
                ),
                target_column=(
                    target_column
                ),
                model_config=(
                    model_config
                ),
                group_column=(
                    self.group_column
                ),
                aggregation_columns=(
                    aggregation_columns
                ),
            )

            self._plot_prediction_groups(
                test_df=test_df,
                predictions=(
                    predictions
                ),
                target_column=(
                    target_column
                ),
                strategy=strategy,
                result_row=(
                    result_row
                ),
                model_config=(
                    model_config
                ),
                output_directory=(
                    output_directory
                ),
            )

    # =========================================================
    # One complete figure
    # =========================================================

    def _plot_prediction_groups(
        self,
        *,
        test_df: pd.DataFrame,
        predictions,
        target_column: str,
        strategy: str,
        result_row: pd.Series,
        model_config: dict,
        output_directory: Path,
    ) -> None:

        prediction_df = (
            pd.DataFrame(
                {
                    "Group_ID": (
                        test_df[
                            self.group_column
                        ].to_numpy()
                    ),
                    "Experiment_ID": (
                        test_df[
                            "Experiment_ID"
                        ].to_numpy()
                    ),
                    "angle": (
                        pd.to_numeric(
                            test_df[
                                self.curve_order_column
                            ],
                            errors="raise",
                        )
                        .to_numpy(
                            dtype=float
                        )
                    ),
                    "y_true": (
                        pd.to_numeric(
                            test_df[
                                target_column
                            ],
                            errors="raise",
                        )
                        .to_numpy(
                            dtype=float
                        )
                    ),
                    "y_mean": (
                        np.asarray(
                            predictions.mean,
                            dtype=float,
                        )
                    ),
                    "y_lower": (
                        np.asarray(
                            predictions.lower,
                            dtype=float,
                        )
                    ),
                    "y_upper": (
                        np.asarray(
                            predictions.upper,
                            dtype=float,
                        )
                    ),
                }
            )
        )

        group_ids = sorted(
            prediction_df[
                "Group_ID"
            ]
            .unique()
            .tolist()
        )

        number_of_columns = min(
            5,
            len(group_ids),
        )

        number_of_rows = (
            math.ceil(
                len(group_ids)
                / number_of_columns
            )
        )

        fig, axes = (
            plt.subplots(
                nrows=number_of_rows,
                ncols=number_of_columns,
                figsize=(
                    5.1
                    * number_of_columns,
                    5.0
                    * number_of_rows,
                ),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
        )

        flat_axes = (
            axes.ravel()
        )

        for (
            plot_index,
            group_id,
        ) in enumerate(
            group_ids
        ):

            ax = (
                flat_axes[
                    plot_index
                ]
            )

            raw_group = (
                prediction_df[
                    prediction_df[
                        "Group_ID"
                    ].eq(
                        group_id
                    )
                ]
                .copy()
            )

            raw_group = (
                raw_group.sort_values(
                    [
                        "angle",
                        "Experiment_ID",
                    ]
                )
            )

            curve_df = (
                raw_group
                .groupby(
                    "angle",
                    as_index=False,
                    sort=True,
                )
                .agg(
                    y_true=(
                        "y_true",
                        "mean",
                    ),
                    y_mean=(
                        "y_mean",
                        "mean",
                    ),
                    y_lower=(
                        "y_lower",
                        "mean",
                    ),
                    y_upper=(
                        "y_upper",
                        "mean",
                    ),
                )
            )

            x = (
                curve_df[
                    "angle"
                ].to_numpy()
            )

            ax.fill_between(
                x,
                curve_df[
                    "y_lower"
                ].to_numpy(),
                curve_df[
                    "y_upper"
                ].to_numpy(),
                alpha=0.30,
                label=(
                    "Prediction interval"
                ),
                zorder=1,
            )

            ax.plot(
                x,
                curve_df[
                    "y_lower"
                ].to_numpy(),
                linewidth=0.75,
                linestyle=":",
                alpha=0.8,
                zorder=2,
            )

            ax.plot(
                x,
                curve_df[
                    "y_upper"
                ].to_numpy(),
                linewidth=0.75,
                linestyle=":",
                alpha=0.8,
                zorder=2,
            )

            ax.plot(
                x,
                curve_df[
                    "y_mean"
                ].to_numpy(),
                linewidth=1.0,
                label=(
                    "Prediction mean"
                ),
                zorder=4,
            )

            ax.plot(
                x,
                curve_df[
                    "y_true"
                ].to_numpy(),
                linewidth=1.0,
                linestyle="--",
                label="Actual",
                zorder=5,
            )

            experiment_count = int(
                raw_group[
                    "Experiment_ID"
                ]
                .dropna()
                .nunique()
            )

            residual = (
                raw_group[
                    "y_mean"
                ].to_numpy()
                - raw_group[
                    "y_true"
                ].to_numpy()
            )

            mae = float(
                np.mean(
                    np.abs(
                        residual
                    )
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            residual
                        )
                    )
                )
            )

            inside = (
                (
                    raw_group[
                        "y_true"
                    ]
                    >= raw_group[
                        "y_lower"
                    ]
                )
                & (
                    raw_group[
                        "y_true"
                    ]
                    <= raw_group[
                        "y_upper"
                    ]
                )
            )

            coverage = float(
                inside.mean()
            )

            mean_width = float(
                np.mean(
                    raw_group[
                        "y_upper"
                    ]
                    - raw_group[
                        "y_lower"
                    ]
                )
            )

            ax.set_title(
                (
                    f"Group {group_id}\n"
                    f"{experiment_count} "
                    "source experiment"
                    f"{'s' if experiment_count != 1 else ''}"
                ),
                fontsize=10,
            )

            ax.set_xlabel(
                "Angle [degree]",
                fontsize=8,
            )

            ax.set_ylabel(
                target_column,
                fontsize=8,
            )

            ax.set_ylim(
                20.5,
                22.5,
            )

            ax.grid(
                True,
                alpha=0.20,
            )

            ax.tick_params(
                labelsize=8
            )

            metric_text = (
                f"Coverage: "
                f"{coverage:.3f}\n"
                f"MAE: {mae:.4f}    "
                f"RMSE: {rmse:.4f}\n"
                f"Mean width: "
                f"{mean_width:.4f}"
            )

            ax.text(
                0.5,
                -0.25,
                metric_text,
                transform=(
                    ax.transAxes
                ),
                ha="center",
                va="top",
                fontsize=7.2,
                bbox={
                    "boxstyle": (
                        "round,pad=0.30"
                    ),
                    "facecolor": (
                        "white"
                    ),
                    "edgecolor": (
                        "0.80"
                    ),
                    "alpha": 0.95,
                },
                clip_on=False,
            )

        for unused_index in range(
            len(group_ids),
            len(flat_axes),
        ):
            flat_axes[
                unused_index
            ].axis(
                "off"
            )

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        cv_curve = float(
            result_row[
                "cv_curve_distance_norm"
            ]
        )

        cv_trend = float(
            result_row[
                "cv_trend_shape_loss"
            ]
        )

        cv_coverage = float(
            result_row[
                "cv_coverage"
            ]
        )

        cv_rmse = float(
            result_row[
                "cv_rmse_norm"
            ]
        )

        test_curve = float(
            result_row[
                "test_curve_distance_norm"
            ]
        )

        test_trend = float(
            result_row[
                "test_trend_shape_loss"
            ]
        )

        test_coverage = float(
            result_row[
                "test_coverage"
            ]
        )

        test_rmse = float(
            result_row[
                "test_rmse_norm"
            ]
        )

        mean_params = dict(
            model_config.get(
                "mean_kernel_params",
                {},
            )
        )

        noise_params = dict(
            model_config.get(
                "noise_kernel_params",
                {},
            )
        )

        model_text = (
            f"mean_kernel="
            f"{model_config.get('mean_kernel')} | "
            f"mean_alpha="
            f"{model_config.get('mean_gp_alpha')} | "
            f"noise_kernel="
            f"{model_config.get('noise_kernel')} | "
            f"noise_alpha="
            f"{model_config.get('noise_gp_alpha')} | "
            f"noise_ls="
            f"{noise_params.get('initial_length_scale')} | "
            f"constant="
            f"{mean_params.get('constant_value')}"
        )

        figure_title = (
            "HGP SECONDARY objective comparison\n"
            f"Strategy: {strategy}\n"
            f"CV: curve={cv_curve:.4f} | "
            f"trend={cv_trend:.4f} | "
            f"RMSE={cv_rmse:.4f} | "
            f"coverage={cv_coverage:.2f}%\n"
            f"TEST: curve={test_curve:.4f} | "
            f"trend={test_trend:.4f} | "
            f"RMSE={test_rmse:.4f} | "
            f"coverage={test_coverage:.2f}%\n"
            f"{model_text}"
        )

        fig.suptitle(
            figure_title,
            fontsize=13,
            y=0.995,
        )

        handles, labels = (
            flat_axes[0]
            .get_legend_handles_labels()
        )

        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(
                    0.5,
                    0.91,
                ),
                ncol=3,
                frameon=True,
            )

        fig.subplots_adjust(
            top=0.84,
            bottom=0.08,
            hspace=0.70,
            wspace=0.25,
        )

        output_path = (
            output_directory
            / (
                "secondary__"
                f"{strategy}.png"
            )
        )

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        print(
            f"Saved plot:\n"
            f"{output_path}"
        )

        plt.show()

        plt.close(
            fig
        )