import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

logger = logging.getLogger(__name__)


class RFModelEvaluator:

    # =========================================================
    # METRICS
    # =========================================================
    @staticmethod
    def evaluate(y_true, y_pred):
        results = {}

        # -------------------------
        # Global metrics
        # -------------------------
        results["r2_global"] = r2_score(y_true, y_pred)
        results["mse_global"] = mean_squared_error(y_true, y_pred)
        results["rmse_global"] = np.sqrt(results["mse_global"])
        results["mae_global"] = mean_absolute_error(y_true, y_pred)

        # -------------------------
        # Per-feature metrics
        # -------------------------
        n_features = y_true.shape[1]

        r2_list, mse_list, rmse_list, mae_list = [], [], [], []

        for i in range(n_features):
            yt = y_true[:, i]
            yp = y_pred[:, i]

            mse = mean_squared_error(yt, yp)

            r2_list.append(r2_score(yt, yp))
            mse_list.append(mse)
            rmse_list.append(np.sqrt(mse))
            mae_list.append(mean_absolute_error(yt, yp))

        results["r2_per_feature"] = np.array(r2_list)
        results["mse_per_feature"] = np.array(mse_list)
        results["rmse_per_feature"] = np.array(rmse_list)
        results["mae_per_feature"] = np.array(mae_list)

        return results

    # =========================================================
    # UTIL
    # =========================================================
    @staticmethod
    def _prepare_dir(output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving evaluation outputs to: {output_dir}")

    @staticmethod
    def to_dataframe(results):
        return pd.DataFrame({
            "feature_index": np.arange(len(results["r2_per_feature"])),
            "r2": results["r2_per_feature"],
            "mse": results["mse_per_feature"],
            "rmse": results["rmse_per_feature"],
            "mae": results["mae_per_feature"],
        })

    # =========================================================
    # SAVE METRICS
    # =========================================================
    @staticmethod
    def save_results(results, output_dir: Path, prefix="model"):

        RFModelEvaluator._prepare_dir(output_dir)

        # Global
        global_df = pd.DataFrame([{
            "r2": results["r2_global"],
            "mse": results["mse_global"],
            "rmse": results["rmse_global"],
            "mae": results["mae_global"],
        }])

        global_path = output_dir / f"{prefix}_global_metrics.csv"
        global_df.to_csv(global_path, index=False)

        # Per-feature
        feature_df = RFModelEvaluator.to_dataframe(results)
        feature_path = output_dir / f"{prefix}_per_feature_metrics.csv"
        feature_df.to_csv(feature_path, index=False)

        logger.info(f"Saved metrics: {global_path}")
        logger.info(f"Saved metrics: {feature_path}")

    # =========================================================
    # SCATTER PLOT
    # =========================================================
    @staticmethod
    def plot_predictions(y_true, y_pred, output_dir: Path, prefix="model", feature_idx=None):

        RFModelEvaluator._prepare_dir(output_dir)

        if feature_idx is not None:
            y_true_plot = y_true[:, feature_idx]
            y_pred_plot = y_pred[:, feature_idx]
            filename = f"{prefix}_scatter_feature_{feature_idx}.png"
            title = f"{prefix} - Feature {feature_idx}"
        else:
            y_true_plot = y_true.flatten()
            y_pred_plot = y_pred.flatten()
            filename = f"{prefix}_scatter_global.png"
            title = f"{prefix} - Global"

        plt.figure()
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.5)

        min_val = min(y_true_plot.min(), y_pred_plot.min())
        max_val = max(y_true_plot.max(), y_pred_plot.max())

        plt.plot([min_val, max_val], [min_val, max_val])

        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(title)
        plt.grid()

        save_path = output_dir / filename
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved plot: {save_path}")

    # =========================================================
    # METRIC CURVE
    # =========================================================
    @staticmethod
    def plot_metric_per_feature(results, output_dir: Path, metric="rmse", prefix="model"):

        RFModelEvaluator._prepare_dir(output_dir)

        key = f"{metric}_per_feature"
        if key not in results:
            raise ValueError(f"{metric} not found in results")

        values = results[key]

        plt.figure()
        plt.plot(values)

        plt.xlabel("Feature Index (Angle)")
        plt.ylabel(metric.upper())
        plt.title(f"{prefix} - {metric.upper()} per Feature")
        plt.grid()

        save_path = output_dir / f"{prefix}_{metric}_per_feature.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved plot: {save_path}")

    # =========================================================
    # R2 (GLOBAL) + MSE (PER FEATURE)
    # Supports: single model OR comparison
    # =========================================================
    @staticmethod
    def plot_r2_mse_per_feature(main_results, output_dir: Path, prefix="model", sec_results=None):

        RFModelEvaluator._prepare_dir(output_dir)

        # -------------------------
        # Main model
        # -------------------------
        mse_main = main_results["mse_per_feature"]
        r2_main = main_results["r2_global"]

        fig, ax1 = plt.subplots()

        ax1.plot(mse_main, linestyle='--', label="MSE Main")
        ax1.set_xlabel("Feature Index (Angle)")
        ax1.set_ylabel("MSE")

        # -------------------------
        # Secondary model (optional)
        # -------------------------
        if sec_results is not None:
            mse_sec = sec_results["mse_per_feature"]
            r2_sec = sec_results["r2_global"]
            ax1.plot(mse_sec, linestyle='--', label="MSE Secondary")

        # -------------------------
        # R2 as horizontal line(s)
        # -------------------------
        ax2 = ax1.twinx()
        ax2.axhline(r2_main, linestyle='-', label=f"R2 Main ({r2_main:.3f})")

        if sec_results is not None:
            ax2.axhline(r2_sec, linestyle='-', label=f"R2 Secondary ({r2_sec:.3f})")

        ax2.set_ylabel("R2")

        plt.title(f"{prefix} - Global R2 & MSE per Feature")

        # -------------------------
        # Legend merge
        # -------------------------
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

        save_path = output_dir / f"{prefix}_r2_mse_per_feature.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved plot: {save_path}")

    # =========================================================
    # GLOBAL TABLE (MAIN vs SECONDARY)
    # =========================================================
    @staticmethod
    def plot_global_metrics_table(main_results, sec_results, output_dir: Path):

        RFModelEvaluator._prepare_dir(output_dir)

        data = pd.DataFrame([
            {
                "Model": "Main",
                "R2": main_results["r2_global"],
                "MSE": main_results["mse_global"],
                "RMSE": main_results["rmse_global"],
                "MAE": main_results["mae_global"],
            },
            {
                "Model": "Secondary",
                "R2": sec_results["r2_global"],
                "MSE": sec_results["mse_global"],
                "RMSE": sec_results["rmse_global"],
                "MAE": sec_results["mae_global"],
            }
        ])

        formatted = data.copy()
        for col in ["R2", "MSE", "RMSE", "MAE"]:
            if col == "R2":
                formatted[col] = formatted[col].map(lambda v: f"{v:.4f}")
            else:
                formatted[col] = formatted[col].map(lambda v: f"{v:.3e}")

        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.axis('off')

        table = ax.table(
            cellText=formatted.values,
            colLabels=formatted.columns,
            loc='center',
            cellLoc='center',
            colLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.4)
        table.auto_set_column_width(col=list(range(len(formatted.columns))))

        for col_idx in range(len(formatted.columns)):
            table[(0, col_idx)].set_text_props(weight='bold')
        for row_idx in range(1, len(formatted) + 1):
            table[(row_idx, 0)].set_text_props(ha='left')

        save_path = output_dir / "global_metrics_table.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved table: {save_path}")