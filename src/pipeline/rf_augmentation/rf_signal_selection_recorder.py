from pathlib import Path

import pandas as pd

from src.pipeline.rf_augmentation.io_utils import write_table


class RFSignalSelectionRecorder:
    MANUAL_FEATURE_TYPES = [
        "zero_crossings",
        "mean_abs",
        "coeff_var",
        "peak_count",
        "slope_mean",
        "kurtosis",
        "variance",
        "median",
        "energy",
        "range",
        "skew",
        "mean",
        "std",
        "min",
        "max",
        "iqr",
        "rms",
        "sum",
        "mad",
        "p10",
        "p25",
        "p75",
        "p90",
    ]

    @staticmethod
    def build_records(
        *,
        group_id: int,
        exp_ids: list,
        aligned_ids: list,
        feature_names_main: list,
        feature_names_secondary: list,
        selection_details: dict,
        synthetic_id_offset: int,
        synthetic_experiment_ids: list | None = None,
    ) -> list:
        records = []
        records.extend(
            RFSignalSelectionRecorder._build_model_records(
                group_id=group_id,
                exp_ids=exp_ids,
                aligned_ids=aligned_ids,
                feature_names=feature_names_main,
                selection_details=selection_details,
                model_input="main",
                synthetic_id_offset=synthetic_id_offset,
                synthetic_experiment_ids=synthetic_experiment_ids,
            )
        )
        records.extend(
            RFSignalSelectionRecorder._build_model_records(
                group_id=group_id,
                exp_ids=exp_ids,
                aligned_ids=aligned_ids,
                feature_names=feature_names_secondary,
                selection_details=selection_details,
                model_input="secondary",
                synthetic_id_offset=synthetic_id_offset,
                synthetic_experiment_ids=synthetic_experiment_ids,
            )
        )

        return records

    @staticmethod
    def save(records: list, output_path) -> Path:
        selection_values_df = pd.DataFrame(records)
        return write_table(selection_values_df, output_path, index=False)

    @staticmethod
    def _build_model_records(
        *,
        group_id: int,
        exp_ids: list,
        aligned_ids: list,
        feature_names: list,
        selection_details: dict,
        model_input: str,
        synthetic_id_offset: int,
        synthetic_experiment_ids: list | None = None,
    ) -> list:
        records = []

        actual_values = selection_details[f"X_{model_input}_actual"]
        paired_actual_values = selection_details.get(f"X_{model_input}_paired_actual")
        selected_values = selection_details[f"X_{model_input}_selected"]
        group_min = selection_details[f"X_{model_input}_group_min"]
        group_max = selection_details[f"X_{model_input}_group_max"]
        sampled_indices = selection_details["sampled_indices"]
        paired_indices = selection_details.get("paired_indices")
        paired_experiment_ids = selection_details.get("paired_experiment_ids")
        interpolation_weight = selection_details.get("interpolation_weight")
        selection_method = selection_details.get("selection_method", "")
        feature_sampling_mode = selection_details.get("feature_sampling_mode", "")
        alpha_min = selection_details.get("alpha_min", "")
        alpha_max = selection_details.get("alpha_max", "")
        sparse_reference_group_id = selection_details.get(
            "sparse_reference_group_id",
            "",
        )
        sparse_reference_experiment_ids = selection_details.get(
            "sparse_reference_experiment_ids",
            "",
        )
        sparse_reference_changed_feature = selection_details.get(
            "sparse_reference_changed_feature",
            "",
        )
        sparse_reference_target_value = selection_details.get(
            "sparse_reference_target_value",
            "",
        )
        sparse_reference_value = selection_details.get(
            "sparse_reference_value",
            "",
        )

        for sample_idx in range(selected_values.shape[0]):
            sampled_row_idx = int(sampled_indices[sample_idx])
            base_experiment_id = aligned_ids[sampled_row_idx]
            paired_experiment_id = ""

            if paired_experiment_ids is not None:
                paired_experiment_id = paired_experiment_ids[sample_idx]
            elif paired_indices is not None:
                paired_row_idx = int(paired_indices[sample_idx])
                paired_experiment_id = aligned_ids[paired_row_idx]

            weight = ""
            if interpolation_weight is not None:
                weight = interpolation_weight[sample_idx]

            synthetic_experiment_id = (
                synthetic_experiment_ids[sample_idx]
                if synthetic_experiment_ids is not None
                else synthetic_id_offset + sample_idx + 1
            )

            for feature_idx, feature_name in enumerate(feature_names):
                signal_name, feature_type, input_source = (
                    RFSignalSelectionRecorder._split_feature_name(feature_name)
                )

                records.append({
                    "group_id": group_id,
                    "source_experiment_ids": str(exp_ids),
                    "synthetic_experiment_id": synthetic_experiment_id,
                    "base_experiment_id": base_experiment_id,
                    "paired_base_experiment_id": paired_experiment_id,
                    "interpolation_weight": weight,
                    "model_input": model_input,
                    "input_source": input_source,
                    "signal_name": signal_name,
                    "feature_type": feature_type,
                    "feature_name": feature_name,
                    "actual_value": actual_values[sample_idx, feature_idx],
                    "paired_actual_value": (
                        paired_actual_values[sample_idx, feature_idx]
                        if paired_actual_values is not None
                        else ""
                    ),
                    "selected_value": selected_values[sample_idx, feature_idx],
                    "group_min_value": group_min[feature_idx],
                    "group_max_value": group_max[feature_idx],
                    "selection_method": selection_method,
                    "feature_sampling_mode": feature_sampling_mode,
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "sparse_reference_group_id": sparse_reference_group_id,
                    "sparse_reference_experiment_ids": sparse_reference_experiment_ids,
                    "sparse_reference_changed_feature": (
                        sparse_reference_changed_feature
                    ),
                    "sparse_reference_target_value": sparse_reference_target_value,
                    "sparse_reference_value": sparse_reference_value,
                })

        return records

    @staticmethod
    def _split_feature_name(feature_name: str):
        for feature_type in RFSignalSelectionRecorder.MANUAL_FEATURE_TYPES:
            suffix = f"_{feature_type}"
            if feature_name.endswith(suffix):
                return feature_name[: -len(suffix)], feature_type, "signal_feature"

        return "", feature_name, "bending_parameter"
