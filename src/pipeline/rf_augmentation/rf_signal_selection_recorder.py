import pandas as pd


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
            )
        )

        return records

    @staticmethod
    def save(records: list, output_path) -> pd.DataFrame:
        selection_values_df = pd.DataFrame(records)
        selection_values_df.to_csv(output_path, index=False)
        return selection_values_df

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
    ) -> list:
        records = []

        actual_values = selection_details[f"X_{model_input}_actual"]
        selected_values = selection_details[f"X_{model_input}_selected"]
        group_min = selection_details[f"X_{model_input}_group_min"]
        group_max = selection_details[f"X_{model_input}_group_max"]
        sampled_indices = selection_details["sampled_indices"]
        paired_indices = selection_details.get("paired_indices")
        interpolation_weight = selection_details.get("interpolation_weight")
        selection_method = selection_details.get("selection_method", "")

        for sample_idx in range(selected_values.shape[0]):
            sampled_row_idx = int(sampled_indices[sample_idx])
            base_experiment_id = aligned_ids[sampled_row_idx]
            paired_experiment_id = ""

            if paired_indices is not None:
                paired_row_idx = int(paired_indices[sample_idx])
                paired_experiment_id = aligned_ids[paired_row_idx]

            weight = ""
            if interpolation_weight is not None:
                weight = interpolation_weight[sample_idx]

            synthetic_experiment_id = synthetic_id_offset + sample_idx + 1

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
                    "selected_value": selected_values[sample_idx, feature_idx],
                    "group_min_value": group_min[feature_idx],
                    "group_max_value": group_max[feature_idx],
                    "selection_method": selection_method,
                })

        return records

    @staticmethod
    def _split_feature_name(feature_name: str):
        for feature_type in RFSignalSelectionRecorder.MANUAL_FEATURE_TYPES:
            suffix = f"_{feature_type}"
            if feature_name.endswith(suffix):
                return feature_name[: -len(suffix)], feature_type, "signal_feature"

        return "", feature_name, "bending_parameter"
