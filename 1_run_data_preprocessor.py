# src/pipeline/augmentation/run_augmentation.py

from pathlib import Path
import pandas as pd

from src.pipeline.augmentation.augmentation_pipeline import GeometryAugmentationPipeline


if __name__ == "__main__":

    data_dir = Path("data")

    input_path = data_dir / "processed" / "geometry_dataset.csv"
    output_dir = data_dir / "augmented"

    geometry_df = pd.read_csv(input_path)

    GeometryAugmentationPipeline.run(
        geometry_df=geometry_df,
        expected_per_group=20,
        output_dir=output_dir,
    )

    print(f"Geometry augmentation finished. CSVs saved to {output_dir}.")