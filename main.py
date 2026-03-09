from pathlib import Path

from src.pipeline.preprocessing.preprocessor import STLGeometryPreprocessPipeline


if __name__ == "__main__":
    STLGeometryPreprocessPipeline.run(
        failed_experiment=[1, 48, 166],
        output_dir=Path("data") / "processed",
    )

    print("STL geometry preprocessing finished. CSVs saved to data/processed.")
