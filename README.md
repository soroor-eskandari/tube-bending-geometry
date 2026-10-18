# tube-bending-geometry

Master thesis code for **interval prediction of rotary tube bending geometry**.
The pipeline preprocesses the experimental tube bending data, augments it with
Random Forest–based synthetic samples, and trains two uncertainty-aware models
that predict the main- and secondary-axis geometry with prediction intervals:

- **QRF** – Quantile Random Forest
- **HGP** – Heteroscedastic Gaussian Process

Results are explored in the online Streamlit dashboard:
**<https://tubebendgeometryintervalpredictionvisualisation-bkgn5b7s8vakc9.streamlit.app>**

## Data

The raw experiment data comes from the public tube bending dataset
[zeyneddinoz/tubebend](https://github.com/zeyneddinoz/tubebend). It is **not**
stored in this repository: `1_run_data_preprocessor.py` downloads it (~800 MB)
into `data/raw/` on first run and verifies its checksum. Please cite that
dataset if you use it.

## Stored thesis results

The split search, hyperparameter tuning and RF augmentation were run once
during the thesis. Their results are committed, so the final runners reuse
them instead of recomputing them:

| File | Used by |
|---|---|
| `src/pipeline/ml/qrf/data/various_splits.parquet` | QRF (best train/test splits) |
| `src/pipeline/ml/qrf/data/qrf_hyperparameter_tuning_results.parquet` | QRF (tuned hyperparameters) |
| `src/pipeline/ml/common/data/various_splits.parquet` | HGP (same splits as QRF) |
| `src/pipeline/ml/hgp/data/hgp_hyperparameter_tuning_results.parquet` | HGP (tuned hyperparameters) |
| `src/pipeline/rf_augmentation/result/*.csv` | RF feature ranking and selected feature subsets |
| `data/rf_augmented/final_geometry_*.parquet` | QRF/HGP (augmented geometry) |

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/soroor-eskandari/tube-bending-geometry.git
cd tube-bending-geometry
uv sync
```

## Reproducing the thesis results

Run from the repository root:

```bash
uv run python 1_run_data_preprocessor.py       # download raw data, clean → data/processed/
uv run python 3_1_run_qrf_interval_predictor.py # final QRF models → src/pipeline/ml/qrf/results/models/
uv run python 3_2_run_hgp_interval_predictor.py # final HGP models → src/pipeline/ml/hgp/results/models/
```

## Running the full pipeline from scratch

This also reruns the RF feature selection and augmentation (steps 2_1_x).
It **overwrites** the committed RF and augmented files, and with current
library versions the subset search selects slightly different features, so
augmented results will differ from the thesis. Restore the thesis files with
`git checkout -- src/pipeline/rf_augmentation/result data/rf_augmented`.

```bash
uv run python 1_run_data_preprocessor.py                   # download raw data, clean → data/processed/
uv run python 2_1_1_run_data_rf_feature_ranker.py          # rank sensor/process features
uv run python 2_1_2_run_data_rf_bending_feature_ranker.py  # rank bending setup features
uv run python 2_1_3_run_data_rf_feature_subset_search.py   # greedy feature subset search
uv run python 2_1_4_run_data_rf_augmentator.py             # train RF augmentation model
uv run python 2_1_5_run_data_augmented_generator.py        # generate augmented geometry → data/rf_augmented/
uv run python 3_1_run_qrf_interval_predictor.py            # final QRF models → src/pipeline/ml/qrf/results/models/
uv run python 3_2_run_hgp_interval_predictor.py            # final HGP models → src/pipeline/ml/hgp/results/models/
```

Each step depends on the outputs of the previous ones. Random seeds are fixed
throughout the pipeline.

Optional sensitivity analyses (slow, many model fits):

```bash
uv run python run_qrf_experiments_per_group.py
uv run python run_hgp_experiments_per_group.py
```

See [`PROJECT_RUN_GUIDE.txt`](PROJECT_RUN_GUIDE.txt) for what each step does,
its inputs and outputs, and troubleshooting.

## Repository layout

```
1_…, 2_1_…, 3_…               numbered pipeline runners
src/pipeline/preprocessing/   raw data extraction, cleaning, export
src/pipeline/rf_augmentation/ RF feature ranking, augmentation and data generation
src/pipeline/ml/qrf/          QRF training, evaluation, configs, stored results
src/pipeline/ml/hgp/          HGP training, evaluation, configs, stored results
src/pipeline/ml/common/       shared split utilities and stored splits
data/                         generated data (not versioned, except UI exports)
tests/                        pytest tests
```

Run the tests with `uv run pytest`.
