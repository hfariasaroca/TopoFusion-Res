# TopoFusion-Res

**Multi-resolution topographic fusion and residual learning for regional climate emulation**

TopoFusion-Res is a convolutional emulator of daily near-surface air temperature from **CNRM-ESM2-1** (22 × 22 GCM grid) to **CNRM-ALADIN** (64 × 64 RCM grid) over north-central Chile. It progressively reconstructs GCM features (22 → 44 → 64), fuses fine-scale physiographic information and applies a bounded residual correction in a second training stage. The architecture is **not a Swin Transformer**.

| Protocol | Setting |
|---|---|
| Inputs | 24 dynamic atmospheric channels, coarse orography, 6 fine-scale static channels, 3 temporal and 4 forcing features |
| Development | Historical + SSP5-8.5; training to 2070, validation 2071–2100 |
| Independent scenario | SSP3-7.0, 2015–2100 (31,390 daily fields) |
| Seed / epochs | 20260813 / 40 base + 12 residual |
| Daily RMSE against RCM | TopoFusion-Res 1.37 °C; U-Net 1.54 °C (reported manuscript results) |

## Repository

```text
models/          CNN base, E2 residual model, dataset loaders
scripts/         Data preparation; final seeded training and evaluation
analysis/scripts/ Statistics, bootstrap, ablations, publication figures
configs/         Experiment configuration
assets/          Small author-supplied 22 × 22 orography array
results/         Reference CSV/JSON summaries and selected figures
tools/           Integrity, input checks and synthetic model test
docs/            Data requirements, validation details and provenance
```

## Installation

```bash
conda create -n topofusion-res python=3.11 -y
conda activate topofusion-res
# Install PyTorch compatible with your platform; then:
python -m pip install numpy scipy pandas xarray matplotlib netCDF4 cftime
python tools/author_source_integrity.py
python tools/check_reference_files.py
python tools/smoke_test_models.py
```

The original package versions were not recorded in the recovered source; Python 3.11 is an installation example, **not a verified reproduction environment**. See [environment notes](docs/ENVIRONMENT.md).

## Data preparation

Obtain the original authorized GCM/RCM NetCDF inputs, forcing series, mask, fine-scale orography and U-Net comparison output. File names and expected locations are specified in [`configs/config_model_E.json`](configs/config_model_E.json). Put the raw climate inputs under `inputs/raw/`, fine-scale orography at `inputs/orography_regridded_64x64_cellmean_copernicus_dem.nc`, and the U-Net NetCDF at `inputs/unet/tas_cmip_cmip_output.nc`.

```bash
python tools/check_run_inputs.py --mode prepare
python scripts/01_prepare_dataset_model_E.py
```

Alternatively, place the equivalent author-prepared NumPy arrays under `data/torch_dataset_model_E_physical_v1/`. See [data requirements](docs/DATA.md).

## Train and evaluate

Run from the **repository root** (no `/workspace` directories or second working copy required):

```bash
# Stage 1: base architecture; full and no-fine-topography ablation (40 epochs)
python tools/check_run_inputs.py --mode train
MODEL_E_SEED=20260813 python scripts/30_train_phase1_full_seeded.py
MODEL_E_SEED=20260813 python scripts/31_train_phase1_no_fine_topo_seeded.py

# Stage 2: frozen base + bounded residual head (12 epochs)
python tools/check_run_inputs.py --mode residual
MODEL_E_SEED=20260813 python scripts/32_train_e2_full_seeded.py
MODEL_E_SEED=20260813 python scripts/33_train_e2_no_fine_topo_seeded.py

# Independent SSP3-7.0 evaluation, including the U-Net benchmark
python tools/check_run_inputs.py --mode evaluate
python scripts/34_evaluate_final_seeded_ssp370.py
```

The final seeded scripts, **not** the historical epoch defaults inside the configuration JSON, define the paper's training schedule. Checkpoints are saved under `checkpoints/` and evaluation results under `diagnostics/` (both ignored by Git).

## Reproduce paper analyses

The original final analysis scripts are under [`analysis/scripts/`](analysis/scripts/). These include moving-block bootstrap (3/5/10 years), persistence, climatic trends and signal, topographic ablations and tail fidelity. For example:

```bash
python analysis/scripts/05_block_bootstrap_core.py
python analysis/scripts/05b_block_bootstrap_3yr.py
python analysis/scripts/05c_block_bootstrap_10yr.py
```

**These analyses additionally require prediction cubes** (`rcm.dat`, `mask.dat`, `unet.dat`, `e2_full_seeded.dat`, metadata). The supplied final-seed scripts can regenerate the E2 and base cubes only after prerequisite reference cubes are available. See [reproducibility and cube instructions](docs/REPRODUCIBILITY.md). Paper summary tables are provided in [`results/tables/`](results/tables/).

## Citation and availability

Please cite the manuscript specified in [`CITATION.cff`](CITATION.cff). Source code, reference summaries and a small derived orography file are included; raw climate datasets, weights and complete prediction cubes are **not**. Redistribution rights for the derived orography and data-access conditions must be confirmed before a public release. No software license has yet been declared.

Español: [README.es.md](README.es.md) · [Data](docs/DATA.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Source provenance](docs/PROVENANCE.md)
