# TopoFusion-Res

**Multi-resolution topographic fusion and residual learning for regional climate emulation**

The code supporting the manuscript *TopoFusion-Res: Multi-resolution topographic fusion and residual learning for regional climate emulation*. This repository prioritizes the **actual author-written Python implementation** and a step-by-step guide to running the experiment. No scripts have been replaced with reimplementations.

[Español](README.es.md) · [Scientific source map](docs/SOURCE_MAP.md) · [Data access and rights](docs/DATA_ACCESS.md) · [Detailed reproducibility notes](docs/REPRODUCIBILITY.md)

> **Reproduction status:** the model/dataset source and seeded training/evaluation entry points have been recovered, and a synthetic forward test has been reported. **An independent full training/data preparation run has not been verified.** Raw climate files, pretrained checkpoints, and the U-Net output are not bundled. One small derived coarse-orography array (22×22) is included for preparation, subject to provenance and redistribution review. Please do not claim exact paper-number reproduction solely by running the synthetic test.

## 1. What is the model?

TopoFusion-Res is a **convolutional** GCM-to-RCM emulator, not SwinIR or a Swin Transformer despite some historical filenames. It combines a 22×22 GCM atmospheric branch with 64×64 physiographic features, reconstructs atmospheric features 22→44→64, and learns a second **bounded residual correction** with the base frozen.

| Item | Manuscript protocol |
|---|---|
| Source → target | CNRM-ESM2-1 → CNRM-ALADIN daily near-surface temperature |
| Dynamic inputs | 24 atmospheric channels, 22×22; plus coarse orography |
| Static / context | 6 fine-scale channels (64×64), 3 temporal and 4 forcing variables |
| Development | Historical 1950–2014, SSP5-8.5 2015–2100; train to 2070, validation 2071–2100 |
| Held-out evaluation | SSP3-7.0 2015–2100, 31,390 days, 1,852 common land cells |
| Final run | Seed `20260813`, first stage 40 epochs, residual stage 12 epochs |
| Ablations | With/without fine topography × with/without residual refinement |

Reported daily RMSE/MAE (°C): TopoFusion-Res 1.37 / 0.98; U-Net 1.54 / 1.11. These are **RCM-emulation** errors, not observed meteorological errors.

## 2. Repository organization

```text
TopoFusion-Res/
├── README.md, README.es.md
├── climate_swinir_experiments/model_E_cmip_swinunet_topography/
│   ├── configs/config_model_E.json          # author-original historical config
│   ├── models/                              # architecture and dataset classes
│   ├── scripts/                             # preparation, stage 1, stage 2, evaluation
│   └── paper_validation_final_seed20260813/scripts/
│                                           # statistics and paper figures
├── reference_results/                      # original CSV/JSON summary results
├── reference_figures/                      # supplementary/reference plots
├── tools/make_portable_copy.py             # path-only changes in an untracked copy
├── tools/check_run_inputs.py               # check missing data and checkpoints
├── docs/                                   # provenance and limitations
└── source_manifest.csv                     # SHA-256 of original sources
```

## 3. Install the software

The scripts were developed with PyTorch, NumPy, SciPy, pandas, xarray and Matplotlib. **Exact original environment versions are not included**; the commands below are a suggested environment, not an asserted reproduction of the author's CUDA stack. Use a Python/PyTorch combination supported by your hardware; Python 3.11 is an illustrative starting point.

```bash
# After cloning or extracting the ZIP:
cd TopoFusion-Res
conda create -n topofusion-res python=3.11 -y
conda activate topofusion-res
# Install a compatible PyTorch build for your GPU/CPU as documented by PyTorch,
# then install remaining Python packages:
python -m pip install numpy pandas scipy xarray matplotlib netCDF4 cftime

# No climate inputs needed for these initial checks:
python tools/author_source_integrity.py
python tools/check_reference_files.py
python tools/smoke_test_models.py
```

Check the actual original Conda/PyTorch versions if exact numerical replication is important; note nondeterminism may remain across hardware/software stacks.

## 4. Make a local runnable copy (original scripts stay unchanged)

The historical code uses `/workspace/SwinIR_HF/Normalizacion`. Instead of demanding that filesystem layout, prepare a **separate untracked working tree**:

```bash
python tools/make_portable_copy.py --dest run
```

This copies the original scientific source into `run/` and changes **only filesystem path constants and local JSON input paths**. It does not change the model layers, weights, objectives, random seeds, or training epochs. The original source remains intact in the Git repository, and `run/` is Git-ignored.

> Caution: the path-relocated working copy passes static Python syntax checks but has not been run through a complete climate experiment. Input datasets must follow the schemas expected by the original code. Do not substitute arbitrary NetCDF files.

## 5. Supply climate inputs (not hosted in this repository)

The preparation script expects the same CNRM-ESM2-1 / CNRM-ALADIN experiment, consistent calendars and grids, GHG forcing files, region land-mask and orography inputs. The small `cmip_orog_22x22_first_valid.npy` (22×22, float32) is now included in the original `diagnostics/` tree and is copied automatically into `run/` by the path-relocation tool; it was supplied by the authors, but its upstream generation procedure is not documented in the recovered source. Place the **authorized original files** using the paths in `run/climate_swinir_experiments/model_E_cmip_swinunet_topography/configs/config_model_E.json`.

After step 4, relative raw input paths are mapped to `run/inputs/raw/` using the original basenames. In addition, place:

```text
run/
├── inputs/raw/                         # GCM/RCM NetCDF, land mask, GHG files (see local JSON)
├── climate_swinir_analysis/data/
│   └── orography_regridded_64x64_cellmean_copernicus_dem.nc
├── climate_swinir_experiments/model_E_cmip_swinunet_topography/
│   ├── diagnostics/cmip_orog_22x22_first_valid.npy
│   └── data/torch_dataset_model_E_physical_v1/     # created by preparation script
└── data_tomas/unet_outputs/
    └── tas_cmip_cmip_output.nc             # needed for U-Net comparison
```

**Known limitation:** the included 22×22 orography array enables use of the authors' derived input, but the script and method that originally generated it have not been recovered; this is **not** a verified independent reconstruction from raw coarse-grid orography. Check redistribution rights before making this derived array public. The `training_ref_path` field is retained from the original configuration but is not used by this preparation entry point.

To check the mandatory input paths:

```bash
python tools/check_run_inputs.py --root run --mode prepare
```

## 6. Replicate the experiment, step by step

All commands are run **from the repository root** and refer to the **local working copy**. The four training scripts are those actually associated with final `seed20260813` experiment. The historical JSON fields `phase1_epochs=60` and `num_epochs=120` are **not** the final epoch settings; the seeded source scripts explicitly select **40 and 12**.

### A. Build the prepared NumPy datasets from raw climate data

```bash
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/01_prepare_dataset_model_E.py
```

This creates memory-mapped train/validation/test arrays and normalization products under `run/climate_swinir_experiments/model_E_cmip_swinunet_topography/data/torch_dataset_model_E_physical_v1/`. If you already have the author's prepared arrays, you may place those files there and **skip this step**, provided you have verified provenance and normalisation.

### B. Train the base models (40 epochs)

```bash
python tools/check_run_inputs.py --root run --mode train
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/30_train_phase1_full_seeded.py
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/31_train_phase1_no_fine_topo_seeded.py
```

Expected best-checkpoint locations, relative to `run/.../model_E_cmip_swinunet_topography/`:

```text
checkpoints/phase1_full_seed20260813_40epochs/model_E_phase1_best.pth
checkpoints/phase1_no_fine_topo_seed20260813_40epochs/model_E_phase1_best.pth
```

### C. Train the bounded residual heads (12 epochs)

```bash
python tools/check_run_inputs.py --root run --mode residual
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/32_train_e2_full_seeded.py
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/33_train_e2_no_fine_topo_seeded.py
```

Expected output:

```text
checkpoints/phase2_e2_full_seed20260813_12epochs/model_E2_v2_best.pth
checkpoints/phase2_e2_no_fine_topo_seed20260813_12epochs/model_E2_v2_best.pth
```

### D. Evaluate the held-out SSP3-7.0 experiment and U-Net benchmark

The evaluation uses **all four best checkpoints**, the preprocessed SSP3-7.0 arrays and the independent U-Net NetCDF. It does not retrain on SSP3-7.0.

```bash
python tools/check_run_inputs.py --root run --mode evaluate
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/34_evaluate_final_seeded_ssp370.py
```

The principal evaluation script writes to `run/climate_swinir_experiments/model_E_cmip_swinunet_topography/diagnostics/final_seeded_ssp370/`. Compare its metrics with the CSVs in `reference_results/` **before** claiming an exact replication.

## 7. Reproduce paper statistics and figures

The paper-validation scripts are preserved in `run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/` (original filenames). These implement paired circular moving-block bootstrap (3/5/10 years), daily anomaly persistence, trend/signal maps, and p01/p99 diagnostics.

**Important additional input:** many validators read precomputed `float32`/`bool` prediction cubes stored in:

```text
run/climate_swinir_experiments/model_E_cmip_swinunet_topography/
  diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes/
    cube_metadata.json, rcm.dat, e2_full_seeded.dat, unet.dat, mask.dat
```

Those cubes are **not** in GitHub. We identified the final seeded producer `34_evaluate_topography_ablation_ssp370.py` for `e2_full_seeded.dat` and `e2_no_fine_topo_seeded.dat`, and `36_create_phase1_full_seeded_cube.py` for `phase1_full_seeded.dat`. The former imports `build_band_masks` from recovered original `scripts/17_train_model_E2_residual_head_base_eval.py` (a historical dependency, **not** the final training script). **Both producers require existing** `rcm.dat`, `mask.dat`, and `cube_metadata.json`; they do not establish how the reference/U-Net cubes were first built. The older `cube_metadata.json` refers to an earlier `e2.dat` and should not be treated as proof of the final seeded E2 provenance. See [`docs/PREDICTION_CUBES.md`](docs/PREDICTION_CUBES.md), [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and [`docs/SOURCE_MAP.md`](docs/SOURCE_MAP.md).

Once the same final prediction cubes are available, the main bootstrap analysis is:

```bash
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/05_block_bootstrap_core.py
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/05b_block_bootstrap_3yr.py
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/05c_block_bootstrap_10yr.py
```

After providing the reference cubes and trained checkpoints, the recovered original additional-cube entry points are:

```bash
# Prerequisite: cube_metadata.json, rcm.dat, mask.dat and both final E2 checkpoints.
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/34_evaluate_topography_ablation_ssp370.py
# Prerequisite: same metadata/reference cubes and the final base checkpoint.
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/36_create_phase1_full_seeded_cube.py
```

These commands write large `.dat` cubes; they have **not** been independently executed against the real climate data in this review. See the exact prerequisites, metadata caveat and required file sizes in [`docs/PREDICTION_CUBES.md`](docs/PREDICTION_CUBES.md).

These commands are recovered original script entry points, **not yet an independently validated full paper-generation pipeline**. The `reference_results/` folder provides original unrounded numerical outputs for auditing the manuscript; the final paper reports suitably rounded values.

## 8. Citation, permissions, and scope

Use [`CITATION.cff`](CITATION.cff) to cite the authors/manuscript; publication metadata and DOI should be added after acceptance. Code redistribution and climate-data availability require approval by the relevant rights holders. No license is asserted in this archival package. Do not commit licensed raw climate datasets, trained weights, full prediction cubes or credentials.

**What this release supports now:** source inspection, model interface smoke tests, and a clearly documented training/evaluation workflow for authorized users with the exact input data. **What it does not yet guarantee:** exact third-party numerical reproduction from public downloads alone.
