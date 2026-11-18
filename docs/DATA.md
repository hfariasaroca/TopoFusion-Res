# Input data and paths

## Data sources

To run the experiment, provide the *same* author-authorized CNRM-ESM2-1 GCM atmospheric fields, CNRM-ALADIN near-surface temperature, historical and SSP5-8.5 training targets, SSP3-7.0 test data, greenhouse-gas/solar forcing time series, the regional land mask and the 64 × 64 Copernicus-derived orography product. The independent U-Net benchmark requires its original NetCDF prediction file. These data are not distributed here; the exact retrieval and preparation permissions must be established by the rights holders.

## Directory conventions

The JSON [`../configs/config_model_E.json`](../configs/config_model_E.json) lists the expected raw-file names under `raw_data`. All file paths are **relative to the repository root**.

```text
inputs/
  raw/                                        # 8 referenced GCM, RCM, mask, forcing files
  orography_regridded_64x64_cellmean_copernicus_dem.nc
  unet/tas_cmip_cmip_output.nc               # comparison, not needed to train base
assets/cmip_orog_22x22_first_valid.npy       # supplied 22x22 float32 coarse orography
data/torch_dataset_model_E_physical_v1/     # created by preparation
checkpoints/                                 # created by training
```

The file `assets/cmip_orog_22x22_first_valid.npy` is a 22 × 22 float32 derived field provided by the authors. Its upstream generation script has not been recovered; **do not claim an independently reproducible orography derivation**. Redistribution of this derived field must be authorized before the repository becomes public.

The code's configuration includes a historical, unused `training_ref_path` and historical epoch fields; final training scripts explicitly fix the 40/12-epoch protocol. The JSON's `model.attention_*` fields are historical and do not imply a Transformer model.

Use `python tools/check_run_inputs.py --mode prepare` to validate required input paths before launching preparation. Precomputed NumPy input arrays can be used if the grids, calendars and normalisation match the author's original preprocessing.
