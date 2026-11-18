# Reproduction of final experiments and paper analyses

## 1. Scientific protocol

Stage 1 learns the convolutional base model for 40 epochs (full-topography and no-fine-topography variants); stage 2 freezes the base and trains a bounded residual correction for 12 epochs. The final seed is `20260813`. Training and validation use historical + SSP5-8.5 data with the 2070/2071 split. Held-out cross-scenario evaluation uses SSP3-7.0, 2015–2100 (31,390 daily fields and 1,852 common land cells). See README for the executable training commands.

The comparison against U-Net uses its original prediction output; it is not a controlled architectural ablation. Errors are against the regional ALADIN simulation, not station observations. Reported single-seed performance and moving-block bootstrap do not establish variability across retraining seeds.

## 2. Inputs needed for SSP3-7.0 evaluation

- Preprocessed test arrays produced by `scripts/01_prepare_dataset_model_E.py`.
- Four best checkpoints in `checkpoints/phase1_{full,no_fine_topo}_seed20260813_40epochs/` and `checkpoints/phase2_e2_{full,no_fine_topo}_seed20260813_12epochs/`.
- U-Net benchmark NetCDF in `inputs/unet/tas_cmip_cmip_output.nc`.

Execute `python tools/check_run_inputs.py --mode evaluate`, then `python scripts/34_evaluate_final_seeded_ssp370.py`. The evaluation writes under `diagnostics/final_seeded_ssp370/`.

## 3. Prediction cubes for paper statistics

Several final-analysis scripts require `diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes/` with:

| Filename | Meaning / expected type |
|---|---|
| `cube_metadata.json` | JSON, `shape=[31390,64,64]` |
| `rcm.dat` | RCM target, float32 |
| `mask.dat` | Common valid mask, bool |
| `unet.dat` | U-Net predictions, float32 |
| `e2_full_seeded.dat` | Final full TopoFusion-Res, float32 |
| `e2_no_fine_topo_seeded.dat` | Final topography ablation, float32 |
| `phase1_full_seeded.dat` | Final base model, float32 |

The authors confirmed these cubes exist on their server, but they are **not** bundled here. The author's `cube_metadata.json` was generated with an earlier `e2.dat`; it provides cube shape but not verified provenance for the final seeded predictions.

The final-seed cube writers recovered from the authors' source are:

```bash
# First supply original rcm.dat, mask.dat, unet.dat and metadata plus final checkpoints.
python analysis/scripts/34_evaluate_topography_ablation_ssp370.py
python analysis/scripts/36_create_phase1_full_seeded_cube.py
```

The first script imports `build_band_masks` from `scripts/17_train_model_E2_residual_head_base_eval.py`, a historical helper kept strictly for this import. Neither cube writer regenerates the prerequisite RCM, U-Net, mask cubes or original metadata. Hence **full figure generation from only public raw downloads has not been independently verified**.

With complete cubes present, run the original source statistical analyses, for example:

```bash
python analysis/scripts/05_block_bootstrap_core.py
python analysis/scripts/05b_block_bootstrap_3yr.py
python analysis/scripts/05c_block_bootstrap_10yr.py
python analysis/scripts/03_daily_persistence.py
python analysis/scripts/39_generate_final_temporal_evolution.py
```

Other original paper figure generators are in `analysis/scripts/`. The `results/` directory retains author-provided reference summaries and selected figures for checking outputs; these were not regenerated during this code reorganization.

## 4. Validation levels

1. **Static:** integrity and syntax checks (`tools/author_source_integrity.py`, `tools/preflight.py`).
2. **Model interface:** synthetic forward for E and E2 (`tools/smoke_test_models.py`), no climate data required.
3. **Numerical:** requires original data, weights, full output cubes and compatible compute environment. **Not executed in this audit**.

Results should be compared using original unrounded values; rounding in the manuscript is for presentation. Nothing in this release establishes independent exact numerical reproduction across arbitrary Python/PyTorch/GPU versions.
