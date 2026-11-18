# Reproducibility: independently verifiable levels

## Level 0 — source provenance

Available now: original SHA-256-identified model, dataset, preparation, training and evaluation files, plus selected paper-validation scripts. Verify with `python tools/author_source_integrity.py` and `python tools/check_reference_files.py`.

## Level 1 — synthetic model forward

Available if PyTorch is installed: `python tools/smoke_test_models.py`. This tests 24-channel GCM inputs, coarse/fine static features, output shape and zero residual at initialization. It does **not** validate climate accuracy or checkpoint compatibility.

## Level 2 — reproduce model predictions (pending)

Required: authorized climate datasets, Copernicus DEM processing products, masks, exact preprocessing and 4 weights files, environment metadata, portable paths. A 22×22 float32 coarse-orography source array (supplied by the author) is now included, but its upstream construction has not been reconstructed. The evaluation's SSP3-7.0 U-Net comparison also requires the external U-Net NetCDF output. Full checkpoints are expected under the specific directories in `docs/SOURCE_MAP.md`.

## Level 3 — reproduce quantitative paper figures (pending)

Many scripts use `diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes/{rcm.dat,e2_full_seeded.dat,unet.dat,mask.dat,cube_metadata.json}`. These files were not provided and are intentionally not committed to GitHub. The recovered historical `24_final_advanced_diagnostics_e2_vs_unet.py` points to an older E2 checkpoint. We must **not** attribute the final seeded cubes to that script without checking cube metadata. We recovered the final seeded E2 cube producer (`34_evaluate_topography_ablation_ssp370.py`), the phase-1 cube producer (`36_create_phase1_full_seeded_cube.py`), and the imported legacy `build_band_masks` dependency (`17_train_model_E2_residual_head_base_eval.py`). The production of the prerequisite `rcm.dat`, `mask.dat`, `unet.dat` and original `cube_metadata.json` is still not fully traced. The on-server `cube_metadata.json` describes historical `e2.dat`, not the final seeded E2 products. See `docs/PREDICTION_CUBES.md`.

The bundled `reference_results/` unrounded CSVs and figures constitute author-provided reference outputs; they are **not** newly recomputed on this platform. The paper's rounded display values do not change decisions computed on original bootstrap intervals.

## Scientific comparison limits

- The U-Net baseline is a system-level benchmark; optimizer/architecture changes are not a pure controlled ablation.
- The evaluation estimates fidelity relative to CNRM-ALADIN, not observational truth.
- Single-seed experiments do not measure training-seed variability; moving-block bootstrap is sensitivity to temporal resampling.
- Persistence, trend and climate signal metrics can favor U-Net even though daily RMSE, MAE and tail fidelity favor TopoFusion-Res.
