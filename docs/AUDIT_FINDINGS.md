# Audit findings — repository v4 (updated dependencies)

## Verified directly from both uploaded code archives

- 6 original workflow scripts plus 25 selected final validation scripts were retained from the prior audit.
- Recovered 4 original implementation/data modules and original `config_model_E.json` from `TopoFusionRes_core_missing.tar.gz`.
- Recovered historical script `24_final_advanced_diagnostics_e2_vs_unet.py` but separated it from the final paper workflow because it selects an older E2 checkpoint.
- The model classes are explicitly convolutional (`Conv2d`, residual blocks, bilinear upsampling and fusion). No transformer/attention operations were identified in those implementations despite a historical filename and stale JSON fields.
- The final seeded training scripts override older JSON epoch settings. Do not mistake config defaults for the final experiment.
- Original code files are unchanged and checksummed. A separate synthetic forward smoke test is included; running it checks only shapes and residual initialization.

## Newly resolved dependencies

- Recovered `scripts/17_train_model_E2_residual_head_base_eval.py`, imported by final topography-ablation evaluation for `build_band_masks`. This is a historic helper, not the final seeded training entry point.
- Recovered author-supplied `diagnostics/cmip_orog_22x22_first_valid.npy`: 22×22, float32, 484 finite cells. Its upstream generator remains unidentified.
- Confirmed the final E2 cube-producing script writes both `e2_full_seeded.dat` and `e2_no_fine_topo_seeded.dat`; it requires existing `rcm.dat`, `mask.dat` and `cube_metadata.json`.

## Remaining blockers

1. Historical `/workspace/SwinIR_HF/Normalizacion` paths across source files, even though CEAZA uses a different path today. These need mechanical, reviewable relocation.
2. Raw climate NetCDF, forcing, masks, topography inputs and their redistribution licenses.
3. Final checkpoints, environment lockfile and optionally original U-Net predictions.
4. Prediction memmaps (not bundled), prerequisite reference cube generation for `rcm.dat`, `unet.dat`, `mask.dat` and `cube_metadata.json`, and historical-to-seeded provenance reconciliation. The recovered script 24 points at an older checkpoint and cannot be presumed to have produced the final paper E2 cubes.
5. Numerical comparison of a newly executed inference/evaluation run against archived unrounded CSVs.
6. Project authors' code license choice, public-release review and journal metadata.

**Status: CORE CODE COMPLETE FOR STATIC REVIEW; NOT VERIFIED REPRODUCIBLE END TO END.**
