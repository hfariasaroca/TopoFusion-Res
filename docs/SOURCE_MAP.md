# Model E + E2 source map — audited files

The structure is preserved verbatim to keep `__file__`-based relative paths valid. Every source file in `source_manifest.csv` has an original SHA-256 hash. No training checkpoint was loaded during this source review.

## End-to-end stages

| Stage | Script/module | Verified role |
|---|---|---|
| Prepare data | `scripts/01_prepare_dataset_model_E.py` | Read GCM, RCM, forcing, masks and topography; prepare `train/val/test` arrays |
| Dataset | `models/dataset_model_E.py` | Memory-mapped datasets with 24-channel atmospherics and 6-channel fine static tensors |
| Topography ablation | `models/dataset_model_E_ablation.py` | Remove fine elevation/gradient/roughness channels for controlled ablations |
| Base architecture | `models/model_e_cmip_swinunet_topography.py` | Two-branch convolutional network; 22→44→64 atmospheric reconstruction + 64×64 physiography |
| Residual architecture | `models/model_e2_residual_head.py` | Bounded ±2.5 °C residual head, base network frozen during training |
| Base full | `scripts/30_train_phase1_full_seeded.py` | 40 epochs; saved under `phase1_full_seed20260813_40epochs` |
| Base without topography | `scripts/31_train_phase1_no_fine_topo_seeded.py` | 40 epochs; no-topography ablation |
| E2 full | `scripts/32_train_e2_full_seeded.py` | 12 epochs; seeded phase-1 full initialization |
| E2 without topography | `scripts/33_train_e2_no_fine_topo_seeded.py` | 12 epochs; matching no-topography phase-1 initialization |
| Final evaluation | `scripts/34_evaluate_final_seeded_ssp370.py` | Four checkpoints + SSP3-7.0 + external U-Net NetCDF |
| Band-mask dependency | `scripts/17_train_model_E2_residual_head_base_eval.py` | Historical script imported **only** for `build_band_masks()` by the final topography-ablation cube generator |
| Final seeded E2 cubes | `paper_validation_final_seed20260813/scripts/34_evaluate_topography_ablation_ssp370.py` | Writes `e2_full_seeded.dat`, `e2_no_fine_topo_seeded.dat` from final E2 weights; requires existing reference cubes/metadata |
| Final seeded phase-1 cube | `paper_validation_final_seed20260813/scripts/36_create_phase1_full_seeded_cube.py` | Writes `phase1_full_seeded.dat`; requires existing reference cubes/metadata |
| Coarse-grid orography | `diagnostics/cmip_orog_22x22_first_valid.npy` | Author-supplied 22×22 float32 file used by dataset preparation; upstream generator unknown |

### Final model checkpoints (not included)

```text
checkpoints/phase1_full_seed20260813_40epochs/model_E_phase1_best.pth
checkpoints/phase1_no_fine_topo_seed20260813_40epochs/model_E_phase1_best.pth
checkpoints/phase2_e2_full_seed20260813_12epochs/model_E2_v2_best.pth
checkpoints/phase2_e2_no_fine_topo_seed20260813_12epochs/model_E2_v2_best.pth
```

## Paper validation

The selected original scripts are under `paper_validation_final_seed20260813/scripts/` with their original numeric filenames. They implement common-mask evaluation, persistence, 3/5/10-year circular moving block bootstrap, topography/refinement ablations, annual evolution, trends, climate-change signals, p01/p99 tails and official figure generation. See `docs/REPRODUCIBILITY.md` for inputs and limitations.

## Historical script explicitly excluded from final workflow

`historical_sources/24_final_advanced_diagnostics_e2_vs_unet.py` references `phase2_e2_residual_head_base_eval_12epochs`, **not** the final seeded phase-2 E2 checkpoint. It is archived to help trace old diagnostics but is not presented as a final experiment entry point.
