# Prediction cubes and the final seeded validation

This document distinguishes **code that produces final seeded predictions** from files created by earlier diagnostics. It is based on original script inspection and the author-provided server directory listing; it does **not** claim a complete independent rerun.

## Outputs expected by statistical validators

Under `diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes/`:

| File | Type/shape | Role | Status |
|---|---|---|---|
| `cube_metadata.json` | JSON, `shape=[31390,64,64]` | Loader dimensions | Original file is historical; refers to `e2.dat` |
| `rcm.dat` | float32, `[31390,64,64]` | RCM target | Upstream creation not recovered as final seeded code |
| `mask.dat` | bool, `[31390,64,64]` | Validity/common-mask source | Upstream creation not recovered as final seeded code |
| `unet.dat` | float32, `[31390,64,64]` | U-Net reference output | Upstream creation not recovered as final seeded code |
| `e2_full_seeded.dat` | float32, `[31390,64,64]` | Final TopoFusion-Res | Produced by final script `34_evaluate_topography_ablation_ssp370.py` |
| `e2_no_fine_topo_seeded.dat` | float32, `[31390,64,64]` | Ablation | Produced by same script `34` |
| `phase1_full_seeded.dat` | float32, `[31390,64,64]` | Base model | Produced by final script `36_create_phase1_full_seeded_cube.py` |

Expected size for one float32 cube: `31390 * 64 * 64 * 4 = 514,293,760` bytes (~490.5 MiB); `mask.dat` is one byte per sample/cell (~122.6 MiB).

**Important:** the author-provided `cube_metadata.json` contains historical `paths.e2` referring to `e2.dat` and names the model `Model E2 v2`; it does not independently prove the provenance of the final seeded E2 cubes. Do not substitute historical `e2.dat` for `e2_full_seeded.dat` when replicating the paper.

## Producing the final seeded E2 cubes

The preserved original code in:

`paper_validation_final_seed20260813/scripts/34_evaluate_topography_ablation_ssp370.py`

- reads **both final** E2 checkpoint paths with seed `20260813`;
- loads the test dataset, `rcm.dat`, `mask.dat` and `cube_metadata.json`;
- imports `build_band_masks()` from `scripts/17_train_model_E2_residual_head_base_eval.py` to reproduce the historical E2 training band definitions;
- writes `e2_full_seeded.dat` and `e2_no_fine_topo_seeded.dat` (`numpy.memmap`, mode `w+`).

**Caution:** script `17` is historical (its own checkpoint is not the final seeded E2). It is imported as a helper; this does **not** mean it should be run as the training stage. Importing it also creates its historical output directories in the local working tree. Its band-mask helper requires the regional 64×64 orography NetCDF and prepared training mask.

After supplying the prerequisites and executing `python tools/make_portable_copy.py --dest run` from a clean checkout, run from the repo root:

```bash
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/34_evaluate_topography_ablation_ssp370.py
```

## Producing the final seeded base cube

`paper_validation_final_seed20260813/scripts/36_create_phase1_full_seeded_cube.py` loads the final **base full** checkpoint and writes `phase1_full_seeded.dat`, given the reference cube files and metadata. Execute it only after the prerequisites exist:

```bash
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/paper_validation_final_seed20260813/scripts/36_create_phase1_full_seeded_cube.py
```

## What remains to be traced

The author provided an actual listing of these files on the CEAZA server (including historical `e2.dat` and `phase1.dat`), but **we do not yet have a verified final-seeded generation procedure for `rcm.dat`, `unet.dat`, `mask.dat` and the initial metadata**. The older script `historical_sources/24_final_advanced_diagnostics_e2_vs_unet.py` is not a substitute for the final seeded workflow because it uses a prior E2 checkpoint.

If a third party already has the author's reference/U-Net prediction cubes, the final E2 and base cube producers can be run with the four training checkpoints and prepared inputs. The original data products and their use rights remain separate from this public-code repository.
