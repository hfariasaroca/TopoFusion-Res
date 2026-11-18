# Configuration and file paths (verified from source)

## Important distinction: original JSON vs. final executed hyperparameters

`configs/config_model_E.json` is included byte-for-byte from the server. **It predates / does not fully reflect the final paper settings:**

| Value | Original JSON | Final seeded scripts |
|---|---|---|
| First-stage training epochs | `training.phase1_epochs = 60` (and `num_epochs = 120`) | `30_...` and `31_...` set `num_epochs = 40` |
| Second-stage epochs | Not represented as the final 12-epoch run | `32_...` / `33_...` set `num_epochs = 12` |
| Seed | Not authoritative in this config | `MODEL_E_SEED` defaults to `20260813` |
| Coarse input channels | `dynamic_channels = null` | `ModelEDataset` validates `(24,22,22)`; model receives 24 + one coarse orography |
| Auxiliary config fields | `attention_heads`, `attention_window_size` | Historical fields; **no attention layer** in the inspected convolutional model source |

**Never cite the JSON alone to describe the training configuration.** Final hyperparameters are in the actual training scripts and should be checked against checkpoint metadata when the weights are available.

## Layout constraints

Original root: `/workspace/SwinIR_HF/Normalizacion`. The CEAZA server currently uses a different directory. Many scripts construct paths relative to that historical root, while `34_evaluate_final_seeded_ssp370.py` uses `__file__` to derive its root. We have **not** changed the original files in this repository, so do not run scripts directly from a new checkout until their path constants are adapted and compared with the source manifest.

The original JSON expects the following raw input categories under `raw_data`: SSP5-8.5 GCM predictors; historical/SSP5-8.5 RCM targets; a reference GCM file; forcing time series for SSP5-8.5 and SSP3-7.0; SSP3-7.0 GCM predictors and RCM target; regional land mask. It also references high-resolution orography, a regional mask and coarse-orography `.npy` output from an upstream preprocessing step. The 22×22 coarse-orography `.npy` is now included as a small author-supplied derived input. **The other climate and geographic inputs are not distributed** here. The original script that generated the coarse-orography file is not recovered; verify provenance and rights before public release.

Before attempting preparation, make a **local untracked configuration**, update data paths there, and ensure the selected project uses it. Because the original code hardcodes a filename `configs/config_model_E.json`, any user-specific edits within a Git clone must be treated as local modifications, not as changes to the authoritative author-source snapshot.

Run `python tools/inspect_external_inputs.py` to list missing external files and final checkpoints. This is read-only.

## Dataset interface expected by the code

Each sample from `models/dataset_model_E.py` has:

```text
dynamic       [24,22,22]
coarse_static [1,22,22]
fine_static   [6,64,64]
time_features [3]
ghg_features  [4]
target        [1,64,64]
mask          [1,64,64]
```

The data preparation code makes memory-mapped NumPy files. `ModelEDataset` validates the input shapes and accesses the training/validation/test arrays from `cfg["dataset_path"]`. Document original normalization and land mask before making alternative datasets.

## Security and distribution

The original config exposes legacy server file paths, not credentials. We found no obvious tokens in these six recovered files, but this is not a comprehensive security review. Verify redistribution rights for all model inputs and license third-party code before making the repository public.
