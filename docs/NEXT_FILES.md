# Remaining inputs to complete public reproducibility

The **core Python modules, config, imported `17` helper and coarse-orography `.npy` are now present**. There is no need to re-send them. Remaining gaps are external data/weights, the **original reference and U-Net prediction-cube creation**, historical cube metadata provenance, environment and code licensing.

1. Collect metadata describing the actual training Conda environment (Python, torch, CUDA, drivers, package versions).
2. Verify the four final checkpoint file metadata (filenames, SHA-256, sizes, optional published download location); do not upload weights or raw datasets to GitHub without rights review.
3. The final E2 cube-producing script is recovered, but we still need to trace the creation of `rcm.dat`, `unet.dat`, `mask.dat` and the historic `cube_metadata.json`; verify that final E2 cubes were generated with the final seeded weights using logs/checksums.
4. Prepare an author-approved data access statement with separate terms for GCM, RCM, DEM, U-Net predictions and derived artifacts.
5. Make and test a minimal **separate** patch replacing `/workspace` paths with portable paths, without altering model math, training losses, splits, masks or random seed.
6. Select a public code license after reviewing project ownership and any third-party source.
7. Run a subset of final calculations and compare against the archived numerical CSV files.

Suggested server-side read-only inventory for checkpoint provenance (does not transfer weights):

```bash
ROOT="$HOME/Documents/SwinIR/SwinIR/SwinIR_HF/Normalizacion/climate_swinir_experiments/model_E_cmip_swinunet_topography"
find "$ROOT/checkpoints" -path '*seed20260813*' -name '*best.pth' -printf '%P %s bytes\n'
find "$ROOT/diagnostics" -name 'cube_metadata.json' -o -name '*cube*metadata*.json'
```

Do not assume that the GitHub repository alone can reconstruct third-party climate simulations. Distinguish executable inference with published weights from independently retraining on restricted training data.
