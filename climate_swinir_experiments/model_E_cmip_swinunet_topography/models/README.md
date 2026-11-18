# Original model and data modules

These four files were recovered directly from the source archive supplied by the authors and are preserved without modification.

- `model_e_cmip_swinunet_topography.py`: **convolutional** multi-resolution phase-1 Model E, despite its historical filename (no Swin Transformer).
- `model_e2_residual_head.py`: second-stage bounded residual correction with frozen phase-1 backbone.
- `dataset_model_E.py`: memory-mapped dataset for train/val/test.
- `dataset_model_E_ablation.py`: masks the three fine-topography channels for controlled ablations.

The original code contains legacy `/workspace/` absolute paths. Original file hashes are in `source_manifest.csv`. Check `docs/CONFIGURATION.md` before using it on another system.
