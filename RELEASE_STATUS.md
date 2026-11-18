# Release status — code + executable README

This archive is a GitHub-ready **code/documentation snapshot**, not a published repository or a claim of independently verified numerical reproduction. Author-original files are preserved in their original directories and recorded with SHA-256 checksums in `source_manifest.csv`. The path-only working-copy helper in `tools/make_portable_copy.py` is explicitly distinct from the original source snapshot.

Recovered `17_train_model_E2_residual_head_base_eval.py` and the 22×22 coarse-orography `.npy` were added in v4. The historical `17` script is only a helper imported by the final ablation generator; the final training scripts remain `30`–`33`.

Known external requirements: licensed climate data and DEM derivatives; source/provenance of the now-included coarse GCM orography array; U-Net benchmark output for comparison; trained weights for inference-only workflows; original prediction cubes and provenance for all figures/tables. The full data-to-paper workflow has not been tested on an external system.

**Prior to public release:** approve a code license, audit rights and secrets, confirm data access instructions and determine how to cite the paper. There is no claim of public data access without approval.
