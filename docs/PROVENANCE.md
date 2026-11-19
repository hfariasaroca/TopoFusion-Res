# Source and result provenance

Of 39 author-provided scientific files in version 4, 38 are retained in this reduced hierarchy; one unrelated historical diagnostics entry point was omitted. The algorithmic implementation and experiment logic were retained. Changes in version 5 are **directory paths, module import names, configuration paths and path-related comments only**; the original model class names, state-dict keys, losses, epochs and learning logic were not intentionally changed.

- `models/topofusion_base.py` corresponds to the original file `models/model_e_cmip_swinunet_topography.py`. Some internal Python class names retain historical labels for checkpoint/script compatibility. This does not indicate a transformer implementation.
- `models/model_e2_residual_head.py` retains the residual architecture.
- `scripts/` includes training entry points and `17_train_model_E2_residual_head_base_eval.py` as a **legacy imported dependency only**.
- `analysis/scripts/` preserves the selected final paper scripts from `paper_validation_final_seed20260813/scripts/`.
- `source_manifest.csv` contains both the v5 source hashes and the original v4 source hashes and paths, allowing exact provenance comparison. Verify current content with `python tools/author_source_integrity.py`.
- `docs/config_model_E_original.json` preserves the original config, including old directory roots. The current `configs/config_model_E.json` contains portable paths; it should not be misrepresented as the exact archived configuration.
- `results/FILE_MANIFEST.csv` tracks the 19 original result artifacts with identical hashes despite relocation. These artifacts are author-supplied, **not recomputed** here.

The script changes were checked for Python syntax; the model-forward test runs without real data. No complete training or numerical evaluation has been independently executed after this reorganization. Users should compare runs against the paper and the reference outputs.

The source and result distribution requires appropriate author and institutional permissions. The repository intentionally contains no checkpoints or raw climate datasets. The source code is distributed under the BSD-3-Clause license. Upstream data rights and scientific provenance are documented separately.
