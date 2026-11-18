# Configuration (author original)

`config_model_E.json` is an original historical configuration snapshot: it has **not** been normalized or changed. Its `training` section contains legacy/default values of `phase1_epochs: 60` and `num_epochs: 120`, not the final published experiment. The final seeded training scripts explicitly set **40 epochs (phase 1)** and **12 epochs (phase 2)**. These script-local settings take precedence in the final experiments. Do not describe this JSON as the definitive training hyperparameter record without comparing to the final scripts/checkpoints.

`raw_data.*` points at historical `/workspace/data_tomas/...` files. These climate datasets are not bundled; change the paths in a local copy to sources you are licensed to access. Never commit private data or paths containing credentials.

Read `docs/CONFIGURATION.md`.
