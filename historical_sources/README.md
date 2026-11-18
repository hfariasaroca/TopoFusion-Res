# Historical diagnostic, not the paper-final seeded workflow

`24_final_advanced_diagnostics_e2_vs_unet.py` was recovered from the same original code repository, but it explicitly references the older checkpoint `phase2_e2_residual_head_base_eval_12epochs/model_E2_v2_best.pth`. It **does not target** the final `phase2_e2_full_seed20260813_12epochs` checkpoint used in the manuscript. It is retained for provenance only. Do not run it as the final protocol or use its outputs for the paper without independent reconciliation.
