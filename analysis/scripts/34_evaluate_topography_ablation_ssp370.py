#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT
MODELS = EXP / "models"
sys.path.insert(0, str(MODELS))

from dataset_model_E import create_dataloader
from model_e2_residual_head import ModelE2ResidualHead

FULL_CKPT = (
    EXP
    / "checkpoints/phase2_e2_full_seed20260813_12epochs/model_E2_v2_best.pth"
)

NOTOPO_CKPT = (
    EXP
    / "checkpoints/phase2_e2_no_fine_topo_seed20260813_12epochs/model_E2_v2_best.pth"
)

MAIN_CUBES = (
    EXP
    / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
)

RCM_PATH = MAIN_CUBES / "rcm.dat"
PAPER_MASK_PATH = MAIN_CUBES / "mask.dat"
META_PATH = MAIN_CUBES / "cube_metadata.json"

FULL_CUBE_PATH = MAIN_CUBES / "e2_full_seeded.dat"
NOTOPO_CUBE_PATH = MAIN_CUBES / "e2_no_fine_topo_seeded.dat"

OUT = EXP / "analysis"
TAB = OUT / "tables"
DIA = OUT / "diagnostics"

TAB.mkdir(parents=True, exist_ok=True)
DIA.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Load exact same band-definition functions used by E2 training
# ----------------------------------------------------------------------

script17 = EXP / "scripts/17_train_model_E2_residual_head_base_eval.py"

spec = importlib.util.spec_from_file_location(
    "e2_original_training",
    script17,
)

mod17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod17)


def build_model(checkpoint, device):
    model = ModelE2ResidualHead(
        dynamic_channels=24,
        coarse_static_channels=1,
        fine_static_channels=6,
        time_dim=3,
        ghg_dim=4,
        base_channels=64,
        dropout=0.0,
        max_residual_delta=2.5,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


class Metrics:
    def __init__(self):
        self.n = 0
        self.sq = 0.0
        self.ab = 0.0
        self.diff = 0.0

    def update(self, pred, target, mask):
        x = pred[mask].astype(np.float64)
        y = target[mask].astype(np.float64)

        d = x - y

        self.n += d.size
        self.sq += float(np.sum(d*d))
        self.ab += float(np.sum(np.abs(d)))
        self.diff += float(np.sum(d))

    def result(self):
        return {
            "n_observations": int(self.n),
            "rmse": (self.sq / self.n) ** 0.5,
            "mae": self.ab / self.n,
            "bias": self.diff / self.n,
            "abs_bias": abs(self.diff / self.n),
        }


print("=" * 100)
print("TOPOGRAPHY ABLATION — SSP370")
print("=" * 100)

with open(META_PATH) as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape

print("shape      :", shape)
print("FULL       :", FULL_CKPT)
print("NO_TOPO    :", NOTOPO_CKPT)

if N % 365 != 0:
    raise RuntimeError("Expected 365-day calendar")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device     :", device)

loader = create_dataloader(
    split="test",
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

if len(loader.dataset) != N:
    raise RuntimeError(
        f"Loader={len(loader.dataset)} vs expected={N}"
    )

full_model, full_ckpt = build_model(FULL_CKPT, device)
notopo_model, notopo_ckpt = build_model(NOTOPO_CKPT, device)

print("\nFULL checkpoint epoch:", full_ckpt.get("epoch"))
print("FULL val metrics     :", full_ckpt.get("metrics"))

print("\nNO_TOPO checkpoint epoch:", notopo_ckpt.get("epoch"))
print("NO_TOPO val metrics     :", notopo_ckpt.get("metrics"))

# ----------------------------------------------------------------------
# Reference and exact paper common mask
# ----------------------------------------------------------------------

rcm = np.memmap(
    RCM_PATH,
    dtype="float32",
    mode="r",
    shape=shape,
)

paper_mask_cube = np.memmap(
    PAPER_MASK_PATH,
    dtype="bool",
    mode="r",
    shape=shape,
)

paper_common = np.all(paper_mask_cube, axis=0)

print("\nPaper common cells:", int(paper_common.sum()))

# ----------------------------------------------------------------------
# Exact topographic bands used in E2
# ----------------------------------------------------------------------

band_masks_torch, original_band_counts = mod17.build_band_masks(
    torch.device("cpu")
)

band_masks = {
    k: v.numpy()[0, 0].astype(bool)
    for k, v in band_masks_torch.items()
}

paper_band_masks = {
    k: v & paper_common
    for k, v in band_masks.items()
}

print("\nBand cells on paper common mask:")
for k, v in paper_band_masks.items():
    print(f"  {k:32s}: {int(v.sum())}")

# ----------------------------------------------------------------------
# Output cubes
# ----------------------------------------------------------------------

full_cube = np.memmap(
    FULL_CUBE_PATH,
    dtype="float32",
    mode="w+",
    shape=shape,
)

notopo_cube = np.memmap(
    NOTOPO_CUBE_PATH,
    dtype="float32",
    mode="w+",
    shape=shape,
)

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

global_full = Metrics()
global_notopo = Metrics()

band_full = {
    k: Metrics()
    for k in paper_band_masks
}

band_notopo = {
    k: Metrics()
    for k in paper_band_masks
}

# Per-cell sufficient statistics for true variance ratios
count = np.zeros((H, W), dtype=np.float64)

sum_t = np.zeros((H, W), dtype=np.float64)
sum_t2 = np.zeros((H, W), dtype=np.float64)

sum_f = np.zeros((H, W), dtype=np.float64)
sum_f2 = np.zeros((H, W), dtype=np.float64)

sum_n = np.zeros((H, W), dtype=np.float64)
sum_n2 = np.zeros((H, W), dtype=np.float64)

cell_sq_f = np.zeros((H, W), dtype=np.float64)
cell_sq_n = np.zeros((H, W), dtype=np.float64)

offset = 0
target_rcm_max_abs_diff = 0.0

with torch.no_grad():

    for batch_idx, batch in enumerate(loader, start=1):

        b = batch["target"].shape[0]

        dynamic = batch["dynamic"].to(device, non_blocking=True)
        coarse = batch["coarse_static"].to(device, non_blocking=True)

        fine_full = batch["fine_static"].to(
            device,
            non_blocking=True
        )

        # Exact ablation:
        # 0 = fine orography
        # 4 = local orographic gradient magnitude
        # 5 = roughness
        fine_notopo = fine_full.clone()
        fine_notopo[:, [0, 4, 5], :, :] = 0.0

        time_features = batch["time_features"].to(
            device,
            non_blocking=True
        )

        ghg = batch["ghg_features"].to(
            device,
            non_blocking=True
        )

        target = (
            batch["target"]
            .numpy()[:, 0]
            .astype(np.float32)
        )

        base_mask = (
            batch["mask"]
            .numpy()[:, 0]
            .astype(bool)
        )

        out_full = full_model(
            dynamic=dynamic,
            coarse_static=coarse,
            fine_static=fine_full,
            time_features=time_features,
            ghg_features=ghg,
            return_parts=True,
        )

        out_notopo = notopo_model(
            dynamic=dynamic,
            coarse_static=coarse,
            fine_static=fine_notopo,
            time_features=time_features,
            ghg_features=ghg,
            return_parts=True,
        )

        pred_full = (
            out_full["tas_final"]
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        pred_notopo = (
            out_notopo["tas_final"]
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        full_cube[offset:offset+b] = pred_full
        notopo_cube[offset:offset+b] = pred_notopo

        ref = np.asarray(
            rcm[offset:offset+b],
            dtype=np.float32,
        )

        refmask = np.asarray(
            paper_mask_cube[offset:offset+b],
            dtype=bool,
        )

        vr = (
            refmask
            & np.isfinite(target)
            & np.isfinite(ref)
        )

        if np.any(vr):
            target_rcm_max_abs_diff = max(
                target_rcm_max_abs_diff,
                float(
                    np.max(
                        np.abs(target[vr] - ref[vr])
                    )
                ),
            )

        valid = (
            base_mask
            & paper_common[None, :, :]
            & np.isfinite(target)
            & np.isfinite(pred_full)
            & np.isfinite(pred_notopo)
        )

        global_full.update(
            pred_full,
            target,
            valid,
        )

        global_notopo.update(
            pred_notopo,
            target,
            valid,
        )

        for name, bm in paper_band_masks.items():

            m = valid & bm[None, :, :]

            band_full[name].update(
                pred_full,
                target,
                m,
            )

            band_notopo[name].update(
                pred_notopo,
                target,
                m,
            )

        # Per-cell statistics
        vf = valid.astype(np.float64)

        tf = target.astype(np.float64)
        pf = pred_full.astype(np.float64)
        pn = pred_notopo.astype(np.float64)

        count += vf.sum(axis=0)

        sum_t += (tf * vf).sum(axis=0)
        sum_t2 += ((tf*tf) * vf).sum(axis=0)

        sum_f += (pf * vf).sum(axis=0)
        sum_f2 += ((pf*pf) * vf).sum(axis=0)

        sum_n += (pn * vf).sum(axis=0)
        sum_n2 += ((pn*pn) * vf).sum(axis=0)

        cell_sq_f += (((pf-tf)**2) * vf).sum(axis=0)
        cell_sq_n += (((pn-tf)**2) * vf).sum(axis=0)

        offset += b

        if batch_idx % 100 == 0 or batch_idx == len(loader):

            mf = global_full.result()
            mn = global_notopo.result()

            print(
                f"batch={batch_idx:05d}/{len(loader)} "
                f"offset={offset}/{N} | "
                f"FULL rmse={mf['rmse']:.4f} "
                f"NO_TOPO rmse={mn['rmse']:.4f}"
            )


if offset != N:
    raise RuntimeError(
        f"Incomplete evaluation: {offset} != {N}"
    )

full_cube.flush()
notopo_cube.flush()

# ----------------------------------------------------------------------
# Global
# ----------------------------------------------------------------------

gf = global_full.result()
gn = global_notopo.result()

global_rows = [
    {
        "model": "E2_FULL_SEEDED",
        **gf,
    },
    {
        "model": "E2_NO_FINE_TOPO_SEEDED",
        **gn,
    },
]

global_df = pd.DataFrame(global_rows)

comparison = {
    "rmse_full_minus_notopo":
        gf["rmse"] - gn["rmse"],

    "rmse_improvement_full_pct":
        100.0 * (gn["rmse"] - gf["rmse"]) / gn["rmse"],

    "mae_full_minus_notopo":
        gf["mae"] - gn["mae"],

    "mae_improvement_full_pct":
        100.0 * (gn["mae"] - gf["mae"]) / gn["mae"],

    "abs_bias_full_minus_notopo":
        gf["abs_bias"] - gn["abs_bias"],
}

# ----------------------------------------------------------------------
# True variance ratio + cell RMSE
# ----------------------------------------------------------------------

valid_cells = paper_common & (count > 0)

mean_t = np.zeros_like(sum_t)
mean_f = np.zeros_like(sum_f)
mean_n = np.zeros_like(sum_n)

mean_t[valid_cells] = sum_t[valid_cells] / count[valid_cells]
mean_f[valid_cells] = sum_f[valid_cells] / count[valid_cells]
mean_n[valid_cells] = sum_n[valid_cells] / count[valid_cells]

var_t = np.full((H, W), np.nan)
var_f = np.full((H, W), np.nan)
var_n = np.full((H, W), np.nan)

var_t[valid_cells] = (
    sum_t2[valid_cells] / count[valid_cells]
    - mean_t[valid_cells]**2
)

var_f[valid_cells] = (
    sum_f2[valid_cells] / count[valid_cells]
    - mean_f[valid_cells]**2
)

var_n[valid_cells] = (
    sum_n2[valid_cells] / count[valid_cells]
    - mean_n[valid_cells]**2
)

ratio_f = 100.0 * var_f / var_t
ratio_n = 100.0 * var_n / var_t

rmse_cell_f = np.full((H, W), np.nan)
rmse_cell_n = np.full((H, W), np.nan)

rmse_cell_f[valid_cells] = np.sqrt(
    cell_sq_f[valid_cells] / count[valid_cells]
)

rmse_cell_n[valid_cells] = np.sqrt(
    cell_sq_n[valid_cells] / count[valid_cells]
)

# ----------------------------------------------------------------------
# Band table
# ----------------------------------------------------------------------

band_rows = []

for name, bm in paper_band_masks.items():

    mf = band_full[name].result()
    mn = band_notopo[name].result()

    cells = bm & valid_cells

    vf = float(np.nanmean(ratio_f[cells]))
    vn = float(np.nanmean(ratio_n[cells]))

    crf = float(np.nanmean(rmse_cell_f[cells]))
    crn = float(np.nanmean(rmse_cell_n[cells]))

    band_rows.append({
        "band": name,
        "n_cells": int(cells.sum()),

        "FULL_rmse": mf["rmse"],
        "NO_TOPO_rmse": mn["rmse"],
        "FULL_advantage_rmse":
            mn["rmse"] - mf["rmse"],
        "FULL_improvement_rmse_pct":
            100.0 * (mn["rmse"] - mf["rmse"]) / mn["rmse"],

        "FULL_mae": mf["mae"],
        "NO_TOPO_mae": mn["mae"],

        "FULL_bias": mf["bias"],
        "NO_TOPO_bias": mn["bias"],

        "FULL_cell_rmse_mean": crf,
        "NO_TOPO_cell_rmse_mean": crn,

        "FULL_variance_ratio_mean_pct": vf,
        "NO_TOPO_variance_ratio_mean_pct": vn,

        "FULL_variance_distance_from_100":
            abs(vf - 100.0),

        "NO_TOPO_variance_distance_from_100":
            abs(vn - 100.0),
    })

band_df = pd.DataFrame(band_rows)

# Global true variance ratio
comparison["FULL_variance_ratio_mean_pct"] = float(
    np.nanmean(ratio_f[valid_cells])
)

comparison["NO_TOPO_variance_ratio_mean_pct"] = float(
    np.nanmean(ratio_n[valid_cells])
)

comparison["paper_common_cells"] = int(
    valid_cells.sum()
)

comparison["target_rcm_max_abs_difference"] = (
    target_rcm_max_abs_diff
)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

global_path = TAB / "34_topography_ablation_ssp370_global.csv"
band_path = TAB / "34_topography_ablation_ssp370_by_band.csv"
comp_path = DIA / "34_topography_ablation_ssp370_comparison.json"

global_df.to_csv(global_path, index=False)
band_df.to_csv(band_path, index=False)

with open(comp_path, "w") as f:
    json.dump(comparison, f, indent=2)

metadata = {
    "period": "SSP370 2015-2100",
    "n_samples": N,
    "shape": list(shape),
    "paper_common_cells": int(valid_cells.sum()),
    "fine_topography_channels_removed": [
        0,
        4,
        5,
    ],
    "removed_features": [
        "fine_orography_64",
        "local_orographic_gradient_magnitude_64",
        "roughness_64",
    ],
    "retained_fine_features": [
        "sftlf_fraction",
        "latitude",
        "longitude",
    ],
    "retained_coarse_topography":
        "CMIP orography 22x22",
    "FULL_checkpoint": str(FULL_CKPT),
    "NO_TOPO_checkpoint": str(NOTOPO_CKPT),
    "FULL_cube": str(FULL_CUBE_PATH),
    "NO_TOPO_cube": str(NOTOPO_CUBE_PATH),
}

with open(
    DIA / "34_topography_ablation_ssp370_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 100)
print("GLOBAL SSP370")
print("=" * 100)
print(global_df.to_string(index=False))

print("\nCOMPARISON")
print("=" * 100)
print(json.dumps(comparison, indent=2))

print("\nBY TOPOGRAPHIC BAND")
print("=" * 100)
print(band_df.to_string(index=False))

print("\nSaved:")
print(global_path)
print(band_path)
print(comp_path)
print(FULL_CUBE_PATH)
print(NOTOPO_CUBE_PATH)
