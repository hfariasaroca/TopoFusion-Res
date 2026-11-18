#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT
MODELS = EXP / "models"

sys.path.insert(0, str(MODELS))

from dataset_model_E import create_dataloader
from topofusion_base import ModelECMIPSwinUNetTopography

CKPT = EXP / "checkpoints/phase1_full_seed20260813_40epochs/model_E_phase1_best.pth"

CUBE = (
    EXP
    / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
)

META_PATH = CUBE / "cube_metadata.json"
PHASE1_PATH = CUBE / "phase1_full_seeded.dat"
RCM_PATH = CUBE / "rcm.dat"
MASK_PATH = CUBE / "mask.dat"

OUT_META = (
    EXP
    / "analysis/diagnostics/36_phase1_full_seeded_cube_metadata.json"
)

OUT_META.parent.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("CREATE PHASE1 SSP370 PREDICTION CUBE")
print("=" * 100)

if not CKPT.exists():
    raise FileNotFoundError(CKPT)
if not META_PATH.exists():
    raise FileNotFoundError(META_PATH)
if not RCM_PATH.exists():
    raise FileNotFoundError(RCM_PATH)
if not MASK_PATH.exists():
    raise FileNotFoundError(MASK_PATH)

with open(META_PATH) as f:
    cube_meta = json.load(f)

shape = tuple(cube_meta["shape"])
N, H, W = shape

print("Expected shape :", shape)
print("Checkpoint     :", CKPT)
print("Output         :", PHASE1_PATH)

expected_bytes = N * H * W * np.dtype("float32").itemsize
print("Expected bytes :", expected_bytes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device         :", device)

loader = create_dataloader(
    split="test",
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print("Loader samples :", len(loader.dataset))
print("Loader batches :", len(loader))

if len(loader.dataset) != N:
    raise RuntimeError(
        f"Loader contains {len(loader.dataset)} samples but cube expects {N}"
    )

model = ModelECMIPSwinUNetTopography(
    dynamic_channels=24,
    coarse_static_channels=1,
    fine_static_channels=6,
    time_dim=3,
    ghg_dim=4,
    base_channels=64,
    dropout=0.05,
).to(device)

ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Checkpoint epoch   :", ckpt.get("epoch"))
print("Checkpoint metrics :", ckpt.get("metrics"))

# Existing reference cubes
rcm = np.memmap(
    RCM_PATH,
    dtype="float32",
    mode="r",
    shape=shape,
)

mask = np.memmap(
    MASK_PATH,
    dtype="bool",
    mode="r",
    shape=shape,
)

# New prediction cube
phase1 = np.memmap(
    PHASE1_PATH,
    dtype="float32",
    mode="w+",
    shape=shape,
)

offset = 0

n = 0
sum_sq = 0.0
sum_abs = 0.0
sum_diff = 0.0

target_rcm_max_abs_diff = 0.0
nonfinite_phase1_on_common = 0

with torch.no_grad():

    for batch_idx, batch in enumerate(loader, start=1):

        dynamic = batch["dynamic"].to(device, non_blocking=True)
        coarse_static = batch["coarse_static"].to(device, non_blocking=True)
        fine_static = batch["fine_static"].to(device, non_blocking=True)
        time_features = batch["time_features"].to(device, non_blocking=True)
        ghg_features = batch["ghg_features"].to(device, non_blocking=True)

        target = (
            batch["target"]
            .numpy()[:, 0]
            .astype(np.float32)
        )

        pred = model(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static,
            time_features=time_features,
            ghg_features=ghg_features,
        )

        pred = (
            pred.detach()
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        b = pred.shape[0]

        if offset + b > N:
            raise RuntimeError(
                f"Writing beyond cube: offset={offset}, batch={b}, N={N}"
            )

        phase1[offset:offset+b] = pred

        r = np.asarray(
            rcm[offset:offset+b],
            dtype=np.float32,
        )

        m = np.asarray(
            mask[offset:offset+b],
            dtype=bool,
        )

        # Verify that the test target used by the loader is exactly
        # aligned with the already frozen RCM reference cube.
        vr = (
            m
            & np.isfinite(target)
            & np.isfinite(r)
        )

        if np.any(vr):
            local_max = float(
                np.max(np.abs(target[vr] - r[vr]))
            )
            target_rcm_max_abs_diff = max(
                target_rcm_max_abs_diff,
                local_max,
            )

        bad = m & ~np.isfinite(pred)
        nonfinite_phase1_on_common += int(bad.sum())

        valid = (
            m
            & np.isfinite(pred)
            & np.isfinite(r)
        )

        d = (
            pred[valid].astype(np.float64)
            - r[valid].astype(np.float64)
        )

        n += d.size
        sum_sq += float(np.sum(d*d))
        sum_abs += float(np.sum(np.abs(d)))
        sum_diff += float(np.sum(d))

        offset += b

        if batch_idx % 100 == 0 or batch_idx == len(loader):

            rmse_now = (sum_sq / max(n, 1)) ** 0.5
            mae_now = sum_abs / max(n, 1)
            bias_now = sum_diff / max(n, 1)

            print(
                f"batch={batch_idx:05d}/{len(loader)} | "
                f"offset={offset}/{N} | "
                f"RMSE={rmse_now:.6f} | "
                f"MAE={mae_now:.6f} | "
                f"bias={bias_now:+.6f}"
            )

if offset != N:
    raise RuntimeError(
        f"Incomplete cube: wrote {offset} samples, expected {N}"
    )

phase1.flush()

rmse = (sum_sq / n) ** 0.5
mae = sum_abs / n
bias = sum_diff / n

actual_bytes = PHASE1_PATH.stat().st_size

print("\n" + "=" * 100)
print("FINAL VALIDATION")
print("=" * 100)

print("Samples written           :", offset)
print("Common valid observations :", n)
print("File bytes                :", actual_bytes)
print("Expected bytes            :", expected_bytes)
print("Target vs RCM max |diff|  :", target_rcm_max_abs_diff)
print("Nonfinite Phase1 on mask  :", nonfinite_phase1_on_common)

print("\nPhase1 metrics on frozen common mask")
print("RMSE :", rmse)
print("MAE  :", mae)
print("Bias :", bias)
print("|Bias|:", abs(bias))

expected = {
    "rmse": 1.4112495,
    "mae": 1.0396059,
    "bias": -0.2477355,
}

delta = {
    "rmse": rmse - expected["rmse"],
    "mae": mae - expected["mae"],
    "bias": bias - expected["bias"],
}

print("\nDifference vs previously reported aligned metrics")
for k, v in delta.items():
    print(f"{k:5s}: {v:+.10f}")

if actual_bytes != expected_bytes:
    raise RuntimeError(
        f"Wrong file size: {actual_bytes} != {expected_bytes}"
    )

metadata = {
    "file": str(PHASE1_PATH),
    "shape": list(shape),
    "dtype": "float32",
    "bytes": actual_bytes,
    "checkpoint": str(CKPT),
    "checkpoint_epoch": ckpt.get("epoch"),
    "n_samples": offset,
    "common_valid_observations": int(n),
    "target_rcm_max_abs_difference": target_rcm_max_abs_diff,
    "nonfinite_phase1_on_common_mask": nonfinite_phase1_on_common,
    "metrics_common_mask": {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
    },
    "expected_previous_metrics": expected,
    "difference_vs_previous": delta,
}

with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved:")
print(PHASE1_PATH)
print(OUT_META)
print("\nDONE")
