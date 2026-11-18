#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


# ============================================================
# Paths relative to the repository root
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
EXP_DIR = SCRIPT_PATH.parents[1]
ROOT = SCRIPT_PATH.parents[1]

MODELS_DIR = EXP_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))

from dataset_model_E import create_dataloader
from topofusion_base import ModelECMIPSwinUNetTopography
from model_e2_residual_head import ModelE2ResidualHead


# ============================================================
# Checkpoints FINALES seeded
# ============================================================

PHASE1_FULL_CKPT = (
    EXP_DIR
    / "checkpoints"
    / "phase1_full_seed20260813_40epochs"
    / "model_E_phase1_best.pth"
)

E2_FULL_CKPT = (
    EXP_DIR
    / "checkpoints"
    / "phase2_e2_full_seed20260813_12epochs"
    / "model_E2_v2_best.pth"
)

PHASE1_NO_TOPO_CKPT = (
    EXP_DIR
    / "checkpoints"
    / "phase1_no_fine_topo_seed20260813_40epochs"
    / "model_E_phase1_best.pth"
)

E2_NO_TOPO_CKPT = (
    EXP_DIR
    / "checkpoints"
    / "phase2_e2_no_fine_topo_seed20260813_12epochs"
    / "model_E2_v2_best.pth"
)

UNET_PATH = ROOT / "inputs/unet/tas_cmip_cmip_output.nc"

OUT_DIR = EXP_DIR / "diagnostics" / "final_seeded_ssp370"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FINE_TOPO_CHANNELS = (0, 4, 5)


# ============================================================
# Helpers
# ============================================================

def find_tas_var(ds: xr.Dataset) -> str:
    for c in ["tas", "tas_pred", "prediction", "output", "y_pred"]:
        if c in ds.data_vars:
            return c

    for v in ds.data_vars:
        if ds[v].ndim >= 3:
            return v

    raise RuntimeError(
        f"No encontré variable tipo tas/prediction. "
        f"Variables: {list(ds.data_vars)}"
    )


def maybe_to_celsius(arr: np.ndarray, units: str | None = None) -> np.ndarray:
    units_l = str(units).lower()

    finite = arr[np.isfinite(arr)]
    mean_val = float(np.mean(finite)) if finite.size else np.nan

    if (
        units_l in ["k", "kelvin"]
        or "kelvin" in units_l
        or mean_val > 100
    ):
        return arr - 273.15

    return arr


class StreamingMetrics:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.n = 0.0
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred2 = 0.0
        self.sum_target2 = 0.0
        self.sum_pred_target = 0.0
        self.sum_diff = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray):
        pred = pred.astype(np.float64, copy=False)
        target = target.astype(np.float64, copy=False)

        valid = np.isfinite(pred) & np.isfinite(target)

        pred = pred[valid]
        target = target[valid]

        if pred.size == 0:
            return

        diff = pred - target

        self.n += pred.size
        self.sum_pred += float(pred.sum())
        self.sum_target += float(target.sum())
        self.sum_pred2 += float((pred ** 2).sum())
        self.sum_target2 += float((target ** 2).sum())
        self.sum_pred_target += float((pred * target).sum())
        self.sum_diff += float(diff.sum())
        self.sum_abs += float(np.abs(diff).sum())
        self.sum_sq += float((diff ** 2).sum())

    def finalize(self) -> dict:
        n = max(self.n, 1.0)

        pred_mean = self.sum_pred / n
        target_mean = self.sum_target / n

        pred_var = max(
            self.sum_pred2 / n - pred_mean ** 2,
            0.0
        )

        target_var = max(
            self.sum_target2 / n - target_mean ** 2,
            0.0
        )

        pred_std = pred_var ** 0.5
        target_std = target_var ** 0.5

        covariance = (
            self.sum_pred_target / n
            - pred_mean * target_mean
        )

        pearson = (
            covariance / (pred_std * target_std)
            if pred_std > 0 and target_std > 0
            else np.nan
        )

        mse = self.sum_sq / n

        return {
            "model": self.model_name,
            "n": int(self.n),
            "rmse": mse ** 0.5,
            "mae": self.sum_abs / n,
            "bias": self.sum_diff / n,
            "abs_bias": abs(self.sum_diff / n),
            "pearson": pearson,
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "pred_std": pred_std,
            "target_std": target_std,
            "std_ratio": (
                pred_std / target_std
                if target_std > 0
                else np.nan
            ),
        }


def build_phase1(checkpoint: Path, device):
    model = ModelECMIPSwinUNetTopography(
        dynamic_channels=24,
        coarse_static_channels=1,
        fine_static_channels=6,
        time_dim=3,
        ghg_dim=4,
        base_channels=64,
        dropout=0.05,
    ).to(device)

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def build_e2(checkpoint: Path, device):
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

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


# ============================================================
# Main evaluation
# ============================================================

@torch.no_grad()
def main():

    print("=" * 100)
    print("FINAL SEEDED SSP3-7.0 EVALUATION")
    print("=" * 100)

    files = {
        "phase1_full": PHASE1_FULL_CKPT,
        "e2_full": E2_FULL_CKPT,
        "phase1_no_topo": PHASE1_NO_TOPO_CKPT,
        "e2_no_topo": E2_NO_TOPO_CKPT,
        "unet": UNET_PATH,
    }

    for name, path in files.items():
        print(f"{name:20s}: {path}")

        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\ndevice:", device)

    loader = create_dataloader(
        split="test",
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("test samples:", len(loader.dataset))
    print("test batches:", len(loader))

    phase1_full, ckpt_p1_full = build_phase1(
        PHASE1_FULL_CKPT,
        device,
    )

    e2_full, ckpt_e2_full = build_e2(
        E2_FULL_CKPT,
        device,
    )

    phase1_no_topo, ckpt_p1_no = build_phase1(
        PHASE1_NO_TOPO_CKPT,
        device,
    )

    e2_no_topo, ckpt_e2_no = build_e2(
        E2_NO_TOPO_CKPT,
        device,
    )

    print("\nSelected checkpoint epochs")
    print(
        "Phase1 full      :",
        ckpt_p1_full.get("epoch"),
    )
    print(
        "E2 full          :",
        ckpt_e2_full.get("epoch"),
    )
    print(
        "Phase1 no topo   :",
        ckpt_p1_no.get("epoch"),
    )
    print(
        "E2 no topo       :",
        ckpt_e2_no.get("epoch"),
    )

    ds_unet = xr.open_dataset(UNET_PATH)

    unet_var = find_tas_var(ds_unet)
    unet_da = ds_unet[unet_var]
    unet_units = unet_da.attrs.get("units", "unknown")

    print("\nU-Net")
    print("variable:", unet_var)
    print("units   :", unet_units)
    print("shape   :", unet_da.shape)

    if unet_da.shape[0] != len(loader.dataset):
        raise RuntimeError(
            f"U-Net time dimension {unet_da.shape[0]} "
            f"!= SSP370 test samples {len(loader.dataset)}"
        )

    metrics = {
        "TopoFusion_Base_full":
            StreamingMetrics("TopoFusion_Base_full"),

        "TopoFusion_Res_full":
            StreamingMetrics("TopoFusion_Res_full"),

        "TopoFusion_Base_no_fine_topo":
            StreamingMetrics("TopoFusion_Base_no_fine_topo"),

        "TopoFusion_Res_no_fine_topo":
            StreamingMetrics("TopoFusion_Res_no_fine_topo"),

        "UNet_CMIP_CMIP":
            StreamingMetrics("UNet_CMIP_CMIP"),
    }

    offset = 0

    valid_cells_per_sample = []

    for batch_idx, batch in enumerate(loader, start=1):

        b = batch["target"].shape[0]

        dynamic = batch["dynamic"].to(
            device,
            non_blocking=True,
        )

        coarse_static = batch["coarse_static"].to(
            device,
            non_blocking=True,
        )

        fine_static_full = batch["fine_static"].to(
            device,
            non_blocking=True,
        )

        # Ablación controlada:
        # misma arquitectura, pero Z/G/R = 0
        fine_static_no_topo = fine_static_full.clone()
        fine_static_no_topo[:, FINE_TOPO_CHANNELS, :, :] = 0.0

        time_features = batch["time_features"].to(
            device,
            non_blocking=True,
        )

        ghg_features = batch["ghg_features"].to(
            device,
            non_blocking=True,
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

        # ----------------------------------------------------
        # Full
        # ----------------------------------------------------

        pred_p1_full = (
            phase1_full(
                dynamic=dynamic,
                coarse_static=coarse_static,
                fine_static=fine_static_full,
                time_features=time_features,
                ghg_features=ghg_features,
            )
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        out_e2_full = e2_full(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static_full,
            time_features=time_features,
            ghg_features=ghg_features,
            return_parts=True,
        )

        pred_e2_full = (
            out_e2_full["tas_final"]
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # No fine topography
        # ----------------------------------------------------

        pred_p1_no = (
            phase1_no_topo(
                dynamic=dynamic,
                coarse_static=coarse_static,
                fine_static=fine_static_no_topo,
                time_features=time_features,
                ghg_features=ghg_features,
            )
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        out_e2_no = e2_no_topo(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static_no_topo,
            time_features=time_features,
            ghg_features=ghg_features,
            return_parts=True,
        )

        pred_e2_no = (
            out_e2_no["tas_final"]
            .cpu()
            .numpy()[:, 0]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # U-Net
        # ----------------------------------------------------

        pred_unet = np.squeeze(
            unet_da
            .isel(time=slice(offset, offset + b))
            .values
        ).astype(np.float32)

        pred_unet = maybe_to_celsius(
            pred_unet,
            unet_units,
        )

        if pred_unet.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch: "
                f"U-Net={pred_unet.shape}, "
                f"target={target.shape}"
            )

        # ----------------------------------------------------
        # MISMA máscara para los cinco modelos
        # ----------------------------------------------------

        aligned_mask = (
            base_mask
            & np.isfinite(target)
            & np.isfinite(pred_p1_full)
            & np.isfinite(pred_e2_full)
            & np.isfinite(pred_p1_no)
            & np.isfinite(pred_e2_no)
            & np.isfinite(pred_unet)
        )

        metrics["TopoFusion_Base_full"].update(
            pred_p1_full[aligned_mask],
            target[aligned_mask],
        )

        metrics["TopoFusion_Res_full"].update(
            pred_e2_full[aligned_mask],
            target[aligned_mask],
        )

        metrics["TopoFusion_Base_no_fine_topo"].update(
            pred_p1_no[aligned_mask],
            target[aligned_mask],
        )

        metrics["TopoFusion_Res_no_fine_topo"].update(
            pred_e2_no[aligned_mask],
            target[aligned_mask],
        )

        metrics["UNet_CMIP_CMIP"].update(
            pred_unet[aligned_mask],
            target[aligned_mask],
        )

        valid_per_sample = (
            aligned_mask
            .reshape(b, -1)
            .sum(axis=1)
        )

        valid_cells_per_sample.extend(
            valid_per_sample.tolist()
        )

        offset += b

        if batch_idx % 100 == 0:

            p1 = metrics[
                "TopoFusion_Base_full"
            ].finalize()

            e2 = metrics[
                "TopoFusion_Res_full"
            ].finalize()

            nt = metrics[
                "TopoFusion_Res_no_fine_topo"
            ].finalize()

            un = metrics[
                "UNet_CMIP_CMIP"
            ].finalize()

            print(
                f"batch={batch_idx:05d}/{len(loader)} | "
                f"P1={p1['rmse']:.4f} | "
                f"E2={e2['rmse']:.4f} | "
                f"E2-noTopo={nt['rmse']:.4f} | "
                f"UNet={un['rmse']:.4f}"
            )

    if offset != len(loader.dataset):
        raise RuntimeError(
            f"Procesados {offset} días, "
            f"esperados {len(loader.dataset)}"
        )

    rows = [
        metric.finalize()
        for metric in metrics.values()
    ]

    df = pd.DataFrame(rows)

    by_name = {
        row["model"]: row
        for row in rows
    }

    base = by_name["TopoFusion_Base_full"]
    full = by_name["TopoFusion_Res_full"]
    base_no = by_name[
        "TopoFusion_Base_no_fine_topo"
    ]
    no_topo = by_name[
        "TopoFusion_Res_no_fine_topo"
    ]
    unet = by_name["UNet_CMIP_CMIP"]

    comparisons = {

        # Residual learning contribution
        "residual_rmse_improvement_abs":
            base["rmse"] - full["rmse"],

        "residual_rmse_improvement_pct":
            100.0
            * (base["rmse"] - full["rmse"])
            / base["rmse"],

        "residual_mae_improvement_abs":
            base["mae"] - full["mae"],

        # Fine topography contribution in final E2
        "fine_topography_rmse_improvement_abs":
            no_topo["rmse"] - full["rmse"],

        "fine_topography_rmse_improvement_pct":
            100.0
            * (no_topo["rmse"] - full["rmse"])
            / no_topo["rmse"],

        "fine_topography_mae_improvement_abs":
            no_topo["mae"] - full["mae"],

        # Fine topography in Stage 1
        "fine_topography_phase1_rmse_improvement_abs":
            base_no["rmse"] - base["rmse"],

        # Benchmark
        "topofusion_vs_unet_rmse_improvement_abs":
            unet["rmse"] - full["rmse"],

        "topofusion_vs_unet_rmse_improvement_pct":
            100.0
            * (unet["rmse"] - full["rmse"])
            / unet["rmse"],

        "topofusion_vs_unet_mae_improvement_abs":
            unet["mae"] - full["mae"],

        # Aligned evaluation audit
        "test_days": int(offset),

        "aligned_valid_cells_per_sample_min":
            int(np.min(valid_cells_per_sample)),

        "aligned_valid_cells_per_sample_max":
            int(np.max(valid_cells_per_sample)),

        "aligned_valid_cells_per_sample_mean":
            float(np.mean(valid_cells_per_sample)),
    }

    # ========================================================
    # Save
    # ========================================================

    csv_path = OUT_DIR / "final_seeded_ssp370_metrics.csv"
    json_path = OUT_DIR / "final_seeded_ssp370_metrics.json"
    comp_path = OUT_DIR / "final_seeded_ssp370_comparisons.json"

    df.to_csv(
        csv_path,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
            allow_nan=True,
        )
    )

    comp_path.write_text(
        json.dumps(
            comparisons,
            indent=2,
            allow_nan=True,
        )
    )

    print("\n" + "=" * 100)
    print("FINAL METRICS")
    print("=" * 100)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 100)
    print("COMPARISONS")
    print("=" * 100)

    print(
        json.dumps(
            comparisons,
            indent=2,
        )
    )

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    print(comp_path)


if __name__ == "__main__":
    main()
