#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
import os
import random

import numpy as np
import torch
import xarray as xr
from torch.amp import GradScaler, autocast


ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP_DIR = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"
MODELS_DIR = EXP_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))

from dataset_model_E_ablation import create_dataloader, load_config, resolve_dataset_dir
from model_e2_residual_head import ModelE2ResidualHead, load_phase1_weights_into_e2


RUN_NAME = "phase2_e2_full_seed20260813_12epochs"

PHASE1_CKPT = Path("/workspace/SwinIR_HF/Normalizacion/climate_swinir_experiments/model_E_cmip_swinunet_topography/checkpoints/phase1_full_seed20260813_40epochs/model_E_phase1_best.pth")
OROG_PATH = ROOT / "climate_swinir_analysis/data/orography_regridded_64x64_cellmean_copernicus_dem.nc"

CHECKPOINT_DIR = EXP_DIR / "checkpoints" / RUN_NAME
LOG_DIR = EXP_DIR / "logs" / RUN_NAME
DIAG_DIR = EXP_DIR / "diagnostics" / RUN_NAME

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def find_orog_var(ds):
    for c in ["orog", "elevation", "z", "Band1", "data"]:
        if c in ds.data_vars:
            return c
    for v in ds.data_vars:
        if np.squeeze(ds[v].values).shape == (64, 64):
            return v
    raise RuntimeError(f"No encontré variable de orografía. Variables: {list(ds.data_vars)}")


def make_edge_mask(land_mask: np.ndarray) -> np.ndarray:
    land = land_mask.astype(bool)
    ocean = ~land
    edge = np.zeros_like(land, dtype=bool)
    ny, nx = land.shape

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            src_y0 = max(0, -dy)
            src_y1 = min(ny, ny - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(nx, nx - dx)

            dst_y0 = max(0, dy)
            dst_y1 = min(ny, ny + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(nx, nx + dx)

            neighbor_ocean = np.zeros_like(land, dtype=bool)
            neighbor_ocean[dst_y0:dst_y1, dst_x0:dst_x1] = ocean[src_y0:src_y1, src_x0:src_x1]
            edge |= land & neighbor_ocean

    return edge


def build_band_masks(device):
    ds_orog = xr.open_dataset(OROG_PATH)
    orog_var = find_orog_var(ds_orog)
    orog = np.squeeze(ds_orog[orog_var].values).astype(np.float32)

    tmp_loader = create_dataloader(split="train", batch_size=1, shuffle=False, num_workers=0)
    first_batch = next(iter(tmp_loader))
    land = first_batch["mask"].numpy()[0, 0].astype(bool)

    coastal = make_edge_mask(land)
    noncoastal = land & ~coastal

    bands_np = {
        "coastal_edge": coastal,
        "low_noncoastal_lt500m": noncoastal & (orog < 500),
        "mid_noncoastal_500_1500m": noncoastal & (orog >= 500) & (orog < 1500),
        "high_noncoastal_gt1500m": noncoastal & (orog >= 1500),
    }

    bands_torch = {
        k: torch.from_numpy(v.astype(np.float32))[None, None].to(device)
        for k, v in bands_np.items()
    }

    band_counts = {k: int(v.sum()) for k, v in bands_np.items()}
    return bands_torch, band_counts


def set_train_mode_residual_only(model):
    """
    Punto crítico:
    - todo el modelo queda en eval()
    - solo residual_head queda en train()
    Así la base phase1 no activa dropout ni cambia comportamiento.
    """
    model.eval()
    model.residual_head.train()


def masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return (diff ** 2).sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred, target, mask):
    diff = torch.abs(pred - target) * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def masked_bias(pred, target, mask):
    diff = (pred - target) * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def masked_smooth_l1(pred, target, mask, beta=0.5):
    diff = pred - target
    abs_diff = torch.abs(diff)

    loss = torch.where(
        abs_diff < beta,
        0.5 * (diff ** 2) / beta,
        abs_diff - 0.5 * beta,
    )

    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def residual_l2(residual, mask):
    r = residual * mask
    return (r ** 2).sum() / mask.sum().clamp_min(1.0)


def residual_smoothness(residual, mask):
    dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    dx = (residual[:, :, :, 1:] - residual[:, :, :, :-1]) * dx_mask
    dy = (residual[:, :, 1:, :] - residual[:, :, :-1, :]) * dy_mask

    dx_loss = (dx ** 2).sum() / dx_mask.sum().clamp_min(1.0)
    dy_loss = (dy ** 2).sum() / dy_mask.sum().clamp_min(1.0)

    return dx_loss + dy_loss


def e2_v2_loss(tas_final, tas_base, target, mask, residual, band_masks):
    target_residual = (target - tas_base).detach()

    final_mse = masked_mse(tas_final, target, mask)
    residual_huber = masked_smooth_l1(residual, target_residual, mask, beta=0.5)

    band_residual_losses = []
    band_bias_losses = []

    for _, bmask in band_masks.items():
        full_mask = mask * bmask

        if full_mask.sum().item() <= 0:
            continue

        band_residual_losses.append(
            masked_smooth_l1(residual, target_residual, full_mask, beta=0.5)
        )

        band_bias_losses.append(
            masked_bias(tas_final, target, full_mask) ** 2
        )

    band_residual_huber = torch.stack(band_residual_losses).mean()
    band_bias_loss = torch.stack(band_bias_losses).mean()

    res_l2 = residual_l2(residual, mask)
    res_smooth = residual_smoothness(residual, mask)

    loss = (
        final_mse
        + 0.20 * residual_huber
        + 0.08 * band_residual_huber
        + 0.02 * band_bias_loss
        + 0.03 * res_l2
        + 0.002 * res_smooth
    )

    parts = {
        "loss": loss.detach(),
        "final_mse": final_mse.detach(),
        "residual_huber": residual_huber.detach(),
        "band_residual_huber": band_residual_huber.detach(),
        "band_bias_loss": band_bias_loss.detach(),
        "res_l2": res_l2.detach(),
        "res_smooth": res_smooth.detach(),
    }

    return loss, parts


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()

    total_sq = 0.0
    total_abs = 0.0
    total_diff = 0.0
    total_count = 0.0

    pred_sum = 0.0
    target_sum = 0.0

    residual_sum = 0.0
    residual_abs_sum = 0.0
    residual_sq_sum = 0.0
    residual_max_abs = 0.0

    for batch in loader:
        dynamic = batch["dynamic"].to(device, non_blocking=True)
        coarse_static = batch["coarse_static"].to(device, non_blocking=True)
        fine_static = batch["fine_static"].to(device, non_blocking=True)
        time_features = batch["time_features"].to(device, non_blocking=True)
        ghg_features = batch["ghg_features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        out = model(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static,
            time_features=time_features,
            ghg_features=ghg_features,
            return_parts=True,
        )

        pred = out["tas_final"]
        residual = out["residual"]

        diff = (pred - target) * mask
        count = mask.sum().item()

        total_sq += float((diff ** 2).sum().item())
        total_abs += float(torch.abs(diff).sum().item())
        total_diff += float(diff.sum().item())
        total_count += count

        pred_sum += float((pred * mask).sum().item())
        target_sum += float((target * mask).sum().item())

        r = residual * mask
        residual_sum += float(r.sum().item())
        residual_abs_sum += float(torch.abs(r).sum().item())
        residual_sq_sum += float((r ** 2).sum().item())
        residual_max_abs = max(residual_max_abs, float(torch.abs(r).max().item()))

    mse = total_sq / max(total_count, 1.0)
    bias = total_diff / max(total_count, 1.0)

    return {
        "val_mse": mse,
        "val_rmse": mse ** 0.5,
        "val_mae": total_abs / max(total_count, 1.0),
        "val_bias": bias,
        "val_abs_bias": abs(bias),
        "val_pred_mean": pred_sum / max(total_count, 1.0),
        "val_target_mean": target_sum / max(total_count, 1.0),
        "val_valid_points": int(total_count),
        "residual_mean": residual_sum / max(total_count, 1.0),
        "residual_abs_mean": residual_abs_sum / max(total_count, 1.0),
        "residual_rms": (residual_sq_sum / max(total_count, 1.0)) ** 0.5,
        "residual_max_abs": residual_max_abs,
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, metrics, cfg, best_score):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "config": cfg,
            "run_name": RUN_NAME,
            "best_score": best_score,
            "source_checkpoint": str(PHASE1_CKPT),
            "note": "Model E2 v2 residual head, frozen base, base eval during training",
        },
        path,
    )


def main():
    cfg = load_config()
    dataset_dir = resolve_dataset_dir(cfg)

    # Reproducibilidad de la ablación controlada
    seed = int(os.environ.get("MODEL_E_SEED", "20260813"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("seed      :", seed)

    if not PHASE1_CKPT.exists():
        raise FileNotFoundError(PHASE1_CKPT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    num_epochs = 12
    learning_rate = 3e-5
    weight_decay = 1e-4
    num_workers = 0

    print("Model E2 v2 residual-head training, base eval")
    print("=" * 100)
    print("run_name :", RUN_NAME)
    print("dataset  :", dataset_dir)
    print("phase1   :", PHASE1_CKPT)
    print("device   :", device)
    print("epochs   :", num_epochs)
    print("lr       :", learning_rate)

    band_masks, band_counts = build_band_masks(device)
    print("\nBand cells")
    print(json.dumps(band_counts, indent=2))
    (DIAG_DIR / "band_counts.json").write_text(json.dumps(band_counts, indent=2))

    train_loader = create_dataloader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = create_dataloader(
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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

    load_info = load_phase1_weights_into_e2(model, str(PHASE1_CKPT), device)
    print("\nLoaded phase1 info")
    print(json.dumps(load_info, indent=2, default=str))

    model.freeze_base()
    print("\nParameter summary after freeze")
    print(json.dumps(model.trainable_parameter_summary(), indent=2))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=2,
        min_lr=1e-6,
    )

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    log_path = LOG_DIR / "training_log.csv"

    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_final_mse",
        "train_residual_huber",
        "train_band_residual_huber",
        "train_band_bias_loss",
        "train_res_l2",
        "train_res_smooth",
        "train_rmse",
        "train_mae",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_abs_bias",
        "val_pred_mean",
        "val_target_mean",
        "val_valid_points",
        "residual_mean",
        "residual_abs_mean",
        "residual_rms",
        "residual_max_abs",
        "selection_score",
        "epoch_seconds",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Epoch 0: baseline idéntico a phase1.
    val0 = evaluate_full(model, val_loader, device)
    score0 = val0["val_rmse"] + 0.05 * val0["val_abs_bias"]

    row0 = {
        "epoch": 0,
        "lr": learning_rate,
        "train_loss": "",
        "train_final_mse": "",
        "train_residual_huber": "",
        "train_band_residual_huber": "",
        "train_band_bias_loss": "",
        "train_res_l2": "",
        "train_res_smooth": "",
        "train_rmse": "",
        "train_mae": "",
        **val0,
        "selection_score": score0,
        "epoch_seconds": 0.0,
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row0)

    best_score = score0

    save_checkpoint(
        CHECKPOINT_DIR / "model_E2_v2_best.pth",
        0,
        model,
        optimizer,
        scheduler,
        scaler,
        val0,
        cfg,
        best_score,
    )

    print("\nEpoch 0 baseline")
    print(json.dumps(row0, indent=2))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        set_train_mode_residual_only(model)

        total_loss = 0.0
        total_final_mse = 0.0
        total_residual_huber = 0.0
        total_band_residual_huber = 0.0
        total_band_bias = 0.0
        total_res_l2 = 0.0
        total_res_smooth = 0.0

        total_sq = 0.0
        total_abs = 0.0
        total_count = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            # Reafirmar cada iteración, por seguridad.
            set_train_mode_residual_only(model)

            dynamic = batch["dynamic"].to(device, non_blocking=True)
            coarse_static = batch["coarse_static"].to(device, non_blocking=True)
            fine_static = batch["fine_static"].to(device, non_blocking=True)
            time_features = batch["time_features"].to(device, non_blocking=True)
            ghg_features = batch["ghg_features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(
                    dynamic=dynamic,
                    coarse_static=coarse_static,
                    fine_static=fine_static,
                    time_features=time_features,
                    ghg_features=ghg_features,
                    return_parts=True,
                )

                tas_final = out["tas_final"]
                tas_base = out["tas_base"]
                residual = out["residual"]

                loss, parts = e2_v2_loss(
                    tas_final=tas_final,
                    tas_base=tas_base,
                    target=target,
                    mask=mask,
                    residual=residual,
                    band_masks=band_masks,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss inválida epoch={epoch} batch={batch_idx}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                diff = (tas_final - target) * mask
                count = mask.sum().item()

                total_sq += float((diff ** 2).sum().item())
                total_abs += float(torch.abs(diff).sum().item())
                total_count += count

            total_loss += float(parts["loss"].item())
            total_final_mse += float(parts["final_mse"].item())
            total_residual_huber += float(parts["residual_huber"].item())
            total_band_residual_huber += float(parts["band_residual_huber"].item())
            total_band_bias += float(parts["band_bias_loss"].item())
            total_res_l2 += float(parts["res_l2"].item())
            total_res_smooth += float(parts["res_smooth"].item())
            n_batches += 1

            if batch_idx % 200 == 0:
                train_mse_now = total_sq / max(total_count, 1.0)
                print(
                    f"epoch={epoch:03d}/{num_epochs} "
                    f"batch={batch_idx:05d}/{len(train_loader)} "
                    f"train_rmse={train_mse_now ** 0.5:.4f} "
                    f"train_mae={total_abs / max(total_count, 1.0):.4f} "
                    f"loss={loss.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        train_mse = total_sq / max(total_count, 1.0)
        train_rmse = train_mse ** 0.5
        train_mae = total_abs / max(total_count, 1.0)

        val_metrics = evaluate_full(model, val_loader, device)

        selection_score = val_metrics["val_rmse"] + 0.05 * val_metrics["val_abs_bias"]
        scheduler.step(selection_score)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(n_batches, 1),
            "train_final_mse": total_final_mse / max(n_batches, 1),
            "train_residual_huber": total_residual_huber / max(n_batches, 1),
            "train_band_residual_huber": total_band_residual_huber / max(n_batches, 1),
            "train_band_bias_loss": total_band_bias / max(n_batches, 1),
            "train_res_l2": total_res_l2 / max(n_batches, 1),
            "train_res_smooth": total_res_smooth / max(n_batches, 1),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            **val_metrics,
            "selection_score": selection_score,
            "epoch_seconds": time.time() - t0,
        }

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print("\nEpoch summary")
        print("-" * 100)
        print(json.dumps(row, indent=2))
        print("-" * 100)

        save_checkpoint(
            CHECKPOINT_DIR / "model_E2_v2_last.pth",
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            val_metrics,
            cfg,
            best_score,
        )

        if selection_score < best_score:
            best_score = selection_score
            save_checkpoint(
                CHECKPOINT_DIR / "model_E2_v2_best.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                val_metrics,
                cfg,
                best_score,
            )
            print("Saved best checkpoint")
            print("Best selection score:", best_score)

    summary = {
        "run_name": RUN_NAME,
        "best_score": best_score,
        "source_checkpoint": str(PHASE1_CKPT),
        "log_path": str(log_path),
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }

    (DIAG_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
