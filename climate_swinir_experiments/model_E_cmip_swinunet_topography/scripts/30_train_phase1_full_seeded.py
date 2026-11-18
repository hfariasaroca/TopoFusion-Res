#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
import os

import torch
from torch.amp import GradScaler, autocast


ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP_DIR = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"
MODELS_DIR = EXP_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))

from dataset_model_E_ablation import create_dataloader, load_config, resolve_dataset_dir
from model_e_cmip_swinunet_topography import ModelECMIPSwinUNetTopography, count_parameters


RUN_NAME = "phase1_full_seed20260813_40epochs"

CHECKPOINT_DIR = EXP_DIR / "checkpoints" / RUN_NAME
LOG_DIR = EXP_DIR / "logs" / RUN_NAME
DIAG_DIR = EXP_DIR / "diagnostics" / RUN_NAME

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff ** 2).sum() / denom


def masked_mae(pred, target, mask):
    diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()

    total_sq = 0.0
    total_abs = 0.0
    total_diff = 0.0
    total_count = 0.0

    pred_sum = 0.0
    target_sum = 0.0

    for batch_idx, batch in enumerate(loader, start=1):
        dynamic = batch["dynamic"].to(device, non_blocking=True)
        coarse_static = batch["coarse_static"].to(device, non_blocking=True)
        fine_static = batch["fine_static"].to(device, non_blocking=True)
        time_features = batch["time_features"].to(device, non_blocking=True)
        ghg_features = batch["ghg_features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        pred = model(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static,
            time_features=time_features,
            ghg_features=ghg_features,
        )

        diff = (pred - target) * mask
        count = mask.sum().item()

        total_sq += float((diff ** 2).sum().item())
        total_abs += float(torch.abs(diff).sum().item())
        total_diff += float(diff.sum().item())
        total_count += count

        pred_sum += float((pred * mask).sum().item())
        target_sum += float((target * mask).sum().item())

    mse = total_sq / max(total_count, 1.0)
    rmse = mse ** 0.5
    mae = total_abs / max(total_count, 1.0)
    bias = total_diff / max(total_count, 1.0)

    return {
        "val_mse": mse,
        "val_rmse": rmse,
        "val_mae": mae,
        "val_bias": bias,
        "val_pred_mean": pred_sum / max(total_count, 1.0),
        "val_target_mean": target_sum / max(total_count, 1.0),
        "val_valid_points": int(total_count),
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, metrics, cfg, best_val_rmse):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "config": cfg,
            "run_name": RUN_NAME,
            "best_val_rmse": best_val_rmse,
        },
        path,
    )


def train():
    cfg = load_config()
    dataset_dir = resolve_dataset_dir(cfg)

    # Reproducibilidad de la ablación controlada
    seed = int(os.environ.get("MODEL_E_SEED", "20260813"))

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("seed        :", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    num_epochs = 40
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_workers = 0

    print("Model E Phase 1 Full Training")
    print("=" * 100)
    print("run_name    :", RUN_NAME)
    print("dataset     :", dataset_dir)
    print("device      :", device)
    print("batch_size  :", batch_size)
    print("num_epochs  :", num_epochs)
    print("lr          :", learning_rate)
    print("weight_decay:", weight_decay)
    print("=" * 100)

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

    model = ModelECMIPSwinUNetTopography(
        dynamic_channels=24,
        coarse_static_channels=1,
        fine_static_channels=6,
        time_dim=3,
        ghg_dim=4,
        base_channels=64,
        dropout=0.05,
    ).to(device)

    print("parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=4,
        min_lr=1e-6,
    )

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    log_path = LOG_DIR / "training_log.csv"

    fieldnames = [
        "epoch",
        "lr",
        "train_mse",
        "train_rmse",
        "train_mae",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_pred_mean",
        "val_target_mean",
        "val_valid_points",
        "epoch_seconds",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best_val_rmse = float("inf")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        total_sq = 0.0
        total_abs = 0.0
        total_count = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            dynamic = batch["dynamic"].to(device, non_blocking=True)
            coarse_static = batch["coarse_static"].to(device, non_blocking=True)
            fine_static = batch["fine_static"].to(device, non_blocking=True)
            time_features = batch["time_features"].to(device, non_blocking=True)
            ghg_features = batch["ghg_features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(
                    dynamic=dynamic,
                    coarse_static=coarse_static,
                    fine_static=fine_static,
                    time_features=time_features,
                    ghg_features=ghg_features,
                )

                loss = masked_mse(pred, target, mask)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss inválida epoch={epoch} batch={batch_idx}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                diff = (pred - target) * mask
                count = mask.sum().item()

                total_sq += float((diff ** 2).sum().item())
                total_abs += float(torch.abs(diff).sum().item())
                total_count += count

            if batch_idx % 200 == 0:
                train_mse_now = total_sq / max(total_count, 1.0)
                train_rmse_now = train_mse_now ** 0.5
                train_mae_now = total_abs / max(total_count, 1.0)

                print(
                    f"epoch={epoch:03d}/{num_epochs} "
                    f"batch={batch_idx:05d}/{len(train_loader)} "
                    f"train_rmse={train_rmse_now:.4f} "
                    f"train_mae={train_mae_now:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        train_mse = total_sq / max(total_count, 1.0)
        train_rmse = train_mse ** 0.5
        train_mae = total_abs / max(total_count, 1.0)

        val_metrics = evaluate_full(model, val_loader, device)

        scheduler.step(val_metrics["val_rmse"])

        epoch_seconds = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            **val_metrics,
            "epoch_seconds": epoch_seconds,
        }

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print("\nEpoch summary")
        print("-" * 100)
        print(json.dumps(row, indent=2))
        print("-" * 100)

        last_path = CHECKPOINT_DIR / "model_E_phase1_last.pth"
        save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            val_metrics,
            cfg,
            best_val_rmse,
        )

        if val_metrics["val_rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["val_rmse"]

            best_path = CHECKPOINT_DIR / "model_E_phase1_best.pth"
            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                val_metrics,
                cfg,
                best_val_rmse,
            )

            print(f"Saved best checkpoint: {best_path}")
            print(f"Best val RMSE: {best_val_rmse:.6f}")

    summary = {
        "run_name": RUN_NAME,
        "best_val_rmse": best_val_rmse,
        "log_path": str(log_path),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "dataset_dir": str(dataset_dir),
    }

    summary_path = DIAG_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nDONE")
    print("=" * 100)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train()
