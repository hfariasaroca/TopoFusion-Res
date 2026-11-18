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
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP_DIR = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"
MODELS_DIR = EXP_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))

from dataset_model_E import create_dataloader
from model_e2_residual_head import ModelE2ResidualHead


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

E2_CKPT = EXP_DIR / "checkpoints/phase2_e2_residual_head_base_eval_12epochs/model_E2_v2_best.pth"
UNET_PATH = ROOT / "data_tomas/unet_outputs/tas_cmip_cmip_output.nc"
TARGET_PATH = ROOT / "data_tomas/data_perfect_model/tas_ssp370_20150101-21001231.nc"
OROG_PATH = ROOT / "climate_swinir_analysis/data/orography_regridded_64x64_cellmean_copernicus_dem.nc"

OUT_DIR = EXP_DIR / "outputs/figures/final_e2_vs_unet_reference_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_DIR = EXP_DIR / "diagnostics/final_e2_vs_unet_advanced_diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

CUBE_DIR = DIAG_DIR / "prediction_cubes"
CUBE_DIR.mkdir(parents=True, exist_ok=True)

FORCE_RECOMPUTE_CUBES = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_tas_var(ds: xr.Dataset) -> str:
    for c in ["tas", "tas_pred", "prediction", "output", "y_pred"]:
        if c in ds.data_vars:
            return c

    for v in ds.data_vars:
        if ds[v].ndim >= 3:
            return v

    raise RuntimeError(f"No encontré variable tipo tas/prediction. Variables: {list(ds.data_vars)}")


def find_orog_var(ds: xr.Dataset) -> str:
    for c in ["orog", "elevation", "z", "Band1", "data"]:
        if c in ds.data_vars:
            return c

    for v in ds.data_vars:
        arr = np.squeeze(ds[v].values)
        if arr.shape == (64, 64):
            return v

    raise RuntimeError(f"No encontré variable de orografía. Variables: {list(ds.data_vars)}")


def maybe_to_celsius(arr: np.ndarray, units: str | None = None) -> np.ndarray:
    units_l = str(units).lower()
    finite = arr[np.isfinite(arr)]
    mean_val = float(np.mean(finite)) if finite.size else np.nan

    if units_l in ["k", "kelvin"] or "kelvin" in units_l or mean_val > 100:
        return arr - 273.15

    return arr


def load_orography() -> np.ndarray:
    ds = xr.open_dataset(OROG_PATH)
    var = find_orog_var(ds)
    orog = np.squeeze(ds[var].values).astype(np.float32)
    return np.where(np.isfinite(orog), orog, np.nan)


def make_hillshade(orog: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(orog), orog, np.nanmedian(orog))
    ls = LightSource(azdeg=315, altdeg=45)

    hs = ls.hillshade(
        filled,
        vert_exag=0.8,
        dx=1,
        dy=1,
    )

    return np.where(np.isfinite(orog), hs, np.nan)


def add_orography(ax, orog, hillshade, alpha=0.65):
    ax.imshow(hillshade, origin="lower", cmap="gray", alpha=alpha)

    ax.contour(
        orog,
        levels=[500, 1500, 2500],
        colors="black",
        linewidths=0.55,
        alpha=0.45,
        origin="lower",
    )


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


def build_bands(orog: np.ndarray, land_mask: np.ndarray) -> dict[str, np.ndarray]:
    coastal = make_edge_mask(land_mask)
    noncoastal = land_mask & ~coastal

    bands = {
        "Coast": coastal,
        "Low interior <500m": noncoastal & (orog < 500),
        "Mid elevation 500-1500m": noncoastal & (orog >= 500) & (orog < 1500),
        "High Andes >1500m": noncoastal & (orog >= 1500),
        "Full domain": land_mask,
    }

    return bands


def get_time_arrays(n: int):
    """
    Retorna:
    - years
    - months
    - days
    - seasons
    - decimal_year
    """

    if TARGET_PATH.exists():
        try:
            ds = xr.open_dataset(TARGET_PATH)

            if "time" in ds.coords:
                times = ds["time"].values[:n]

                years = []
                months = []
                days = []

                for t in times:
                    if hasattr(t, "year") and hasattr(t, "month") and hasattr(t, "day"):
                        years.append(int(t.year))
                        months.append(int(t.month))
                        days.append(int(t.day))
                    else:
                        ts = pd.Timestamp(t)
                        years.append(int(ts.year))
                        months.append(int(ts.month))
                        days.append(int(ts.day))

                years = np.array(years, dtype=int)
                months = np.array(months, dtype=int)
                days = np.array(days, dtype=int)

                month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
                cum_days = np.concatenate([[0], np.cumsum(month_days)])
                doys = cum_days[months - 1] + days

                decimal_year = years.astype(float) + (doys.astype(float) - 1.0) / 365.0
                seasons = months_to_seasons(months)

                return years, months, days, seasons, decimal_year

        except Exception as e:
            print("WARNING: no pude leer tiempo desde TARGET_PATH. Uso fallback no-leap.")
            print("Reason:", repr(e))

    idx = np.arange(n)
    years = 2015 + idx // 365
    day_of_year = idx % 365

    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cum_days = np.concatenate([[0], np.cumsum(month_days)])

    months = np.searchsorted(cum_days[1:], day_of_year, side="right") + 1
    days = day_of_year - cum_days[months - 1] + 1
    decimal_year = 2015.0 + idx / 365.0
    seasons = months_to_seasons(months)

    return years.astype(int), months.astype(int), days.astype(int), seasons, decimal_year


def months_to_seasons(months: np.ndarray) -> np.ndarray:
    out = np.empty(months.shape, dtype=object)

    out[np.isin(months, [12, 1, 2])] = "DJF"
    out[np.isin(months, [3, 4, 5])] = "MAM"
    out[np.isin(months, [6, 7, 8])] = "JJA"
    out[np.isin(months, [9, 10, 11])] = "SON"

    return out


def load_e2_model(device):
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

    ckpt = torch.load(E2_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded E2 checkpoint:", E2_CKPT)
    print("E2 checkpoint epoch:", ckpt.get("epoch"))
    print("E2 validation metrics:", ckpt.get("metrics"))

    return model


# ---------------------------------------------------------------------
# Prediction cubes
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_or_load_cubes():
    """
    Crea memmaps:
    - rcm.dat
    - e2.dat
    - unet.dat
    - mask.dat

    Shape: [time, 64, 64]
    """

    meta_path = CUBE_DIR / "cube_metadata.json"

    if meta_path.exists() and not FORCE_RECOMPUTE_CUBES:
        meta = json.loads(meta_path.read_text())
        shape = tuple(meta["shape"])

        print("Using existing prediction cubes:")
        print(meta_path)

        rcm = np.memmap(CUBE_DIR / "rcm.dat", dtype="float32", mode="r", shape=shape)
        e2 = np.memmap(CUBE_DIR / "e2.dat", dtype="float32", mode="r", shape=shape)
        unet = np.memmap(CUBE_DIR / "unet.dat", dtype="float32", mode="r", shape=shape)
        mask = np.memmap(CUBE_DIR / "mask.dat", dtype="bool", mode="r", shape=shape)

        return rcm, e2, unet, mask, meta

    print("Computing prediction cubes")
    print("=" * 100)

    for p in [E2_CKPT, UNET_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    e2_model = load_e2_model(device)

    ds_unet = xr.open_dataset(UNET_PATH)
    unet_var = find_tas_var(ds_unet)
    unet_da = ds_unet[unet_var]
    unet_units = unet_da.attrs.get("units", "unknown")

    print("U-Net variable:", unet_var)
    print("U-Net units:", unet_units)
    print("U-Net shape:", unet_da.shape)

    loader = create_dataloader(
        split="test",
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    n_time = len(loader.dataset)
    shape = (n_time, 64, 64)

    rcm = np.memmap(CUBE_DIR / "rcm.dat", dtype="float32", mode="w+", shape=shape)
    e2 = np.memmap(CUBE_DIR / "e2.dat", dtype="float32", mode="w+", shape=shape)
    unet = np.memmap(CUBE_DIR / "unet.dat", dtype="float32", mode="w+", shape=shape)
    mask = np.memmap(CUBE_DIR / "mask.dat", dtype="bool", mode="w+", shape=shape)

    offset = 0

    for batch_idx, batch in enumerate(loader, start=1):
        b = batch["target"].shape[0]

        dynamic = batch["dynamic"].to(device, non_blocking=True)
        coarse_static = batch["coarse_static"].to(device, non_blocking=True)
        fine_static = batch["fine_static"].to(device, non_blocking=True)
        time_features = batch["time_features"].to(device, non_blocking=True)
        ghg_features = batch["ghg_features"].to(device, non_blocking=True)

        target = batch["target"].numpy()[:, 0].astype(np.float32)
        base_mask = batch["mask"].numpy()[:, 0].astype(bool)

        pred_e2 = e2_model(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static,
            time_features=time_features,
            ghg_features=ghg_features,
            return_parts=False,
        ).detach().cpu().numpy()[:, 0].astype(np.float32)

        pred_unet = np.squeeze(
            unet_da.isel(time=slice(offset, offset + b)).values
        ).astype(np.float32)

        if pred_unet.ndim == 2:
            pred_unet = pred_unet[None, :, :]

        pred_unet = maybe_to_celsius(pred_unet, unet_units)

        if pred_unet.shape != target.shape:
            raise RuntimeError(f"U-Net shape mismatch: {pred_unet.shape} vs {target.shape}")

        aligned = (
            base_mask
            & np.isfinite(target)
            & np.isfinite(pred_e2)
            & np.isfinite(pred_unet)
        )

        rcm[offset:offset + b] = target
        e2[offset:offset + b] = pred_e2
        unet[offset:offset + b] = pred_unet
        mask[offset:offset + b] = aligned

        offset += b

        if batch_idx % 100 == 0:
            print(f"batch={batch_idx:05d}/{len(loader)} | offset={offset}")

    rcm.flush()
    e2.flush()
    unet.flush()
    mask.flush()

    if offset != n_time:
        raise RuntimeError(f"offset final {offset} != n_time {n_time}")

    meta = {
        "shape": list(shape),
        "n_time": int(n_time),
        "paths": {
            "rcm": str(CUBE_DIR / "rcm.dat"),
            "e2": str(CUBE_DIR / "e2.dat"),
            "unet": str(CUBE_DIR / "unet.dat"),
            "mask": str(CUBE_DIR / "mask.dat"),
        },
        "reference": "ALADIN/RCM target",
        "model": "Model E2 v2",
        "baseline": "U-Net CMIP-CMIP",
    }

    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved cubes metadata:", meta_path)

    rcm = np.memmap(CUBE_DIR / "rcm.dat", dtype="float32", mode="r", shape=shape)
    e2 = np.memmap(CUBE_DIR / "e2.dat", dtype="float32", mode="r", shape=shape)
    unet = np.memmap(CUBE_DIR / "unet.dat", dtype="float32", mode="r", shape=shape)
    mask = np.memmap(CUBE_DIR / "mask.dat", dtype="bool", mode="r", shape=shape)

    return rcm, e2, unet, mask, meta


# ---------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------

class MetricAcc:
    def __init__(self):
        self.n = 0.0
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred2 = 0.0
        self.sum_target2 = 0.0
        self.sum_pred_target = 0.0
        self.sum_diff = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0

    def update(self, pred, target, m):
        valid = m & np.isfinite(pred) & np.isfinite(target)

        if not valid.any():
            return

        p = pred[valid].astype(np.float64, copy=False)
        t = target[valid].astype(np.float64, copy=False)

        diff = p - t

        self.n += p.size
        self.sum_pred += float(p.sum())
        self.sum_target += float(t.sum())
        self.sum_pred2 += float((p ** 2).sum())
        self.sum_target2 += float((t ** 2).sum())
        self.sum_pred_target += float((p * t).sum())
        self.sum_diff += float(diff.sum())
        self.sum_abs += float(np.abs(diff).sum())
        self.sum_sq += float((diff ** 2).sum())

    def finalize(self):
        n = max(self.n, 1.0)

        pred_mean = self.sum_pred / n
        target_mean = self.sum_target / n

        pred_var = max(self.sum_pred2 / n - pred_mean ** 2, 0.0)
        target_var = max(self.sum_target2 / n - target_mean ** 2, 0.0)

        pred_std = pred_var ** 0.5
        target_std = target_var ** 0.5

        cov = self.sum_pred_target / n - pred_mean * target_mean
        pearson = cov / (pred_std * target_std) if pred_std > 0 and target_std > 0 else np.nan

        mse = self.sum_sq / n
        bias = self.sum_diff / n

        return {
            "n": int(self.n),
            "rmse": mse ** 0.5,
            "mae": self.sum_abs / n,
            "bias": bias,
            "abs_bias": abs(bias),
            "pearson": pearson,
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "pred_std": pred_std,
            "target_std": target_std,
            "std_ratio": pred_std / target_std if target_std > 0 else np.nan,
        }


# ---------------------------------------------------------------------
# 1. Seasonal metrics
# ---------------------------------------------------------------------

def compute_seasonal_metrics(rcm, e2, unet, mask, seasons, bands):
    print("\nComputing seasonal metrics")
    print("=" * 100)

    season_order = ["DJF", "MAM", "JJA", "SON"]
    model_map = {
        "Model E2 v2": e2,
        "U-Net CMIP-CMIP": unet,
    }

    accs = {
        (model_name, season, band_name): MetricAcc()
        for model_name in model_map
        for season in season_order
        for band_name in bands
    }

    n_time = rcm.shape[0]
    chunk = 512

    for start in range(0, n_time, chunk):
        end = min(start + chunk, n_time)

        r = np.asarray(rcm[start:end])
        m0 = np.asarray(mask[start:end])
        season_chunk = seasons[start:end]

        for season in season_order:
            sm = season_chunk == season

            if not sm.any():
                continue

            for band_name, band_mask in bands.items():
                mm = m0[sm] & band_mask[None, :, :]

                for model_name, pred_mm in model_map.items():
                    p = np.asarray(pred_mm[start:end])[sm]
                    accs[(model_name, season, band_name)].update(p, r[sm], mm)

        if start % (chunk * 20) == 0:
            print(f"seasonal metrics: {start:06d}/{n_time}")

    rows = []

    for (model_name, season, band_name), acc in accs.items():
        row = {
            "model": model_name,
            "season": season,
            "band": band_name,
            **acc.finalize(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = DIAG_DIR / "final_seasonal_metrics_by_band.csv"
    out_json = DIAG_DIR / "final_seasonal_metrics_by_band.json"

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    return df


def plot_seasonal_bars(df):
    selected_bands = [
        "Coast",
        "Low interior <500m",
        "Mid elevation 500-1500m",
        "High Andes >1500m",
        "Full domain",
    ]

    season_order = ["DJF", "MAM", "JJA", "SON"]

    for metric, ylabel, outname in [
        ("rmse", "RMSE (°C)", "final_seasonal_rmse_by_band_e2_vs_unet.png"),
        ("mae", "MAE (°C)", "final_seasonal_mae_by_band_e2_vs_unet.png"),
        ("bias", "Bias: model - RCM (°C)", "final_seasonal_bias_by_band_e2_vs_unet.png"),
        ("abs_bias", "Absolute bias (°C)", "final_seasonal_abs_bias_by_band_e2_vs_unet.png"),
    ]:
        fig, axes = plt.subplots(
            len(selected_bands),
            1,
            figsize=(12, 12),
            sharex=True,
            constrained_layout=True,
        )

        fig.suptitle(
            f"Seasonal {ylabel} by topographic band",
            fontsize=20,
            weight="bold",
            y=1.02,
        )

        x = np.arange(len(season_order))
        width = 0.36

        for ax, band in zip(axes, selected_bands):
            e2_vals = []
            unet_vals = []

            for season in season_order:
                e2_row = df[
                    (df["model"] == "Model E2 v2")
                    & (df["band"] == band)
                    & (df["season"] == season)
                ].iloc[0]

                unet_row = df[
                    (df["model"] == "U-Net CMIP-CMIP")
                    & (df["band"] == band)
                    & (df["season"] == season)
                ].iloc[0]

                e2_vals.append(float(e2_row[metric]))
                unet_vals.append(float(unet_row[metric]))

            ax.bar(x - width / 2, e2_vals, width, label="Model E2 v2")
            ax.bar(x + width / 2, unet_vals, width, label="U-Net CMIP-CMIP")

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.set_title(band, loc="left", fontsize=12, weight="bold")
            ax.grid(axis="y", alpha=0.25)

            for i, (a, b) in enumerate(zip(e2_vals, unet_vals)):
                ax.text(i - width / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
                ax.text(i + width / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(season_order)

        axes[0].legend(ncol=2, loc="upper left")

        fig.text(
            0.5,
            -0.02,
            "All metrics computed against ALADIN/RCM SSP3-7.0 reference.",
            ha="center",
            va="top",
            fontsize=10.5,
        )

        out = OUT_DIR / outname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", out)


# ---------------------------------------------------------------------
# 2. Extremes p01 / p99
# ---------------------------------------------------------------------

def compute_percentile_maps(arr_mm, mask_mm, q):
    n_time, ny, nx = arr_mm.shape

    out = np.full((ny, nx), np.nan, dtype=np.float32)

    for y in range(ny):
        for x in range(nx):
            m = np.asarray(mask_mm[:, y, x]).astype(bool)

            if not m.any():
                continue

            vals = np.asarray(arr_mm[:, y, x])[m]
            vals = vals[np.isfinite(vals)]

            if vals.size > 10:
                out[y, x] = np.nanpercentile(vals, q)

    return out


def summarize_map_by_band(diff_map, bands, model_name, quantile_name):
    rows = []

    for band_name, band_mask in bands.items():
        vals = diff_map[band_mask & np.isfinite(diff_map)]

        if vals.size == 0:
            continue

        rows.append({
            "model": model_name,
            "quantile": quantile_name,
            "band": band_name,
            "n_cells": int(vals.size),
            "mean_difference_model_minus_rcm": float(np.nanmean(vals)),
            "median_difference_model_minus_rcm": float(np.nanmedian(vals)),
            "mean_abs_difference": float(np.nanmean(np.abs(vals))),
            "p10_difference": float(np.nanpercentile(vals, 10)),
            "p90_difference": float(np.nanpercentile(vals, 90)),
        })

    return rows


def plot_extreme_difference_maps(qname, e2_diff, unet_diff, orog, hillshade):
    improvement = np.abs(unet_diff) - np.abs(e2_diff)

    vals_diff = np.concatenate([
        e2_diff[np.isfinite(e2_diff)],
        unet_diff[np.isfinite(unet_diff)],
    ])

    vmax = float(np.nanpercentile(np.abs(vals_diff), 98))
    vmax = max(vmax, 1.0)

    vals_imp = improvement[np.isfinite(improvement)]
    imp_vmax = float(np.nanpercentile(np.abs(vals_imp), 98))
    imp_vmax = max(imp_vmax, 0.5)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 6.8),
        constrained_layout=True,
    )

    fig.suptitle(
        f"{qname.upper()} temperature extreme difference against ALADIN/RCM",
        fontsize=20,
        weight="bold",
        y=1.03,
    )

    panels = [
        ("Model E2 v2 - RCM", e2_diff, "RdBu_r", -vmax, vmax, "model - RCM (°C)"),
        ("U-Net CMIP-CMIP - RCM", unet_diff, "RdBu_r", -vmax, vmax, "model - RCM (°C)"),
        ("E2 improvement vs U-Net", improvement, "RdBu", -imp_vmax, imp_vmax, "|U-Net error| - |E2 error| (°C)"),
    ]

    ims = []

    for ax, (title, arr, cmap, vmin, vmax_, _) in zip(axes, panels):
        add_orography(ax, orog, hillshade, alpha=0.62)

        im = ax.imshow(
            np.ma.masked_invalid(arr),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_,
            alpha=0.80,
        )

        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.18, linewidth=0.4)

        vals = arr[np.isfinite(arr)]
        txt = (
            f"mean: {np.nanmean(vals):.2f}\n"
            f"p50: {np.nanpercentile(vals, 50):.2f}\n"
            f"p90: {np.nanpercentile(vals, 90):.2f}"
        )

        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.84, linewidth=0.5),
        )

        ims.append(im)

    cbar1 = fig.colorbar(ims[1], ax=axes[:2], orientation="vertical", shrink=0.82, pad=0.02)
    cbar1.set_label("model - RCM (°C)", fontsize=11)

    cbar2 = fig.colorbar(ims[2], ax=axes[2], orientation="vertical", shrink=0.82, pad=0.02)
    cbar2.set_label("positive = E2 closer to RCM (°C)", fontsize=11)

    fig.text(
        0.5,
        -0.02,
        "p01 represents cold-tail behavior; p99 represents warm-tail behavior.",
        ha="center",
        va="top",
        fontsize=10.5,
    )

    out = OUT_DIR / f"final_extreme_{qname}_difference_e2_vs_unet.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out)


def compute_extreme_diagnostics(rcm, e2, unet, mask, bands, orog, hillshade):
    print("\nComputing p01/p99 extreme diagnostics")
    print("=" * 100)

    rows = []
    maps_to_save = {}

    for q, qname in [(1, "p01"), (99, "p99")]:
        print(f"Computing {qname} maps")

        rcm_q = compute_percentile_maps(rcm, mask, q)
        e2_q = compute_percentile_maps(e2, mask, q)
        unet_q = compute_percentile_maps(unet, mask, q)

        e2_diff = e2_q - rcm_q
        unet_diff = unet_q - rcm_q

        maps_to_save[f"rcm_{qname}"] = rcm_q
        maps_to_save[f"e2_{qname}"] = e2_q
        maps_to_save[f"unet_{qname}"] = unet_q
        maps_to_save[f"e2_minus_rcm_{qname}"] = e2_diff
        maps_to_save[f"unet_minus_rcm_{qname}"] = unet_diff
        maps_to_save[f"improvement_abs_error_{qname}"] = np.abs(unet_diff) - np.abs(e2_diff)

        rows.extend(summarize_map_by_band(e2_diff, bands, "Model E2 v2", qname))
        rows.extend(summarize_map_by_band(unet_diff, bands, "U-Net CMIP-CMIP", qname))

        plot_extreme_difference_maps(qname, e2_diff, unet_diff, orog, hillshade)

    np.savez_compressed(
        DIAG_DIR / "final_extreme_p01_p99_maps_e2_unet_rcm.npz",
        **maps_to_save,
    )

    df = pd.DataFrame(rows)

    out_csv = DIAG_DIR / "final_extreme_p01_p99_summary_by_band.csv"
    out_json = DIAG_DIR / "final_extreme_p01_p99_summary_by_band.json"

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    return df


# ---------------------------------------------------------------------
# 3. Monthly anomaly correlation
# ---------------------------------------------------------------------

def compute_monthly_climatology(arr_mm, mask_mm, months):
    n_time, ny, nx = arr_mm.shape

    sums = np.zeros((12, ny, nx), dtype=np.float64)
    counts = np.zeros((12, ny, nx), dtype=np.float64)

    chunk = 512

    for start in range(0, n_time, chunk):
        end = min(start + chunk, n_time)

        arr = np.asarray(arr_mm[start:end], dtype=np.float64)
        m0 = np.asarray(mask_mm[start:end]).astype(bool)
        months_chunk = months[start:end]

        for month in range(1, 13):
            sm = months_chunk == month

            if not sm.any():
                continue

            mm = m0[sm]
            vals = np.where(mm, arr[sm], 0.0)

            sums[month - 1] += vals.sum(axis=0)
            counts[month - 1] += mm.sum(axis=0)

    clim = sums / np.where(counts > 0, counts, np.nan)
    return clim.astype(np.float32)


class CorrMapAcc:
    def __init__(self):
        self.n = np.zeros((64, 64), dtype=np.float64)
        self.sum_x = np.zeros((64, 64), dtype=np.float64)
        self.sum_y = np.zeros((64, 64), dtype=np.float64)
        self.sum_x2 = np.zeros((64, 64), dtype=np.float64)
        self.sum_y2 = np.zeros((64, 64), dtype=np.float64)
        self.sum_xy = np.zeros((64, 64), dtype=np.float64)

    def update(self, x, y, m):
        valid = m & np.isfinite(x) & np.isfinite(y)

        self.n += valid.sum(axis=0)
        self.sum_x += np.where(valid, x, 0.0).sum(axis=0)
        self.sum_y += np.where(valid, y, 0.0).sum(axis=0)
        self.sum_x2 += np.where(valid, x ** 2, 0.0).sum(axis=0)
        self.sum_y2 += np.where(valid, y ** 2, 0.0).sum(axis=0)
        self.sum_xy += np.where(valid, x * y, 0.0).sum(axis=0)

    def finalize(self):
        n = np.where(self.n > 0, self.n, np.nan)

        mx = self.sum_x / n
        my = self.sum_y / n

        vx = self.sum_x2 / n - mx ** 2
        vy = self.sum_y2 / n - my ** 2
        cov = self.sum_xy / n - mx * my

        vx = np.where(vx > 0, vx, np.nan)
        vy = np.where(vy > 0, vy, np.nan)

        corr = cov / np.sqrt(vx * vy)
        corr = np.where(np.isfinite(corr), corr, np.nan)

        return corr.astype(np.float32)


def summarize_corr_by_band(corr_map, bands, model_name):
    rows = []

    for band_name, band_mask in bands.items():
        vals = corr_map[band_mask & np.isfinite(corr_map)]

        if vals.size == 0:
            continue

        rows.append({
            "model": model_name,
            "band": band_name,
            "n_cells": int(vals.size),
            "monthly_anomaly_corr_mean": float(np.nanmean(vals)),
            "monthly_anomaly_corr_median": float(np.nanmedian(vals)),
            "monthly_anomaly_corr_p10": float(np.nanpercentile(vals, 10)),
            "monthly_anomaly_corr_p90": float(np.nanpercentile(vals, 90)),
        })

    return rows


def compute_monthly_anomaly_correlation(rcm, e2, unet, mask, months, bands, orog, hillshade):
    print("\nComputing monthly anomaly correlation")
    print("=" * 100)

    print("Computing monthly climatologies")
    clim_rcm = compute_monthly_climatology(rcm, mask, months)
    clim_e2 = compute_monthly_climatology(e2, mask, months)
    clim_unet = compute_monthly_climatology(unet, mask, months)

    n_time = rcm.shape[0]
    chunk = 512

    acc_e2 = CorrMapAcc()
    acc_unet = CorrMapAcc()

    for start in range(0, n_time, chunk):
        end = min(start + chunk, n_time)

        r = np.asarray(rcm[start:end], dtype=np.float64)
        p_e2 = np.asarray(e2[start:end], dtype=np.float64)
        p_unet = np.asarray(unet[start:end], dtype=np.float64)
        m0 = np.asarray(mask[start:end]).astype(bool)

        month_idx = months[start:end] - 1

        r_anom = r - clim_rcm[month_idx]
        e2_anom = p_e2 - clim_e2[month_idx]
        unet_anom = p_unet - clim_unet[month_idx]

        acc_e2.update(e2_anom, r_anom, m0)
        acc_unet.update(unet_anom, r_anom, m0)

        if start % (chunk * 20) == 0:
            print(f"anomaly corr: {start:06d}/{n_time}")

    corr_e2 = acc_e2.finalize()
    corr_unet = acc_unet.finalize()
    corr_diff = corr_e2 - corr_unet

    np.savez_compressed(
        DIAG_DIR / "final_monthly_anomaly_correlation_maps_e2_unet_rcm.npz",
        corr_e2=corr_e2,
        corr_unet=corr_unet,
        corr_diff_e2_minus_unet=corr_diff,
    )

    rows = []
    rows.extend(summarize_corr_by_band(corr_e2, bands, "Model E2 v2"))
    rows.extend(summarize_corr_by_band(corr_unet, bands, "U-Net CMIP-CMIP"))

    df = pd.DataFrame(rows)

    out_csv = DIAG_DIR / "final_monthly_anomaly_correlation_summary_by_band.csv"
    out_json = DIAG_DIR / "final_monthly_anomaly_correlation_summary_by_band.json"

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    plot_anomaly_corr_maps(corr_e2, corr_unet, corr_diff, orog, hillshade)

    return df


def plot_anomaly_corr_maps(corr_e2, corr_unet, corr_diff, orog, hillshade):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 6.8),
        constrained_layout=True,
    )

    fig.suptitle(
        "Monthly anomaly correlation against ALADIN/RCM",
        fontsize=20,
        weight="bold",
        y=1.03,
    )

    panels = [
        ("Model E2 v2", corr_e2, "viridis", 0.5, 1.0, "correlation"),
        ("U-Net CMIP-CMIP", corr_unet, "viridis", 0.5, 1.0, "correlation"),
        ("E2 - U-Net", corr_diff, "RdBu_r", -0.15, 0.15, "correlation difference"),
    ]

    ims = []

    for ax, (title, arr, cmap, vmin, vmax, _) in zip(axes, panels):
        add_orography(ax, orog, hillshade, alpha=0.62)

        im = ax.imshow(
            np.ma.masked_invalid(arr),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.80,
        )

        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.18, linewidth=0.4)

        vals = arr[np.isfinite(arr)]
        txt = (
            f"mean: {np.nanmean(vals):.2f}\n"
            f"p50: {np.nanpercentile(vals, 50):.2f}\n"
            f"p10: {np.nanpercentile(vals, 10):.2f}"
        )

        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.84, linewidth=0.5),
        )

        ims.append(im)

    cbar1 = fig.colorbar(ims[1], ax=axes[:2], orientation="vertical", shrink=0.82, pad=0.02)
    cbar1.set_label("monthly anomaly correlation", fontsize=11)

    cbar2 = fig.colorbar(ims[2], ax=axes[2], orientation="vertical", shrink=0.82, pad=0.02)
    cbar2.set_label("E2 - U-Net", fontsize=11)

    fig.text(
        0.5,
        -0.02,
        "Anomalies are computed after removing each dataset's own monthly climatology over SSP3-7.0.",
        ha="center",
        va="top",
        fontsize=10.5,
    )

    out = OUT_DIR / "final_monthly_anomaly_correlation_e2_vs_unet.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Final advanced diagnostics: seasonal, extremes, anomaly correlation")
    print("=" * 100)

    for p in [E2_CKPT, UNET_PATH, OROG_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    rcm, e2, unet, mask, meta = compute_or_load_cubes()

    n_time = rcm.shape[0]
    years, months, days, seasons, decimal_year = get_time_arrays(n_time)

    orog = load_orography()
    hillshade = make_hillshade(orog)

    land_mask = np.any(np.asarray(mask), axis=0)
    bands = build_bands(orog, land_mask)

    band_counts = {k: int(v.sum()) for k, v in bands.items()}
    (DIAG_DIR / "advanced_diagnostics_band_counts.json").write_text(json.dumps(band_counts, indent=2))

    time_meta = {
        "n_time": int(n_time),
        "year_min": int(np.min(years)),
        "year_max": int(np.max(years)),
        "months_unique": sorted([int(x) for x in np.unique(months)]),
        "seasons_unique": sorted([str(x) for x in np.unique(seasons)]),
    }

    (DIAG_DIR / "advanced_diagnostics_time_metadata.json").write_text(json.dumps(time_meta, indent=2))

    print("\nBand counts")
    print(json.dumps(band_counts, indent=2))

    print("\nTime metadata")
    print(json.dumps(time_meta, indent=2))

    seasonal_df = compute_seasonal_metrics(rcm, e2, unet, mask, seasons, bands)
    plot_seasonal_bars(seasonal_df)

    extreme_df = compute_extreme_diagnostics(rcm, e2, unet, mask, bands, orog, hillshade)

    anomaly_df = compute_monthly_anomaly_correlation(rcm, e2, unet, mask, months, bands, orog, hillshade)

    print("\nDONE")
    print("Figures saved in:")
    print(OUT_DIR)
    print("Diagnostics saved in:")
    print(DIAG_DIR)

    print("\nKey output files:")
    print(DIAG_DIR / "final_seasonal_metrics_by_band.csv")
    print(DIAG_DIR / "final_extreme_p01_p99_summary_by_band.csv")
    print(DIAG_DIR / "final_monthly_anomaly_correlation_summary_by_band.csv")
    print(OUT_DIR / "final_seasonal_rmse_by_band_e2_vs_unet.png")
    print(OUT_DIR / "final_extreme_p01_difference_e2_vs_unet.png")
    print(OUT_DIR / "final_extreme_p99_difference_e2_vs_unet.png")
    print(OUT_DIR / "final_monthly_anomaly_correlation_e2_vs_unet.png")


if __name__ == "__main__":
    main()
