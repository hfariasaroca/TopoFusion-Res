from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT

CUBE = EXP / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
OUT = EXP / "analysis"
DIA = OUT / "diagnostics"

DIA.mkdir(parents=True, exist_ok=True)

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape

if shape != (31390, 64, 64):
    raise RuntimeError(f"Unexpected cube shape: {shape}")

if N % 365 != 0:
    raise RuntimeError("Expected an exact 365-day calendar")

NY = N // 365
years = np.arange(2015, 2015 + NY)

if years[-1] != 2100:
    raise RuntimeError(f"Unexpected year range: {years[0]}-{years[-1]}")

print("=" * 100)
print("FINAL SEEDED TREND AND CLIMATE-SIGNAL DIAGNOSTICS")
print("=" * 100)
print("Shape :", shape)
print("Years :", years[0], "->", years[-1])
print("N years:", NY)

rcm = np.memmap(
    CUBE / "rcm.dat",
    dtype="float32",
    mode="r",
    shape=shape,
)

e2 = np.memmap(
    CUBE / "e2_full_seeded.dat",
    dtype="float32",
    mode="r",
    shape=shape,
)

unet = np.memmap(
    CUBE / "unet.dat",
    dtype="float32",
    mode="r",
    shape=shape,
)

mask = np.memmap(
    CUBE / "mask.dat",
    dtype="bool",
    mode="r",
    shape=shape,
)

# Common spatial support valid for every day and every compared product.
common = np.all(mask, axis=0)
n_common = int(common.sum())

print("Common cells:", n_common)

if n_common != 1852:
    raise RuntimeError(f"Expected 1852 common cells, found {n_common}")

# -------------------------------------------------------------------------
# Annual means
# -------------------------------------------------------------------------

annual_rcm = np.full((NY, H, W), np.nan, dtype=np.float64)
annual_e2 = np.full((NY, H, W), np.nan, dtype=np.float64)
annual_unet = np.full((NY, H, W), np.nan, dtype=np.float64)

for iy, year in enumerate(years):
    s = iy * 365
    e = s + 365

    annual_rcm[iy] = np.mean(
        np.asarray(rcm[s:e], dtype=np.float64),
        axis=0,
    )

    annual_e2[iy] = np.mean(
        np.asarray(e2[s:e], dtype=np.float64),
        axis=0,
    )

    annual_unet[iy] = np.mean(
        np.asarray(unet[s:e], dtype=np.float64),
        axis=0,
    )

    if (iy + 1) % 10 == 0 or iy == 0 or iy == NY - 1:
        print(f"Annual means: {year} ({iy+1}/{NY})")

# -------------------------------------------------------------------------
# OLS trend of annual mean temperature, °C / decade
# -------------------------------------------------------------------------

x = years.astype(np.float64)
xc = x - x.mean()
den = np.sum(xc ** 2)

def trend_per_decade(annual):
    slope_per_year = np.sum(
        xc[:, None, None] * annual,
        axis=0,
    ) / den

    trend = slope_per_year * 10.0
    trend[~common] = np.nan
    return trend

trend_r = trend_per_decade(annual_rcm)
trend_e = trend_per_decade(annual_e2)
trend_u = trend_per_decade(annual_unet)

trend_err_e = trend_e - trend_r
trend_err_u = trend_u - trend_r

trend_path = DIA / "04_temperature_trend_maps.npz"

np.savez_compressed(
    trend_path,
    rcm_trend=trend_r,
    e2_trend=trend_e,
    unet_trend=trend_u,
    e2_minus_rcm=trend_err_e,
    unet_minus_rcm=trend_err_u,
)

print("\nSaved:", trend_path)

# -------------------------------------------------------------------------
# Climate-change signal:
# mean(2080-2100) - mean(2015-2035)
# -------------------------------------------------------------------------

early = (years >= 2015) & (years <= 2035)
late = (years >= 2080) & (years <= 2100)

if early.sum() != 21 or late.sum() != 21:
    raise RuntimeError(
        f"Unexpected climate windows: early={early.sum()}, late={late.sum()}"
    )

def climate_signal(annual):
    sig = (
        np.mean(annual[late], axis=0)
        -
        np.mean(annual[early], axis=0)
    )
    sig[~common] = np.nan
    return sig.astype(np.float32)

signal_r = climate_signal(annual_rcm)
signal_e = climate_signal(annual_e2)
signal_u = climate_signal(annual_unet)

signal_err_e = signal_e - signal_r
signal_err_u = signal_u - signal_r

signal_path = DIA / "11_climate_change_signal_maps.npz"

np.savez_compressed(
    signal_path,
    rcm=signal_r,
    e2=signal_e,
    unet=signal_u,
    e2_minus_rcm=signal_err_e,
    unet_minus_rcm=signal_err_u,
)

print("Saved:", signal_path)

# -------------------------------------------------------------------------
# Compact audit
# -------------------------------------------------------------------------

def spatial_stats(a):
    v = np.asarray(a)[common]
    v = v[np.isfinite(v)]
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }

audit = {
    "sources": {
        "rcm": str(CUBE / "rcm.dat"),
        "topofusion_res": str(CUBE / "e2_full_seeded.dat"),
        "unet": str(CUBE / "unet.dat"),
    },
    "period": "2015-2100",
    "calendar": "365-day",
    "n_days": N,
    "n_years": NY,
    "n_common_cells": n_common,
    "trend_definition": "OLS slope of annual mean tas, multiplied by 10 and reported in degC/decade",
    "climate_change_signal_definition": "mean(2080-2100) - mean(2015-2035)",
    "trend_summary": {
        "RCM": spatial_stats(trend_r),
        "TopoFusion_Res": spatial_stats(trend_e),
        "UNet_CMIP_CMIP": spatial_stats(trend_u),
    },
    "climate_signal_summary": {
        "RCM": spatial_stats(signal_r),
        "TopoFusion_Res": spatial_stats(signal_e),
        "UNet_CMIP_CMIP": spatial_stats(signal_u),
    },
}

audit_path = DIA / "37_final_trend_signal_metadata.json"

with open(audit_path, "w") as f:
    json.dump(audit, f, indent=2)

print("Saved:", audit_path)
print("\nDONE")
