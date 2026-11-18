from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"

CUBE = EXP / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
PAPER = EXP / "paper_validation_final_seed20260813"

TAB = PAPER / "tables"
DIA = PAPER / "diagnostics"

TAB.mkdir(parents=True, exist_ok=True)
DIA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Bootstrap configuration
# ---------------------------------------------------------------------

N_BOOT = 2000
BLOCK_YEARS = 5
SEED = 20260812
BATCH_BOOT = 100

# ---------------------------------------------------------------------
# Load prediction cubes
# ---------------------------------------------------------------------

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape

if N % 365 != 0:
    raise RuntimeError(
        f"Expected 365-day calendar, but N={N} is not divisible by 365"
    )

NY = N // 365

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

common = np.all(mask, axis=0)
cell_idx = np.flatnonzero(common.ravel())
NC = len(cell_idx)

print("=" * 100)
print("PAIRED MOVING-BLOCK BOOTSTRAP")
print("=" * 100)
print("Cube shape:", shape)
print("Years:", NY)
print("Common cells:", NC)
print("Bootstrap replicates:", N_BOOT)
print("Block length:", BLOCK_YEARS, "years")
print("Seed:", SEED)

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def get_year(arr, iy):
    s = iy * 365
    e = s + 365

    x = np.asarray(arr[s:e], dtype=np.float64)
    x = x.reshape(365, -1)[:, cell_idx]

    return x


# ---------------------------------------------------------------------
# Daily climatologies for anomaly calculations
# ---------------------------------------------------------------------

print("\n[1/7] Computing own 365-day climatologies...")

clim_r = np.zeros((365, NC), dtype=np.float64)
clim_e = np.zeros((365, NC), dtype=np.float64)
clim_u = np.zeros((365, NC), dtype=np.float64)

for iy in range(NY):
    clim_r += get_year(rcm, iy)
    clim_e += get_year(e2, iy)
    clim_u += get_year(unet, iy)

clim_r /= NY
clim_e /= NY
clim_u /= NY

# ---------------------------------------------------------------------
# First pass:
# - primary error sufficient statistics
# - raw variance sufficient statistics
# - anomaly trend coefficients
# ---------------------------------------------------------------------

print("[2/7] Building yearly sufficient statistics...")

# Scalar yearly error sufficient stats
sse_e = np.zeros(NY)
sse_u = np.zeros(NY)

sae_e = np.zeros(NY)
sae_u = np.zeros(NY)

se_e = np.zeros(NY)
se_u = np.zeros(NY)

nobs_year = np.full(NY, 365 * NC, dtype=np.float64)

# For variance per grid cell
sum_r = np.zeros((NY, NC), dtype=np.float64)
sum_e = np.zeros((NY, NC), dtype=np.float64)
sum_u = np.zeros((NY, NC), dtype=np.float64)

ss_r = np.zeros((NY, NC), dtype=np.float64)
ss_e = np.zeros((NY, NC), dtype=np.float64)
ss_u = np.zeros((NY, NC), dtype=np.float64)

# Trend estimation for daily anomalies
sum_ty_r = np.zeros(NC, dtype=np.float64)
sum_ty_e = np.zeros(NC, dtype=np.float64)
sum_ty_u = np.zeros(NC, dtype=np.float64)

sum_y_r = np.zeros(NC, dtype=np.float64)
sum_y_e = np.zeros(NC, dtype=np.float64)
sum_y_u = np.zeros(NC, dtype=np.float64)

T = np.arange(N, dtype=np.float64)
sum_t = float(T.sum())
sum_t2 = float((T*T).sum())
t_mean = float(T.mean())

for iy in range(NY):
    r = get_year(rcm, iy)
    e = get_year(e2, iy)
    u = get_year(unet, iy)

    er_e = e - r
    er_u = u - r

    sse_e[iy] = np.sum(er_e * er_e)
    sse_u[iy] = np.sum(er_u * er_u)

    sae_e[iy] = np.sum(np.abs(er_e))
    sae_u[iy] = np.sum(np.abs(er_u))

    se_e[iy] = np.sum(er_e)
    se_u[iy] = np.sum(er_u)

    sum_r[iy] = r.sum(axis=0)
    sum_e[iy] = e.sum(axis=0)
    sum_u[iy] = u.sum(axis=0)

    ss_r[iy] = (r*r).sum(axis=0)
    ss_e[iy] = (e*e).sum(axis=0)
    ss_u[iy] = (u*u).sum(axis=0)

    ar = r - clim_r
    ae = e - clim_e
    au = u - clim_u

    s = iy * 365
    t = np.arange(s, s+365, dtype=np.float64)[:, None]

    sum_ty_r += (t * ar).sum(axis=0)
    sum_ty_e += (t * ae).sum(axis=0)
    sum_ty_u += (t * au).sum(axis=0)

    sum_y_r += ar.sum(axis=0)
    sum_y_e += ae.sum(axis=0)
    sum_y_u += au.sum(axis=0)

    if (iy + 1) % 10 == 0 or iy == NY-1:
        print(f"  year {iy+1}/{NY}")

den_t = sum_t2 - (sum_t**2)/N

slope_r = (sum_ty_r - sum_t*sum_y_r/N) / den_t
slope_e = (sum_ty_e - sum_t*sum_y_e/N) / den_t
slope_u = (sum_ty_u - sum_t*sum_y_u/N) / den_t

mean_y_r = sum_y_r / N
mean_y_e = sum_y_e / N
mean_y_u = sum_y_u / N

# ---------------------------------------------------------------------
# Correlation sufficient statistics
# ---------------------------------------------------------------------

print("[3/7] Building anomaly-correlation sufficient statistics...")

def empty_corr_stats():
    return {
        "sx": np.zeros((NY, NC), dtype=np.float64),
        "sy": np.zeros((NY, NC), dtype=np.float64),
        "sxx": np.zeros((NY, NC), dtype=np.float64),
        "syy": np.zeros((NY, NC), dtype=np.float64),
        "sxy": np.zeros((NY, NC), dtype=np.float64),
    }

corr_e = empty_corr_stats()
corr_u = empty_corr_stats()

corr_dt_e = empty_corr_stats()
corr_dt_u = empty_corr_stats()

for iy in range(NY):
    r = get_year(rcm, iy)
    e = get_year(e2, iy)
    u = get_year(unet, iy)

    ar = r - clim_r
    ae = e - clim_e
    au = u - clim_u

    s = iy * 365
    t = np.arange(s, s+365, dtype=np.float64)[:, None]

    # Full-sample linear detrending
    ar_dt = ar - (
        mean_y_r[None, :] +
        slope_r[None, :] * (t - t_mean)
    )
    ae_dt = ae - (
        mean_y_e[None, :] +
        slope_e[None, :] * (t - t_mean)
    )
    au_dt = au - (
        mean_y_u[None, :] +
        slope_u[None, :] * (t - t_mean)
    )

    for stats, x, y in [
        (corr_e, ar, ae),
        (corr_u, ar, au),
        (corr_dt_e, ar_dt, ae_dt),
        (corr_dt_u, ar_dt, au_dt),
    ]:
        stats["sx"][iy] = x.sum(axis=0)
        stats["sy"][iy] = y.sum(axis=0)
        stats["sxx"][iy] = (x*x).sum(axis=0)
        stats["syy"][iy] = (y*y).sum(axis=0)
        stats["sxy"][iy] = (x*y).sum(axis=0)

    if (iy + 1) % 10 == 0 or iy == NY-1:
        print(f"  year {iy+1}/{NY}")


# ---------------------------------------------------------------------
# Functions to evaluate metrics from year weights
# ---------------------------------------------------------------------

def mean_corr_from_weights(weights, stats):
    """
    weights shape: [B, NY]
    returns: [B]
    """
    sx = weights @ stats["sx"]
    sy = weights @ stats["sy"]
    sxx = weights @ stats["sxx"]
    syy = weights @ stats["syy"]
    sxy = weights @ stats["sxy"]

    n = weights.sum(axis=1)[:, None] * 365.0

    cov = sxy - sx*sy/n
    vx = sxx - sx*sx/n
    vy = syy - sy*sy/n

    with np.errstate(divide="ignore", invalid="ignore"):
        cor = cov / np.sqrt(vx*vy)

    return np.nanmean(cor, axis=1)


def variance_metrics_from_weights(weights):
    """
    returns:
      ratio_mean_e
      ratio_mean_u
      ratio_mae100_e
      ratio_mae100_u
    """
    n = weights.sum(axis=1)[:, None] * 365.0

    sr = weights @ sum_r
    se = weights @ sum_e
    su = weights @ sum_u

    qr = weights @ ss_r
    qe = weights @ ss_e
    qu = weights @ ss_u

    vr = qr/n - (sr/n)**2
    ve = qe/n - (se/n)**2
    vu = qu/n - (su/n)**2

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_e = 100.0 * ve/vr
        ratio_u = 100.0 * vu/vr

    ratio_mean_e = np.nanmean(ratio_e, axis=1)
    ratio_mean_u = np.nanmean(ratio_u, axis=1)

    ratio_mae_e = np.nanmean(np.abs(ratio_e - 100.0), axis=1)
    ratio_mae_u = np.nanmean(np.abs(ratio_u - 100.0), axis=1)

    return ratio_mean_e, ratio_mean_u, ratio_mae_e, ratio_mae_u


# ---------------------------------------------------------------------
# Point estimates using original chronology
# ---------------------------------------------------------------------

print("[4/7] Computing point estimates...")

w_full = np.ones((1, NY), dtype=np.float64)
n_full = float(N * NC)

point_rmse_e = math.sqrt(sse_e.sum()/n_full)
point_rmse_u = math.sqrt(sse_u.sum()/n_full)

point_mae_e = sae_e.sum()/n_full
point_mae_u = sae_u.sum()/n_full

point_bias_e = se_e.sum()/n_full
point_bias_u = se_u.sum()/n_full

point_absbias_e = abs(point_bias_e)
point_absbias_u = abs(point_bias_u)

point_corr_e = mean_corr_from_weights(w_full, corr_e)[0]
point_corr_u = mean_corr_from_weights(w_full, corr_u)[0]

point_corrdt_e = mean_corr_from_weights(w_full, corr_dt_e)[0]
point_corrdt_u = mean_corr_from_weights(w_full, corr_dt_u)[0]

(
    point_var_e,
    point_var_u,
    point_varmad_e,
    point_varmad_u,
) = [x[0] for x in variance_metrics_from_weights(w_full)]

print("RMSE:", point_rmse_e, point_rmse_u)
print("MAE :", point_mae_e, point_mae_u)
print("Bias:", point_bias_e, point_bias_u)
print("Daily anomaly corr:", point_corr_e, point_corr_u)
print("Detrended daily anomaly corr:", point_corrdt_e, point_corrdt_u)
print("Variance ratio mean:", point_var_e, point_var_u)
print("Variance ratio |error from 100|:", point_varmad_e, point_varmad_u)


# ---------------------------------------------------------------------
# Generate circular moving-block bootstrap year weights
# ---------------------------------------------------------------------

print("[5/7] Generating bootstrap samples...")

rng = np.random.default_rng(SEED)

n_blocks = math.ceil(NY / BLOCK_YEARS)

boot_weights = np.zeros((N_BOOT, NY), dtype=np.float64)

for b in range(N_BOOT):
    years = []

    starts = rng.integers(0, NY, size=n_blocks)

    for s in starts:
        years.extend(
            [(s + j) % NY for j in range(BLOCK_YEARS)]
        )

    years = years[:NY]

    boot_weights[b] = np.bincount(
        years,
        minlength=NY
    )

# ---------------------------------------------------------------------
# Bootstrap distributions
# ---------------------------------------------------------------------

print("[6/7] Running bootstrap...")

metrics_boot = {
    "rmse_e": np.zeros(N_BOOT),
    "rmse_u": np.zeros(N_BOOT),
    "mae_e": np.zeros(N_BOOT),
    "mae_u": np.zeros(N_BOOT),
    "absbias_e": np.zeros(N_BOOT),
    "absbias_u": np.zeros(N_BOOT),
    "corr_e": np.zeros(N_BOOT),
    "corr_u": np.zeros(N_BOOT),
    "corrdt_e": np.zeros(N_BOOT),
    "corrdt_u": np.zeros(N_BOOT),
    "varmean_e": np.zeros(N_BOOT),
    "varmean_u": np.zeros(N_BOOT),
    "varmad_e": np.zeros(N_BOOT),
    "varmad_u": np.zeros(N_BOOT),
}

for b0 in range(0, N_BOOT, BATCH_BOOT):
    b1 = min(b0 + BATCH_BOOT, N_BOOT)

    w = boot_weights[b0:b1]

    n = w.sum(axis=1) * (365.0 * NC)

    SSE_E = w @ sse_e
    SSE_U = w @ sse_u

    SAE_E = w @ sae_e
    SAE_U = w @ sae_u

    SE_E = w @ se_e
    SE_U = w @ se_u

    metrics_boot["rmse_e"][b0:b1] = np.sqrt(SSE_E/n)
    metrics_boot["rmse_u"][b0:b1] = np.sqrt(SSE_U/n)

    metrics_boot["mae_e"][b0:b1] = SAE_E/n
    metrics_boot["mae_u"][b0:b1] = SAE_U/n

    metrics_boot["absbias_e"][b0:b1] = np.abs(SE_E/n)
    metrics_boot["absbias_u"][b0:b1] = np.abs(SE_U/n)

    metrics_boot["corr_e"][b0:b1] = mean_corr_from_weights(
        w, corr_e
    )
    metrics_boot["corr_u"][b0:b1] = mean_corr_from_weights(
        w, corr_u
    )

    metrics_boot["corrdt_e"][b0:b1] = mean_corr_from_weights(
        w, corr_dt_e
    )
    metrics_boot["corrdt_u"][b0:b1] = mean_corr_from_weights(
        w, corr_dt_u
    )

    ve, vu, vme, vmu = variance_metrics_from_weights(w)

    metrics_boot["varmean_e"][b0:b1] = ve
    metrics_boot["varmean_u"][b0:b1] = vu
    metrics_boot["varmad_e"][b0:b1] = vme
    metrics_boot["varmad_u"][b0:b1] = vmu

    print(f"  bootstrap {b1}/{N_BOOT}")


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

print("[7/7] Writing summary...")

def ci(x):
    return (
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


def probability_positive(x):
    return float(np.mean(x > 0.0))


rows = []

def add_metric(
    name,
    point_e,
    point_u,
    boot_e,
    boot_u,
    higher_is_better=False,
    improvement_override=None,
    units=""
):
    e_lo, e_hi = ci(boot_e)
    u_lo, u_hi = ci(boot_u)

    if improvement_override is not None:
        improvement = improvement_override
    elif higher_is_better:
        # positive = E2 better
        improvement = boot_e - boot_u
    else:
        # positive = E2 better
        improvement = boot_u - boot_e

    imp_point = (
        point_e - point_u
        if higher_is_better
        else point_u - point_e
    )

    if improvement_override is not None:
        imp_point = float(np.median(improvement_override))

    imp_lo, imp_hi = ci(improvement)

    rows.append({
        "metric": name,
        "units": units,
        "E2_point": point_e,
        "E2_ci2.5": e_lo,
        "E2_ci97.5": e_hi,
        "UNet_point": point_u,
        "UNet_ci2.5": u_lo,
        "UNet_ci97.5": u_hi,
        "E2_advantage_point": imp_point,
        "advantage_ci2.5": imp_lo,
        "advantage_ci97.5": imp_hi,
        "bootstrap_probability_E2_better": probability_positive(
            improvement
        ),
        "robust_95pct": bool(imp_lo > 0 or imp_hi < 0),
    })


add_metric(
    "RMSE_global",
    point_rmse_e,
    point_rmse_u,
    metrics_boot["rmse_e"],
    metrics_boot["rmse_u"],
    higher_is_better=False,
    units="degC"
)

add_metric(
    "MAE_global",
    point_mae_e,
    point_mae_u,
    metrics_boot["mae_e"],
    metrics_boot["mae_u"],
    higher_is_better=False,
    units="degC"
)

add_metric(
    "absolute_bias_global",
    point_absbias_e,
    point_absbias_u,
    metrics_boot["absbias_e"],
    metrics_boot["absbias_u"],
    higher_is_better=False,
    units="degC"
)

add_metric(
    "daily_anomaly_correlation",
    point_corr_e,
    point_corr_u,
    metrics_boot["corr_e"],
    metrics_boot["corr_u"],
    higher_is_better=True,
    units="correlation"
)

add_metric(
    "detrended_daily_anomaly_correlation",
    point_corrdt_e,
    point_corrdt_u,
    metrics_boot["corrdt_e"],
    metrics_boot["corrdt_u"],
    higher_is_better=True,
    units="correlation"
)

# Variance ratio mean itself is descriptive; to test "closer to 100"
# use the mean cell-wise absolute deviation from 100%.
var_improvement = (
    metrics_boot["varmad_u"] -
    metrics_boot["varmad_e"]
)

var_imp_lo, var_imp_hi = ci(var_improvement)

rows.append({
    "metric": "variance_ratio_mean",
    "units": "percent",
    "E2_point": point_var_e,
    "E2_ci2.5": ci(metrics_boot["varmean_e"])[0],
    "E2_ci97.5": ci(metrics_boot["varmean_e"])[1],
    "UNet_point": point_var_u,
    "UNet_ci2.5": ci(metrics_boot["varmean_u"])[0],
    "UNet_ci97.5": ci(metrics_boot["varmean_u"])[1],
    "E2_advantage_point":
        point_varmad_u - point_varmad_e,
    "advantage_ci2.5": var_imp_lo,
    "advantage_ci97.5": var_imp_hi,
    "bootstrap_probability_E2_better":
        probability_positive(var_improvement),
    "robust_95pct":
        bool(var_imp_lo > 0 or var_imp_hi < 0),
})

df = pd.DataFrame(rows)

out_csv = TAB / "05_block_bootstrap_core_summary.csv"
df.to_csv(out_csv, index=False)

# Raw bootstrap distributions for reproducibility
np.savez_compressed(
    DIA / "05_block_bootstrap_core_distributions.npz",
    **metrics_boot,
    boot_weights=boot_weights.astype(np.int16),
)

method = {
    "method":
        "paired circular moving-block bootstrap over complete years",
    "n_bootstrap": N_BOOT,
    "block_length_years": BLOCK_YEARS,
    "calendar_days_per_year": 365,
    "n_years": NY,
    "n_common_cells": NC,
    "seed": SEED,
    "pairing":
        "identical resampled year blocks used for ALADIN/RCM, E2 v2 and U-Net",
    "spatial_structure":
        "all common grid cells retained together within every resampled block",
    "daily_structure":
        "all 365 daily fields retained within each resampled year",
    "confidence_interval":
        "percentile 95% interval [2.5,97.5]",
    "inference":
        "A paired E2 advantage is considered robust when the 95% bootstrap interval does not include zero.",
    "variance_test":
        "Variance-ratio superiority evaluated using mean cell-wise absolute deviation from the ideal 100% ratio.",
    "note":
        "Wasserstein and p01/p99 remain descriptive in this core inferential analysis."
}

with open(DIA / "05_block_bootstrap_core_method.json", "w") as f:
    json.dump(method, f, indent=2)

print("\n" + "="*100)
print(df.to_string(index=False))
print("\nSaved:")
print(out_csv)
print(DIA / "05_block_bootstrap_core_distributions.npz")
print(DIA / "05_block_bootstrap_core_method.json")
