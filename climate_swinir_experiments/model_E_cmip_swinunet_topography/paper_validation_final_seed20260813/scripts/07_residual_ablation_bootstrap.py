from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"

CUBE = (
    EXP
    / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
)

PAPER = EXP / "paper_validation_final_seed20260813"
TAB = PAPER / "tables"
DIA = PAPER / "diagnostics"

TAB.mkdir(parents=True, exist_ok=True)
DIA.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
BLOCK_SIZES = [3, 5, 10]
SEED = 20260813
BATCH_BOOT = 100

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape

if N % 365 != 0:
    raise RuntimeError("Expected 365-day calendar")

NY = N // 365

rcm = np.memmap(
    CUBE / "rcm.dat",
    dtype="float32",
    mode="r",
    shape=shape,
)

phase1 = np.memmap(
    CUBE / "phase1_full_seeded.dat",
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
print("RESIDUAL ABLATION: PHASE1 vs E2 v2")
print("=" * 100)
print("Shape:", shape)
print("Years:", NY)
print("Common cells:", NC)
print("Bootstrap replicates:", N_BOOT)
print("Block sizes:", BLOCK_SIZES)
print("Seed:", SEED)


def get_year(arr, iy):
    s = iy * 365
    e = s + 365
    return (
        np.asarray(arr[s:e], dtype=np.float64)
        .reshape(365, -1)[:, cell_idx]
    )


# ============================================================================
# Daily climatologies
# ============================================================================

print("\n[1/6] Daily climatologies")

clim_r = np.zeros((365, NC), dtype=np.float64)
clim_p = np.zeros_like(clim_r)
clim_e = np.zeros_like(clim_r)

for iy in range(NY):
    clim_r += get_year(rcm, iy)
    clim_p += get_year(phase1, iy)
    clim_e += get_year(e2, iy)

clim_r /= NY
clim_p /= NY
clim_e /= NY


# ============================================================================
# Yearly sufficient statistics
# ============================================================================

print("[2/6] Error / variance sufficient statistics")

sse_p = np.zeros(NY)
sse_e = np.zeros(NY)

sae_p = np.zeros(NY)
sae_e = np.zeros(NY)

se_p = np.zeros(NY)
se_e = np.zeros(NY)

sum_r = np.zeros((NY, NC))
sum_p = np.zeros((NY, NC))
sum_e = np.zeros((NY, NC))

ss_r = np.zeros((NY, NC))
ss_p = np.zeros((NY, NC))
ss_e = np.zeros((NY, NC))

# Trend sufficient statistics for anomaly detrending
sum_ty_r = np.zeros(NC)
sum_ty_p = np.zeros(NC)
sum_ty_e = np.zeros(NC)

sum_y_r = np.zeros(NC)
sum_y_p = np.zeros(NC)
sum_y_e = np.zeros(NC)

T = np.arange(N, dtype=np.float64)
sum_t = T.sum()
sum_t2 = (T*T).sum()
t_mean = T.mean()

for iy in range(NY):
    r = get_year(rcm, iy)
    p = get_year(phase1, iy)
    e = get_year(e2, iy)

    dp = p - r
    de = e - r

    sse_p[iy] = np.sum(dp*dp)
    sse_e[iy] = np.sum(de*de)

    sae_p[iy] = np.sum(np.abs(dp))
    sae_e[iy] = np.sum(np.abs(de))

    se_p[iy] = np.sum(dp)
    se_e[iy] = np.sum(de)

    sum_r[iy] = r.sum(axis=0)
    sum_p[iy] = p.sum(axis=0)
    sum_e[iy] = e.sum(axis=0)

    ss_r[iy] = (r*r).sum(axis=0)
    ss_p[iy] = (p*p).sum(axis=0)
    ss_e[iy] = (e*e).sum(axis=0)

    ar = r - clim_r
    ap = p - clim_p
    ae = e - clim_e

    s = iy * 365
    t = np.arange(s, s+365, dtype=np.float64)[:, None]

    sum_ty_r += (t*ar).sum(axis=0)
    sum_ty_p += (t*ap).sum(axis=0)
    sum_ty_e += (t*ae).sum(axis=0)

    sum_y_r += ar.sum(axis=0)
    sum_y_p += ap.sum(axis=0)
    sum_y_e += ae.sum(axis=0)

den_t = sum_t2 - sum_t**2 / N

slope_r = (sum_ty_r - sum_t*sum_y_r/N) / den_t
slope_p = (sum_ty_p - sum_t*sum_y_p/N) / den_t
slope_e = (sum_ty_e - sum_t*sum_y_e/N) / den_t

mean_y_r = sum_y_r / N
mean_y_p = sum_y_p / N
mean_y_e = sum_y_e / N


# ============================================================================
# Correlation sufficient statistics
# ============================================================================

print("[3/6] Correlation sufficient statistics")


def corr_store():
    return {
        "sx": np.zeros((NY, NC)),
        "sy": np.zeros((NY, NC)),
        "sxx": np.zeros((NY, NC)),
        "syy": np.zeros((NY, NC)),
        "sxy": np.zeros((NY, NC)),
    }


corr_p = corr_store()
corr_e = corr_store()

corr_dt_p = corr_store()
corr_dt_e = corr_store()


def add_corr(stats, iy, x, y):
    stats["sx"][iy] = x.sum(axis=0)
    stats["sy"][iy] = y.sum(axis=0)
    stats["sxx"][iy] = (x*x).sum(axis=0)
    stats["syy"][iy] = (y*y).sum(axis=0)
    stats["sxy"][iy] = (x*y).sum(axis=0)


for iy in range(NY):
    r = get_year(rcm, iy)
    p = get_year(phase1, iy)
    e = get_year(e2, iy)

    ar = r - clim_r
    ap = p - clim_p
    ae = e - clim_e

    s = iy * 365
    t = np.arange(s, s+365, dtype=np.float64)[:, None]

    ar_dt = ar - (
        mean_y_r[None, :] +
        slope_r[None, :] * (t - t_mean)
    )

    ap_dt = ap - (
        mean_y_p[None, :] +
        slope_p[None, :] * (t - t_mean)
    )

    ae_dt = ae - (
        mean_y_e[None, :] +
        slope_e[None, :] * (t - t_mean)
    )

    add_corr(corr_p, iy, ar, ap)
    add_corr(corr_e, iy, ar, ae)

    add_corr(corr_dt_p, iy, ar_dt, ap_dt)
    add_corr(corr_dt_e, iy, ar_dt, ae_dt)


def mean_corr(weights, stats):
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
        c = cov / np.sqrt(vx*vy)

    return np.nanmean(c, axis=1)


def variance_stats(weights):
    n = weights.sum(axis=1)[:, None] * 365.0

    sr = weights @ sum_r
    sp = weights @ sum_p
    se = weights @ sum_e

    qr = weights @ ss_r
    qp = weights @ ss_p
    qe = weights @ ss_e

    vr = qr/n - (sr/n)**2
    vp = qp/n - (sp/n)**2
    ve = qe/n - (se/n)**2

    with np.errstate(divide="ignore", invalid="ignore"):
        rp = 100.0 * vp/vr
        re = 100.0 * ve/vr

    mean_p = np.nanmean(rp, axis=1)
    mean_e = np.nanmean(re, axis=1)

    mad_p = np.nanmean(np.abs(rp - 100.0), axis=1)
    mad_e = np.nanmean(np.abs(re - 100.0), axis=1)

    return mean_p, mean_e, mad_p, mad_e


# ============================================================================
# Point estimates
# ============================================================================

print("[4/6] Point estimates")

wfull = np.ones((1, NY))
nfull = float(N * NC)

point = {}

point["rmse_p"] = np.sqrt(sse_p.sum()/nfull)
point["rmse_e"] = np.sqrt(sse_e.sum()/nfull)

point["mae_p"] = sae_p.sum()/nfull
point["mae_e"] = sae_e.sum()/nfull

point["bias_p"] = se_p.sum()/nfull
point["bias_e"] = se_e.sum()/nfull

point["absbias_p"] = abs(point["bias_p"])
point["absbias_e"] = abs(point["bias_e"])

point["corr_p"] = mean_corr(wfull, corr_p)[0]
point["corr_e"] = mean_corr(wfull, corr_e)[0]

point["corrdt_p"] = mean_corr(wfull, corr_dt_p)[0]
point["corrdt_e"] = mean_corr(wfull, corr_dt_e)[0]

vp, ve, vmp, vme = variance_stats(wfull)

point["varmean_p"] = vp[0]
point["varmean_e"] = ve[0]

point["varmad_p"] = vmp[0]
point["varmad_e"] = vme[0]

for k, v in point.items():
    print(f"{k:15s}: {v}")


# ============================================================================
# Bootstrap
# ============================================================================

print("[5/6] Bootstrap")

rng_master = np.random.default_rng(SEED)

rows = []
all_distributions = {}

def ci(x):
    return (
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


for block_years in BLOCK_SIZES:

    print("\n" + "-"*100)
    print("BLOCK =", block_years, "years")
    print("-"*100)

    # Different reproducible stream per block size
    rng = np.random.default_rng(
        int(rng_master.integers(0, 2**31-1))
    )

    n_blocks = math.ceil(NY / block_years)

    weights = np.zeros((N_BOOT, NY))

    for b in range(N_BOOT):
        years = []

        starts = rng.integers(
            0,
            NY,
            size=n_blocks
        )

        for s in starts:
            years.extend(
                [(s+j) % NY for j in range(block_years)]
            )

        years = years[:NY]

        weights[b] = np.bincount(
            years,
            minlength=NY
        )

    dist = {
        "rmse_p": np.zeros(N_BOOT),
        "rmse_e": np.zeros(N_BOOT),
        "mae_p": np.zeros(N_BOOT),
        "mae_e": np.zeros(N_BOOT),
        "absbias_p": np.zeros(N_BOOT),
        "absbias_e": np.zeros(N_BOOT),
        "corr_p": np.zeros(N_BOOT),
        "corr_e": np.zeros(N_BOOT),
        "corrdt_p": np.zeros(N_BOOT),
        "corrdt_e": np.zeros(N_BOOT),
        "varmean_p": np.zeros(N_BOOT),
        "varmean_e": np.zeros(N_BOOT),
        "varmad_p": np.zeros(N_BOOT),
        "varmad_e": np.zeros(N_BOOT),
    }

    for b0 in range(0, N_BOOT, BATCH_BOOT):
        b1 = min(b0+BATCH_BOOT, N_BOOT)

        w = weights[b0:b1]

        n = w.sum(axis=1) * 365.0 * NC

        SSE_P = w @ sse_p
        SSE_E = w @ sse_e

        SAE_P = w @ sae_p
        SAE_E = w @ sae_e

        SE_P = w @ se_p
        SE_E = w @ se_e

        dist["rmse_p"][b0:b1] = np.sqrt(SSE_P/n)
        dist["rmse_e"][b0:b1] = np.sqrt(SSE_E/n)

        dist["mae_p"][b0:b1] = SAE_P/n
        dist["mae_e"][b0:b1] = SAE_E/n

        dist["absbias_p"][b0:b1] = np.abs(SE_P/n)
        dist["absbias_e"][b0:b1] = np.abs(SE_E/n)

        dist["corr_p"][b0:b1] = mean_corr(
            w, corr_p
        )

        dist["corr_e"][b0:b1] = mean_corr(
            w, corr_e
        )

        dist["corrdt_p"][b0:b1] = mean_corr(
            w, corr_dt_p
        )

        dist["corrdt_e"][b0:b1] = mean_corr(
            w, corr_dt_e
        )

        vmp, vme, madp, made = variance_stats(w)

        dist["varmean_p"][b0:b1] = vmp
        dist["varmean_e"][b0:b1] = vme
        dist["varmad_p"][b0:b1] = madp
        dist["varmad_e"][b0:b1] = made

    comparisons = {
        "RMSE_global": (
            point["rmse_p"],
            point["rmse_e"],
            dist["rmse_p"],
            dist["rmse_e"],
            "lower",
            "degC",
        ),
        "MAE_global": (
            point["mae_p"],
            point["mae_e"],
            dist["mae_p"],
            dist["mae_e"],
            "lower",
            "degC",
        ),
        "absolute_bias_global": (
            point["absbias_p"],
            point["absbias_e"],
            dist["absbias_p"],
            dist["absbias_e"],
            "lower",
            "degC",
        ),
        "daily_anomaly_correlation": (
            point["corr_p"],
            point["corr_e"],
            dist["corr_p"],
            dist["corr_e"],
            "higher",
            "correlation",
        ),
        "detrended_daily_anomaly_correlation": (
            point["corrdt_p"],
            point["corrdt_e"],
            dist["corrdt_p"],
            dist["corrdt_e"],
            "higher",
            "correlation",
        ),
    }

    for name, (
        pp, pe, bp, be, criterion, units
    ) in comparisons.items():

        if criterion == "lower":
            advantage = bp - be
            point_adv = pp - pe
        else:
            advantage = be - bp
            point_adv = pe - pp

        lo, hi = ci(advantage)

        rows.append({
            "block_years": block_years,
            "metric": name,
            "units": units,
            "Phase1_point": pp,
            "E2_point": pe,
            "E2_advantage_point": point_adv,
            "advantage_ci2.5": lo,
            "advantage_ci97.5": hi,
            "bootstrap_probability_E2_better":
                float(np.mean(advantage > 0)),
            "robust_95pct":
                bool(lo > 0),
        })

    # Variance fidelity: lower distance from 100% is better
    advantage = (
        dist["varmad_p"]
        - dist["varmad_e"]
    )

    lo, hi = ci(advantage)

    rows.append({
        "block_years": block_years,
        "metric": "variance_ratio_fidelity",
        "units": "percentage_points",
        "Phase1_point": point["varmean_p"],
        "E2_point": point["varmean_e"],
        "E2_advantage_point":
            point["varmad_p"] - point["varmad_e"],
        "advantage_ci2.5": lo,
        "advantage_ci97.5": hi,
        "bootstrap_probability_E2_better":
            float(np.mean(advantage > 0)),
        "robust_95pct":
            bool(lo > 0),
    })

    for k, v in dist.items():
        all_distributions[
            f"block{block_years}_{k}"
        ] = v


# ============================================================================
# Output
# ============================================================================

print("[6/6] Saving")

df = pd.DataFrame(rows)

OUT = TAB / "07_residual_ablation_bootstrap_summary.csv"
df.to_csv(OUT, index=False)

np.savez_compressed(
    DIA / "07_residual_ablation_bootstrap_distributions.npz",
    **all_distributions,
)

method = {
    "comparison": "Phase1 versus E2 v2",
    "interpretation":
        "Residual-head ablation because all 104 Phase1 base tensors are exactly identical in E2 v2.",
    "phase1_base_tensors": 104,
    "base_tensors_exactly_equal": 104,
    "maximum_base_parameter_difference": 0.0,
    "e2_additional_tensors": 14,
    "additional_component": "residual_head",
    "n_bootstrap": N_BOOT,
    "block_lengths_years": BLOCK_SIZES,
    "bootstrap":
        "paired circular moving-block bootstrap over complete years",
    "n_years": NY,
    "n_common_cells": NC,
    "seed": SEED,
    "confidence_interval":
        "percentile 95% interval",
    "criterion":
        "E2 residual-head advantage considered robust when paired 95% interval is entirely above zero.",
}

with open(
    DIA / "07_residual_ablation_method.json",
    "w"
) as f:
    json.dump(method, f, indent=2)

print("\n" + "="*100)
print(df.to_string(index=False))
print("\nSaved:")
print(OUT)
print(DIA / "07_residual_ablation_bootstrap_distributions.npz")
print(DIA / "07_residual_ablation_method.json")
