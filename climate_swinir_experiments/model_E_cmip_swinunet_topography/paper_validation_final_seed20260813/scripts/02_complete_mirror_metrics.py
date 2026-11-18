from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"

CUBE = EXP / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
DATA = EXP / "data/torch_dataset_model_E_physical_v1"

PAPER = EXP / "paper_validation_final_seed20260813"
FIG = PAPER / "figures/mirror_unet_protocol"
TAB = PAPER / "tables"
DIA = PAPER / "diagnostics"

for p in [FIG, TAB, DIA]:
    p.mkdir(parents=True, exist_ok=True)

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N,H,W = shape

rcm = np.memmap(CUBE/"rcm.dat", dtype="float32", mode="r", shape=shape)
e2 = np.memmap(CUBE/"e2_full_seeded.dat", dtype="float32", mode="r", shape=shape)
unet = np.memmap(CUBE/"unet.dat", dtype="float32", mode="r", shape=shape)
mask = np.memmap(CUBE/"mask.dat", dtype="bool", mode="r", shape=shape)

target_mask = np.load(DATA/"target_valid_mask_64x64.npy").astype(bool)

ds = xr.open_dataset(ROOT/"data_tomas/unet_outputs/tas_cmip_cmip_output.nc")
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

# ------------------------------------------------------------------
# 1. MASK AUDIT
# ------------------------------------------------------------------

print("="*90)
print("MASK AUDIT")
print("="*90)

common_any = np.any(mask, axis=0)
common_all = np.all(mask, axis=0)

print("Target valid cells :", int(target_mask.sum()))
print("Common ANY cells   :", int(common_any.sum()))
print("Common ALL cells   :", int(common_all.sum()))
print("Target lost cells  :", int((target_mask & ~common_any).sum()))

# Determine which cube lacks finite values in those cells
allfinite_r = np.ones((H,W), dtype=bool)
allfinite_e = np.ones((H,W), dtype=bool)
allfinite_u = np.ones((H,W), dtype=bool)

anyfinite_r = np.zeros((H,W), dtype=bool)
anyfinite_e = np.zeros((H,W), dtype=bool)
anyfinite_u = np.zeros((H,W), dtype=bool)

chunk = 365

for s in range(0,N,chunk):
    ee = min(s+chunk,N)

    rr = np.asarray(rcm[s:ee])
    e2b = np.asarray(e2[s:ee])
    ub = np.asarray(unet[s:ee])

    allfinite_r &= np.all(np.isfinite(rr),axis=0)
    allfinite_e &= np.all(np.isfinite(e2b),axis=0)
    allfinite_u &= np.all(np.isfinite(ub),axis=0)

    anyfinite_r |= np.any(np.isfinite(rr),axis=0)
    anyfinite_e |= np.any(np.isfinite(e2b),axis=0)
    anyfinite_u |= np.any(np.isfinite(ub),axis=0)

audit = {
    "target_valid_cells": int(target_mask.sum()),
    "common_any_cells": int(common_any.sum()),
    "common_all_cells": int(common_all.sum()),
    "target_cells_never_common": int((target_mask & ~common_any).sum()),
    "target_cells_rcm_never_finite": int((target_mask & ~anyfinite_r).sum()),
    "target_cells_e2_never_finite": int((target_mask & ~anyfinite_e).sum()),
    "target_cells_unet_never_finite": int((target_mask & ~anyfinite_u).sum()),
    "target_cells_rcm_not_always_finite": int((target_mask & ~allfinite_r).sum()),
    "target_cells_e2_not_always_finite": int((target_mask & ~allfinite_e).sum()),
    "target_cells_unet_not_always_finite": int((target_mask & ~allfinite_u).sum()),
}

print(json.dumps(audit,indent=2))

with open(DIA/"02_common_mask_audit.json","w") as f:
    json.dump(audit,f,indent=2)

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def panel(ax,z,title,cmap="viridis",vmin=None,vmax=None,label=""):
    zz=np.asarray(z,dtype=float).copy()
    zz[~common_any]=np.nan

    im=ax.pcolormesh(
        lon,lat,zz,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb=plt.colorbar(im,ax=ax,shrink=.82)
    cb.set_label(label)


def spatial_error_stats(err):
    v=np.asarray(err)[common_any]
    v=v[np.isfinite(v)]

    return {
        "bias": float(np.mean(v)),
        "mae": float(np.mean(np.abs(v))),
        "rmse": float(np.sqrt(np.mean(v*v))),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v,10)),
        "p90": float(np.percentile(v,90)),
    }

# ------------------------------------------------------------------
# 2. TREND + CLIMATE SIGNAL SPATIAL ERROR METRICS
# ------------------------------------------------------------------

print("\nTREND / CLIMATE SIGNAL SPATIAL ERRORS")

trend=np.load(DIA/"04_temperature_trend_maps.npz")
signal=np.load(DIA/"11_climate_change_signal_maps.npz")

rows=[]

for diagnostic,npz in [
    ("temperature_trend_C_decade",trend),
    ("climate_change_signal_C",signal),
]:
    for model,key in [
        ("E2_v2","e2_minus_rcm"),
        ("UNet_CMIP_CMIP","unet_minus_rcm"),
    ]:
        s=spatial_error_stats(npz[key])
        rows.append({"diagnostic":diagnostic,"model":model,**s})
        print(diagnostic,model,s)

pd.DataFrame(rows).to_csv(
    TAB/"02_trend_climate_signal_spatial_error_metrics.csv",
    index=False
)

# ------------------------------------------------------------------
# 3. TRUE VARIANCE RATIO
# ------------------------------------------------------------------

print("\nTRUE VARIANCE RATIO")

n=np.zeros((H,W),dtype=np.float64)

sr=np.zeros((H,W),dtype=np.float64)
se=np.zeros((H,W),dtype=np.float64)
su=np.zeros((H,W),dtype=np.float64)

srr=np.zeros((H,W),dtype=np.float64)
see=np.zeros((H,W),dtype=np.float64)
suu=np.zeros((H,W),dtype=np.float64)

for s in range(0,N,chunk):
    ee=min(s+chunk,N)

    m=np.asarray(mask[s:ee],dtype=bool)
    rr=np.asarray(rcm[s:ee],dtype=np.float64)
    e2b=np.asarray(e2[s:ee],dtype=np.float64)
    ub=np.asarray(unet[s:ee],dtype=np.float64)

    w=m.astype(np.float64)

    n+=w.sum(axis=0)

    sr+=np.where(m,rr,0).sum(axis=0)
    se+=np.where(m,e2b,0).sum(axis=0)
    su+=np.where(m,ub,0).sum(axis=0)

    srr+=np.where(m,rr*rr,0).sum(axis=0)
    see+=np.where(m,e2b*e2b,0).sum(axis=0)
    suu+=np.where(m,ub*ub,0).sum(axis=0)

mr=sr/n
me=se/n
mu=su/n

vr=srr/n-mr*mr
ve=see/n-me*me
vu=suu/n-mu*mu

ratio_e=100.0*ve/vr
ratio_u=100.0*vu/vr

ratio_e[~common_any]=np.nan
ratio_u[~common_any]=np.nan

improve=np.abs(ratio_u-100)-np.abs(ratio_e-100)

fig,ax=plt.subplots(1,3,figsize=(16,5))

panel(ax[0],ratio_e,"E2 v2","viridis",60,120,"Variance ratio (%)")
panel(ax[1],ratio_u,"U-Net CMIP-CMIP","viridis",60,120,"Variance ratio (%)")

lim=np.nanpercentile(np.abs(improve[common_any]),98)

panel(
    ax[2],improve,
    "Improvement toward 100%\npositive = E2 closer to RCM",
    "RdBu_r",-lim,lim,"percentage points"
)

fig.suptitle("Ratio of variance — daily SSP3-7.0")
fig.tight_layout()
fig.savefig(FIG/"06b_true_ratio_of_variance.png",dpi=180,bbox_inches="tight")
plt.close(fig)

def simple_stats(z):
    v=z[common_any]
    v=v[np.isfinite(v)]
    return {
        "mean":float(np.mean(v)),
        "median":float(np.median(v)),
        "p10":float(np.percentile(v,10)),
        "p90":float(np.percentile(v,90)),
    }

variance_rows=[
    {"model":"E2_v2",**simple_stats(ratio_e)},
    {"model":"UNet_CMIP_CMIP",**simple_stats(ratio_u)},
]

pd.DataFrame(variance_rows).to_csv(
    TAB/"06b_true_variance_ratio_summary.csv",
    index=False
)

np.savez_compressed(
    DIA/"06b_true_variance_ratio_maps.npz",
    e2_variance_ratio_pct=ratio_e,
    unet_variance_ratio_pct=ratio_u,
    improvement_toward_100_pct=improve
)

# ------------------------------------------------------------------
# 4. WASSERSTEIN DISTANCE PER GRID CELL
# exact W1 for equal-size equally weighted samples:
# mean(abs(sort(model)-sort(reference)))
# ------------------------------------------------------------------

print("\nWASSERSTEIN DISTANCE")

wd_e=np.full((H,W),np.nan,dtype=np.float32)
wd_u=np.full((H,W),np.nan,dtype=np.float32)

cells=np.argwhere(common_any)

for k,(iy,ix) in enumerate(cells,1):

    valid=np.asarray(mask[:,iy,ix],dtype=bool)

    rr=np.asarray(rcm[:,iy,ix],dtype=np.float64)[valid]
    ee=np.asarray(e2[:,iy,ix],dtype=np.float64)[valid]
    uu=np.asarray(unet[:,iy,ix],dtype=np.float64)[valid]

    if len(rr)>1:
        rs=np.sort(rr)
        wd_e[iy,ix]=np.mean(np.abs(np.sort(ee)-rs))
        wd_u[iy,ix]=np.mean(np.abs(np.sort(uu)-rs))

    if k%100==0 or k==len(cells):
        print(f"  cell {k}/{len(cells)}")

wd_improve=wd_u-wd_e

vmax=np.nanpercentile(
    np.concatenate([
        wd_e[common_any],
        wd_u[common_any]
    ]),
    98
)

lim=np.nanpercentile(np.abs(wd_improve[common_any]),98)

fig,ax=plt.subplots(1,3,figsize=(16,5))

panel(ax[0],wd_e,"E2 v2","viridis",0,vmax,"Wasserstein distance (°C)")
panel(ax[1],wd_u,"U-Net CMIP-CMIP","viridis",0,vmax,"Wasserstein distance (°C)")
panel(
    ax[2],wd_improve,
    "U-Net - E2\npositive = E2 closer to RCM",
    "RdBu_r",-lim,lim,"Δ Wasserstein (°C)"
)

fig.suptitle("Daily temperature distribution — Wasserstein distance")
fig.tight_layout()

fig.savefig(
    FIG/"07_wasserstein_distance_e2_vs_unet.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

wasserstein_rows=[
    {"model":"E2_v2",**simple_stats(wd_e)},
    {"model":"UNet_CMIP_CMIP",**simple_stats(wd_u)},
    {"model":"UNet_minus_E2",**simple_stats(wd_improve)},
]

pd.DataFrame(wasserstein_rows).to_csv(
    TAB/"07_wasserstein_distance_summary.csv",
    index=False
)

np.savez_compressed(
    DIA/"07_wasserstein_distance_maps.npz",
    e2=wd_e,
    unet=wd_u,
    improvement_unet_minus_e2=wd_improve
)

# ------------------------------------------------------------------
# 5. P01/P99 WITH ORIGINAL U-NET SIGN CONVENTION: RCM - MODEL
# ------------------------------------------------------------------

print("\nP01/P99 original sign convention")

ext=np.load(
    EXP/
    "diagnostics/final_e2_vs_unet_advanced_diagnostics/"
    "final_extreme_p01_p99_maps_e2_unet_rcm.npz"
)

for q in ["p01","p99"]:

    e_diff = -ext[f"e2_minus_rcm_{q}"]
    u_diff = -ext[f"unet_minus_rcm_{q}"]

    lim=np.nanpercentile(
        np.abs(
            np.concatenate([
                e_diff[common_any],
                u_diff[common_any]
            ])
        ),
        98
    )

    fig,ax=plt.subplots(1,2,figsize=(11,5))

    panel(
        ax[0],e_diff,
        f"RCM - E2 v2 ({q})",
        "RdBu_r",-lim,lim,"°C"
    )

    panel(
        ax[1],u_diff,
        f"RCM - U-Net ({q})",
        "RdBu_r",-lim,lim,"°C"
    )

    fig.suptitle(
        f"{q} difference — sign convention matching original U-Net report"
    )

    fig.tight_layout()

    fig.savefig(
        FIG/f"{'08b' if q=='p01' else '09b'}_{q}_RCM_minus_model.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

print("\nDONE")
print("Tables:",TAB)
print("Figures:",FIG)

# BEGIN PUBLICATION FIGURE 6
# ============================================================
# FIGURE 6 — VARIANCE FIDELITY + WASSERSTEIN DISTANCE
#
# Publication-only rendering.
# No diagnostic values are recomputed or modified.
#
# Row 1:
#   variance ratio relative to CNRM-ALADIN64
#
# Row 2:
#   Wasserstein distance to CNRM-ALADIN64
#
# Difference conventions:
#
#   variance fidelity =
#       |VR_U-Net - 100| - |VR_TopoFusion - 100|
#
#   Wasserstein =
#       W_U-Net - W_TopoFusion
#
# Positive values therefore favor TopoFusion-Res.
# ============================================================

from matplotlib.colors import TwoSlopeNorm


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

PUB_FIG = PAPER / "figures" / "final_paper"
PUB_SEL = PAPER / "paper_figures_selected"

PUB_FIG.mkdir(parents=True, exist_ok=True)
PUB_SEL.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Audit definitions
# ------------------------------------------------------------

# Recompute only the algebraic display differences from the
# already-computed diagnostic maps, to make the plotted
# definitions explicit and auditable.
variance_fidelity_delta = (
    np.abs(ratio_u - 100.0)
    -
    np.abs(ratio_e - 100.0)
)

wasserstein_delta = (
    wd_u
    -
    wd_e
)


# Check agreement with the original diagnostic variables.
if not np.allclose(
    variance_fidelity_delta[common_any],
    improve[common_any],
    equal_nan=True,
):
    raise RuntimeError(
        "Variance-fidelity difference does not match "
        "the original diagnostic variable 'improve'."
    )

if not np.allclose(
    wasserstein_delta[common_any],
    wd_improve[common_any],
    equal_nan=True,
):
    raise RuntimeError(
        "Wasserstein difference does not match "
        "the original diagnostic variable 'wd_improve'."
    )


# ------------------------------------------------------------
# Mask helper
# ------------------------------------------------------------

def pub_masked(a):
    x = np.asarray(
        a,
        dtype=np.float64,
    ).copy()

    x[~common_any] = np.nan

    return x


ratio_topo_plot = pub_masked(ratio_e)
ratio_unet_plot = pub_masked(ratio_u)

variance_delta_plot = pub_masked(
    variance_fidelity_delta
)

wd_topo_plot = pub_masked(wd_e)
wd_unet_plot = pub_masked(wd_u)

wd_delta_plot = pub_masked(
    wasserstein_delta
)


# ------------------------------------------------------------
# Publication display ranges
#
# Variance ratio:
# keep the ORIGINAL physically interpretable range 60–120 %.
#
# Difference maps:
# symmetric p98, as in the established spatial-figure style.
#
# Wasserstein absolute maps:
# shared p98 across both models.
#
# Clipping is visual only.
# ------------------------------------------------------------

DISPLAY_PERCENTILE = 98.0

VR_VMIN = 60.0
VR_VMAX = 120.0


vr_diff_values = np.abs(
    variance_fidelity_delta[common_any]
)

vr_diff_values = vr_diff_values[
    np.isfinite(vr_diff_values)
]

VR_DELTA_LIM = float(
    np.percentile(
        vr_diff_values,
        DISPLAY_PERCENTILE,
    )
)


wd_values = np.concatenate([
    wd_e[common_any],
    wd_u[common_any],
])

wd_values = wd_values[
    np.isfinite(wd_values)
]

WD_VMAX = float(
    np.percentile(
        wd_values,
        DISPLAY_PERCENTILE,
    )
)


wd_diff_values = np.abs(
    wasserstein_delta[common_any]
)

wd_diff_values = wd_diff_values[
    np.isfinite(wd_diff_values)
]

WD_DELTA_LIM = float(
    np.percentile(
        wd_diff_values,
        DISPLAY_PERCENTILE,
    )
)


VR_DELTA_LIM = max(
    VR_DELTA_LIM,
    1e-8,
)

WD_VMAX = max(
    WD_VMAX,
    1e-8,
)

WD_DELTA_LIM = max(
    WD_DELTA_LIM,
    1e-8,
)


# ------------------------------------------------------------
# Console audit
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("PUBLICATION FIGURE 6")
print("=" * 90)

print(
    f"Variance ratio scale : "
    f"{VR_VMIN:.1f} .. {VR_VMAX:.1f} %"
)

print(
    f"Variance delta p98   : "
    f"{-VR_DELTA_LIM:.6f} .. "
    f"{VR_DELTA_LIM:.6f} percentage points"
)

print(
    f"Wasserstein p98      : "
    f"0.000000 .. {WD_VMAX:.6f} °C"
)

print(
    f"Wasserstein delta p98: "
    f"{-WD_DELTA_LIM:.6f} .. "
    f"{WD_DELTA_LIM:.6f} °C"
)


# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


mean_lat = float(
    np.nanmean(
        lat[common_any]
    )
)

MAP_ASPECT = float(
    1.0
    /
    np.cos(
        np.deg2rad(
            mean_lat
        )
    )
)


def format_pub_axis(
    ax,
    show_ylabel,
):
    ax.set_xlabel(
        "Longitude"
    )

    if show_ylabel:
        ax.set_ylabel(
            "Latitude"
        )
    else:
        ax.set_ylabel(
            ""
        )

    ax.set_aspect(
        MAP_ASPECT
    )

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8,
    )

    ax.grid(False)


# ------------------------------------------------------------
# Colormaps
# ------------------------------------------------------------

cmap_seq = plt.get_cmap(
    "viridis"
).copy()

cmap_seq.set_bad(
    "white"
)


cmap_diff = plt.get_cmap(
    "RdBu_r"
).copy()

cmap_diff.set_bad(
    "white"
)


# ------------------------------------------------------------
# Layout:
#
# row 1:
# a | b | shared CB | gap | c | delta CB
#
# row 2:
# d | e | shared CB | gap | f | delta CB
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(
        14.8,
        9.2,
    )
)

gs = fig.add_gridspec(
    2,
    6,

    width_ratios=[
        1.0,
        1.0,
        0.045,
        0.10,
        1.0,
        0.045,
    ],

    height_ratios=[
        1.0,
        1.0,
    ],

    left=0.055,
    right=0.965,
    bottom=0.075,
    top=0.94,

    hspace=0.28,
    wspace=0.20,
)


# ============================================================
# ROW 1 — VARIANCE RATIO
# ============================================================

ax_a = fig.add_subplot(
    gs[0, 0]
)

ax_b = fig.add_subplot(
    gs[0, 1]
)

cax_vr = fig.add_subplot(
    gs[0, 2]
)

gap1 = fig.add_subplot(
    gs[0, 3]
)

gap1.axis(
    "off"
)

ax_c = fig.add_subplot(
    gs[0, 4]
)

cax_vrd = fig.add_subplot(
    gs[0, 5]
)


im_a = ax_a.pcolormesh(
    lon,
    lat,
    ratio_topo_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_seq,

    vmin=VR_VMIN,
    vmax=VR_VMAX,
)

ax_a.set_title(
    "(a) TopoFusion-Res"
)

format_pub_axis(
    ax_a,
    show_ylabel=True,
)


im_b = ax_b.pcolormesh(
    lon,
    lat,
    ratio_unet_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_seq,

    vmin=VR_VMIN,
    vmax=VR_VMAX,
)

ax_b.set_title(
    "(b) U-Net CMIP-CMIP"
)

format_pub_axis(
    ax_b,
    show_ylabel=False,
)


cb_vr = fig.colorbar(
    im_b,
    cax=cax_vr,
    extend="both",
    extendfrac=0.035,
)

cb_vr.set_label(
    "Variance ratio (%)"
)


vr_norm = TwoSlopeNorm(
    vmin=-VR_DELTA_LIM,
    vcenter=0.0,
    vmax=VR_DELTA_LIM,
)

im_c = ax_c.pcolormesh(
    lon,
    lat,
    variance_delta_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_diff,
    norm=vr_norm,
)

ax_c.set_title(
    r"(c) $\Delta$ variance fidelity"
)

format_pub_axis(
    ax_c,
    show_ylabel=False,
)


cb_vrd = fig.colorbar(
    im_c,
    cax=cax_vrd,
    extend="both",
    extendfrac=0.035,
)

cb_vrd.set_label(
    "Percentage points"
)


# ============================================================
# ROW 2 — WASSERSTEIN DISTANCE
# ============================================================

ax_d = fig.add_subplot(
    gs[1, 0]
)

ax_e = fig.add_subplot(
    gs[1, 1]
)

cax_w = fig.add_subplot(
    gs[1, 2]
)

gap2 = fig.add_subplot(
    gs[1, 3]
)

gap2.axis(
    "off"
)

ax_f = fig.add_subplot(
    gs[1, 4]
)

cax_wd = fig.add_subplot(
    gs[1, 5]
)


im_d = ax_d.pcolormesh(
    lon,
    lat,
    wd_topo_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_seq,

    vmin=0.0,
    vmax=WD_VMAX,
)

ax_d.set_title(
    "(d) TopoFusion-Res"
)

format_pub_axis(
    ax_d,
    show_ylabel=True,
)


im_e = ax_e.pcolormesh(
    lon,
    lat,
    wd_unet_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_seq,

    vmin=0.0,
    vmax=WD_VMAX,
)

ax_e.set_title(
    "(e) U-Net CMIP-CMIP"
)

format_pub_axis(
    ax_e,
    show_ylabel=False,
)


cb_w = fig.colorbar(
    im_e,
    cax=cax_w,
    extend="max",
    extendfrac=0.035,
)

cb_w.set_label(
    r"Wasserstein distance ($^\circ$C)"
)


wd_norm = TwoSlopeNorm(
    vmin=-WD_DELTA_LIM,
    vcenter=0.0,
    vmax=WD_DELTA_LIM,
)

im_f = ax_f.pcolormesh(
    lon,
    lat,
    wd_delta_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_diff,
    norm=wd_norm,
)

ax_f.set_title(
    r"(f) $\Delta$ Wasserstein distance"
)

format_pub_axis(
    ax_f,
    show_ylabel=False,
)


cb_wd = fig.colorbar(
    im_f,
    cax=cax_wd,
    extend="both",
    extendfrac=0.035,
)

cb_wd.set_label(
    r"$\Delta$ Wasserstein ($^\circ$C)"
)


# ------------------------------------------------------------
# Figure title
# ------------------------------------------------------------

fig.suptitle(
    "Variance fidelity and distributional distance under SSP3-7.0",
    fontsize=13,
    y=0.985,
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

figure6_png = (
    PUB_FIG
    / "Figure06_variance_wasserstein_FINAL.png"
)

figure6_pdf = (
    PUB_FIG
    / "Figure06_variance_wasserstein_FINAL.pdf"
)


fig.savefig(
    figure6_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    figure6_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(
    fig
)


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

figure6_meta = {
    "figure": "Figure 6",

    "support": {
        "n_common_cells":
            int(
                common_any.sum()
            ),
    },

    "variance_ratio": {
        "definition":
            "100 * model variance / CNRM-ALADIN64 variance",

        "fidelity_difference":
            (
                "abs(U-Net variance ratio - 100) minus "
                "abs(TopoFusion-Res variance ratio - 100)"
            ),

        "positive_difference_means":
            "TopoFusion-Res is closer to the ideal variance ratio of 100%.",

        "display_range_pct": [
            VR_VMIN,
            VR_VMAX,
        ],

        "difference_display_percentile":
            DISPLAY_PERCENTILE,

        "difference_limit_percentage_points":
            VR_DELTA_LIM,
    },

    "wasserstein": {
        "difference_definition":
            (
                "U-Net Wasserstein distance minus "
                "TopoFusion-Res Wasserstein distance"
            ),

        "positive_difference_means":
            "TopoFusion-Res has lower Wasserstein distance.",

        "absolute_map_display_percentile":
            DISPLAY_PERCENTILE,

        "absolute_map_vmax_C":
            WD_VMAX,

        "difference_display_percentile":
            DISPLAY_PERCENTILE,

        "difference_limit_C":
            WD_DELTA_LIM,
    },

    "visualization_note":
        (
            "Percentile-based limits are used only for visualization. "
            "Diagnostic arrays and reported statistics are unchanged."
        ),
}


figure6_json = (
    PUB_FIG
    / "Figure06_variance_wasserstein_FINAL_metadata.json"
)

with open(
    figure6_json,
    "w",
) as f:
    json.dump(
        figure6_meta,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Copy final candidate
# ------------------------------------------------------------

import shutil

for src in [
    figure6_png,
    figure6_pdf,
    figure6_json,
]:
    shutil.copy2(
        src,
        PUB_SEL / src.name,
    )


print("\nSaved publication Figure 6:")
print(figure6_png)
print(figure6_pdf)
print(figure6_json)

print("\nCopied to:")
print(PUB_SEL)

print("=" * 90)
print("END PUBLICATION FIGURE 6")
print("=" * 90)

# END PUBLICATION FIGURE 6
