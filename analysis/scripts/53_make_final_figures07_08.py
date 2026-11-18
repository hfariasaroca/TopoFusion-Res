from pathlib import Path
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
FINAL = SCRIPT.parents[1]
EXP = FINAL.parent
DIA = FINAL / "diagnostics"

RMSE_FILE = DIA / "38_final_seeded_spatial_rmse_maps.npz"

PERSISTENCE_FILE = (
    DIA
    / "03_daily_persistence_maps.npz"
)

TREND_FILE = (
    DIA
    / "04_temperature_trend_maps.npz"
)

SIGNAL_FILE = (
    DIA
    / "11_climate_change_signal_maps.npz"
)

DATASET_DIR = (
    EXP
    / "data"
    / "torch_dataset_model_E_physical_v1"
)

STATIC_FILE = (
    DATASET_DIR
    / "static_features_64x64.npy"
)

STATIC_NAMES_FILE = (
    DATASET_DIR
    / "static_feature_names_64x64.json"
)

STATS_FILE = (
    DATASET_DIR
    / "normalization_stats.json"
)

OUT = (
    FINAL
    / "final_publication_figures"
)

META_OUT = (
    FINAL
    / "final_publication_metadata"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

META_OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT AUDIT
# ============================================================

for p in [
    RMSE_FILE,
    PERSISTENCE_FILE,
    TREND_FILE,
    SIGNAL_FILE,
    STATIC_FILE,
    STATIC_NAMES_FILE,
    STATS_FILE,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",

    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.3,

    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,

    "axes.linewidth": 0.7,

    "figure.dpi": 150,
    "savefig.dpi": 400,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# COMMON FINAL MASK + COORDINATES
# ============================================================

with np.load(
    RMSE_FILE,
    allow_pickle=False,
) as z:

    valid = np.asarray(
        z["valid"]
    ).astype(bool)

    LAT = np.asarray(
        z["lat"],
        dtype=np.float64,
    )

    LON = np.asarray(
        z["lon"],
        dtype=np.float64,
    )


if valid.shape != (64, 64):
    raise RuntimeError(
        f"Expected 64x64 support, got {valid.shape}"
    )

if LAT.shape != valid.shape:
    raise RuntimeError(
        f"LAT mismatch: {LAT.shape}"
    )

if LON.shape != valid.shape:
    raise RuntimeError(
        f"LON mismatch: {LON.shape}"
    )


print("=" * 90)
print("COMMON SUPPORT")
print("=" * 90)

print(
    "Common cells:",
    int(valid.sum()),
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def masked(a, mask):
    x = np.asarray(
        a,
        dtype=np.float64,
    ).copy()

    x[
        ~mask
    ] = np.nan

    return x


def values(a, mask):
    x = np.asarray(
        a,
        dtype=np.float64,
    )[mask]

    return x[
        np.isfinite(x)
    ]


def symmetric_p98(a, mask):
    v = np.abs(
        values(a, mask)
    )

    return max(
        float(
            np.percentile(v, 98)
        ),
        1e-12,
    )


def display_range(*arrays, mask):
    v = np.concatenate([
        values(a, mask)
        for a in arrays
    ])

    lo = float(
        np.percentile(v, 2)
    )

    hi = float(
        np.percentile(v, 98)
    )

    if np.isclose(lo, hi):
        lo = float(np.min(v))
        hi = float(np.max(v))

    return lo, hi


def pick_map(
    npz,
    *,
    exact=(),
    token_groups=(),
    exclude=(),
):
    """
    Select one 64x64 diagnostic map without assuming one exact
    key name. Exact names have priority.
    """

    files = list(npz.files)

    for key in exact:

        if key not in files:
            continue

        a = np.asarray(
            npz[key]
        )

        if a.shape == valid.shape:
            return key, a.astype(
                np.float64
            )


    for group in token_groups:

        for key in files:

            kl = key.lower()

            if any(
                x in kl
                for x in exclude
            ):
                continue

            if not all(
                x in kl
                for x in group
            ):
                continue

            a = np.asarray(
                npz[key]
            )

            if a.shape == valid.shape:

                return key, a.astype(
                    np.float64
                )


    raise RuntimeError(
        "\nCould not identify required 64x64 map.\n"
        f"Available keys:\n  {files}\n"
        f"Exact candidates:\n  {exact}\n"
        f"Token groups:\n  {token_groups}\n"
    )


# ============================================================
# PHYSICAL OROGRAPHY
# ============================================================

with open(
    STATIC_NAMES_FILE,
    "r",
) as f:

    static_names = json.load(f)


if len(static_names) != 6:
    raise RuntimeError(
        f"Unexpected static features: {static_names}"
    )


if (
    "orog" not in str(static_names[0]).lower()
    and "elev" not in str(static_names[0]).lower()
):
    raise RuntimeError(
        "Fine-static channel 0 is not orography."
    )


static = np.load(
    STATIC_FILE
).astype(
    np.float64
)


if static.shape != (
    6,
    64,
    64,
):
    raise RuntimeError(
        f"Unexpected static shape: {static.shape}"
    )


with open(
    STATS_FILE,
    "r",
) as f:

    stats = json.load(f)


orog_stats = (
    stats[
        "static_64x64"
    ][
        "orography_64"
    ]
)


orog_mean = float(
    orog_stats["mean"]
)

orog_std = float(
    orog_stats["std"]
)


orography = (
    static[0]
    * orog_std
    + orog_mean
)


orography[
    (orography < 0)
    & (orography > -1e-2)
] = 0.0


orography_plot = np.asarray(
    orography,
    dtype=np.float64,
).copy()

orography_plot[
    ~valid
] = np.nan


CONTOUR_LEVELS = [
    500.0,
    1500.0,
    3000.0,
]


def add_topography(ax):

    return ax.contour(
        LON,
        LAT,
        orography_plot,

        levels=CONTOUR_LEVELS,

        colors="#59666b",

        linewidths=0.62,

        alpha=0.82,

        zorder=10,
    )


# ============================================================
# GEOGRAPHIC DISPLAY
# ============================================================

mean_lat = float(
    np.nanmean(
        LAT[valid]
    )
)


MAP_ASPECT = float(
    1.0
    /
    np.cos(
        np.deg2rad(mean_lat)
    )
)


def format_map(
    ax,
    show_ylabel=False,
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
        length=3.0,
        width=0.7,
    )

    ax.grid(False)


# ============================================================
# ============================================================
# FIGURE 7
# LAG-1 AUTOCORRELATION OF DAILY TEMPERATURE ANOMALIES
# ============================================================
# ============================================================

print()
print("=" * 90)
print("FIGURE 7 — DAILY PERSISTENCE")
print("=" * 90)


with np.load(
    PERSISTENCE_FILE,
    allow_pickle=False,
) as z:

    print(
        "Persistence keys:",
        z.files,
    )


    rcm_key, rho_rcm = pick_map(
        z,

        exact=(
            "rcm_lag1",
            "lag1_rcm",
            "aladin_lag1",
            "lag1_aladin",
            "target_lag1",
            "rho1_rcm",
            "rcm",
        ),

        token_groups=(
            ("rcm", "lag"),
            ("aladin", "lag"),
            ("target", "lag"),
            ("rcm", "autocorr"),
            ("aladin", "autocorr"),
        ),

        exclude=(
            "warm",
            "cold",
            "duration",
            "threshold",
            "diff",
            "error",
        ),
    )


    topo_key, rho_topo = pick_map(
        z,

        exact=(
            "e2_lag1",
            "lag1_e2",
            "topofusion_lag1",
            "lag1_topofusion",
            "res_lag1",
            "rho1_e2",
            "e2",
        ),

        token_groups=(
            ("e2", "lag"),
            ("topofusion", "lag"),
            ("res", "lag"),
            ("e2", "autocorr"),
            ("topofusion", "autocorr"),
        ),

        exclude=(
            "unet",
            "warm",
            "cold",
            "duration",
            "threshold",
            "diff",
            "error",
        ),
    )


    unet_key, rho_unet = pick_map(
        z,

        exact=(
            "unet_lag1",
            "lag1_unet",
            "rho1_unet",
            "unet",
        ),

        token_groups=(
            ("unet", "lag"),
            ("unet", "autocorr"),
        ),

        exclude=(
            "warm",
            "cold",
            "duration",
            "threshold",
            "diff",
            "error",
        ),
    )


valid7 = (
    valid
    & np.isfinite(rho_rcm)
    & np.isfinite(rho_topo)
    & np.isfinite(rho_unet)
)


rho_rcm = masked(
    rho_rcm,
    valid7,
)

rho_topo = masked(
    rho_topo,
    valid7,
)

rho_unet = masked(
    rho_unet,
    valid7,
)


# Positive means TopoFusion-Res is closer to RCM.
persistence_diff = (
    np.abs(
        rho_unet
        - rho_rcm
    )
    -
    np.abs(
        rho_topo
        - rho_rcm
    )
)


rho_lo, rho_hi = display_range(
    rho_rcm,
    rho_topo,
    rho_unet,
    mask=valid7,
)


rho_lo = max(
    -1.0,
    rho_lo,
)

rho_hi = min(
    1.0,
    rho_hi,
)


persistence_diff_lim = symmetric_p98(
    persistence_diff,
    valid7,
)


rcm_mean = float(
    np.mean(
        values(
            rho_rcm,
            valid7,
        )
    )
)

topo_mean = float(
    np.mean(
        values(
            rho_topo,
            valid7,
        )
    )
)

unet_mean = float(
    np.mean(
        values(
            rho_unet,
            valid7,
        )
    )
)


topo_abs_error = np.abs(
    rho_topo
    - rho_rcm
)

unet_abs_error = np.abs(
    rho_unet
    - rho_rcm
)


topo_mae_rho = float(
    np.mean(
        values(
            topo_abs_error,
            valid7,
        )
    )
)

unet_mae_rho = float(
    np.mean(
        values(
            unet_abs_error,
            valid7,
        )
    )
)


pdiff_vals = values(
    persistence_diff,
    valid7,
)


pdiff_mean = float(
    np.mean(
        pdiff_vals
    )
)

pdiff_positive = float(
    np.mean(
        pdiff_vals > 0
    )
)


print(
    "RCM key  :",
    rcm_key,
)

print(
    "Topo key :",
    topo_key,
)

print(
    "U-Net key:",
    unet_key,
)

print()

print(
    "Mean rho1 RCM :",
    rcm_mean,
)

print(
    "Mean rho1 Topo:",
    topo_mean,
)

print(
    "Mean rho1 U   :",
    unet_mean,
)

print(
    "Mean |error| Topo:",
    topo_mae_rho,
)

print(
    "Mean |error| U-Net:",
    unet_mae_rho,
)


cmap_rho = plt.get_cmap(
    "viridis"
).copy()

cmap_rho.set_bad(
    "white"
)


cmap_diff = plt.get_cmap(
    "RdBu_r"
).copy()

cmap_diff.set_bad(
    "white"
)


fig = plt.figure(
    figsize=(
        8.6,
        3.35,
    )
)


gs = fig.add_gridspec(
    2,
    4,

    height_ratios=[
        1.0,
        0.075,
    ],

    left=0.055,
    right=0.985,

    bottom=0.13,
    top=0.94,

    wspace=0.16,
    hspace=0.15,
)


ax0 = fig.add_subplot(
    gs[0, 0]
)

ax1 = fig.add_subplot(
    gs[0, 1]
)

ax2 = fig.add_subplot(
    gs[0, 2]
)

ax3 = fig.add_subplot(
    gs[0, 3]
)


im0 = ax0.pcolormesh(
    LON,
    LAT,
    rho_rcm,

    shading="auto",
    rasterized=True,

    cmap=cmap_rho,

    vmin=rho_lo,
    vmax=rho_hi,
)

ax0.set_title(
    "(a) CNRM-ALADIN64",
    fontweight="bold",
)

format_map(
    ax0,
    show_ylabel=True,
)

add_topography(
    ax0
)


im1 = ax1.pcolormesh(
    LON,
    LAT,
    rho_topo,

    shading="auto",
    rasterized=True,

    cmap=cmap_rho,

    vmin=rho_lo,
    vmax=rho_hi,
)

ax1.set_title(
    "(b) TopoFusion-Res",
    fontweight="bold",
)

format_map(
    ax1
)

add_topography(
    ax1
)


im2 = ax2.pcolormesh(
    LON,
    LAT,
    rho_unet,

    shading="auto",
    rasterized=True,

    cmap=cmap_rho,

    vmin=rho_lo,
    vmax=rho_hi,
)

ax2.set_title(
    "(c) U-Net CMIP-CMIP",
    fontweight="bold",
)

format_map(
    ax2
)

add_topography(
    ax2
)


norm7 = TwoSlopeNorm(
    vmin=-persistence_diff_lim,
    vcenter=0.0,
    vmax=persistence_diff_lim,
)


im3 = ax3.pcolormesh(
    LON,
    LAT,
    persistence_diff,

    shading="auto",
    rasterized=True,

    cmap=cmap_diff,

    norm=norm7,
)

ax3.set_title(
    "(d) Difference in absolute error",
    fontweight="bold",
)

format_map(
    ax3
)

add_topography(
    ax3
)


ax3.text(
    0.035,
    0.035,

    (
        f"mean ρ₁: RCM {rcm_mean:.3f}\n"
        f"Topo {topo_mean:.3f} · U-Net {unet_mean:.3f}"
    ),

    transform=ax3.transAxes,

    fontsize=5.8,

    ha="left",
    va="bottom",

    bbox=dict(
        boxstyle="round,pad=0.18",

        facecolor="white",
        edgecolor="#666666",

        linewidth=0.4,
        alpha=0.86,
    ),

    zorder=30,
)


cax7a = fig.add_subplot(
    gs[
        1,
        0:3
    ]
)

cax7b = fig.add_subplot(
    gs[
        1,
        3
    ]
)


cb7a = fig.colorbar(
    im2,

    cax=cax7a,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)

cb7a.set_label(
    r"Lag-1 autocorrelation, $\rho_1$"
)


cb7b = fig.colorbar(
    im3,

    cax=cax7b,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)

cb7b.set_label(
    r"$|e_{\rm U-Net}|-|e_{\rm Topo}|$"
)


FIG7_PNG = (
    OUT
    / "Figure07_lag1_persistence.png"
)

FIG7_PDF = (
    OUT
    / "Figure07_lag1_persistence.pdf"
)


fig.savefig(
    FIG7_PNG,

    dpi=400,

    bbox_inches="tight",

    facecolor="white",
)

fig.savefig(
    FIG7_PDF,

    bbox_inches="tight",

    facecolor="white",
)

plt.close(fig)


meta7 = {

    "figure":
        "Figure 7",

    "quantity":
        "Lag-1 autocorrelation of daily temperature anomalies",

    "support_cells":
        int(
            valid7.sum()
        ),

    "keys": {
        "reference":
            rcm_key,

        "topofusion":
            topo_key,

        "unet":
            unet_key,
    },

    "mean_rho1": {
        "CNRM-ALADIN64":
            rcm_mean,

        "TopoFusion-Res":
            topo_mean,

        "U-Net CMIP-CMIP":
            unet_mean,
    },

    "mean_absolute_error_against_reference": {
        "TopoFusion-Res":
            topo_mae_rho,

        "U-Net CMIP-CMIP":
            unet_mae_rho,
    },

    "difference_definition":
        (
            "abs(U-Net - RCM) minus "
            "abs(TopoFusion-Res - RCM). "
            "Positive values indicate that "
            "TopoFusion-Res is locally closer."
        ),

    "difference_mean":
        pdiff_mean,

    "fraction_positive":
        pdiff_positive,

    "topographic_contours_m": [
        500,
        1500,
        3000,
    ],

    "outputs": {
        "png":
            str(FIG7_PNG),

        "pdf":
            str(FIG7_PDF),
    },
}


with open(
    META_OUT
    / "Figure07_lag1_persistence_metadata.json",
    "w",
) as f:

    json.dump(
        meta7,
        f,
        indent=2,
    )


# ============================================================
# ============================================================
# FIGURE 8
# LONG-TERM TREND + CLIMATE CHANGE SIGNAL
# ============================================================
# ============================================================

print()
print("=" * 90)
print("FIGURE 8 — LONG-TERM CLIMATE RESPONSE")
print("=" * 90)


# ============================================================
# TREND MAPS
# ============================================================

with np.load(
    TREND_FILE,
    allow_pickle=False,
) as z:

    print(
        "Trend keys:",
        z.files,
    )


    trend_rcm_key, trend_rcm = pick_map(
        z,

        exact=(
            "rcm_trend",
            "trend_rcm",
            "aladin_trend",
            "trend_aladin",
            "target_trend",
        ),

        token_groups=(
            ("rcm", "trend"),
            ("aladin", "trend"),
            ("target", "trend"),
        ),

        exclude=(
            "error",
            "bias",
            "diff",
            "pvalue",
            "p_value",
        ),
    )


    trend_topo_key, trend_topo = pick_map(
        z,

        exact=(
            "e2_trend",
            "trend_e2",
            "topofusion_trend",
            "trend_topofusion",
            "res_trend",
        ),

        token_groups=(
            ("e2", "trend"),
            ("topofusion", "trend"),
            ("res", "trend"),
        ),

        exclude=(
            "unet",
            "error",
            "bias",
            "diff",
            "pvalue",
            "p_value",
        ),
    )


    trend_unet_key, trend_unet = pick_map(
        z,

        exact=(
            "unet_trend",
            "trend_unet",
        ),

        token_groups=(
            ("unet", "trend"),
        ),

        exclude=(
            "error",
            "bias",
            "diff",
            "pvalue",
            "p_value",
        ),
    )


# ============================================================
# CLIMATE-CHANGE SIGNAL MAPS
# ============================================================

with np.load(
    SIGNAL_FILE,
    allow_pickle=False,
) as z:

    print()
    print(
        "Signal keys:",
        z.files,
    )


    signal_rcm_key, signal_rcm = pick_map(
        z,

        exact=(
            "rcm",
            "rcm_signal",
            "signal_rcm",
            "aladin_signal",
            "target_signal",
            "rcm_climate_signal",
        ),

        token_groups=(
            ("rcm", "signal"),
            ("aladin", "signal"),
            ("target", "signal"),
            ("rcm", "change"),
            ("aladin", "change"),
        ),

        exclude=(
            "error",
            "bias",
            "diff",
        ),
    )


    signal_topo_key, signal_topo = pick_map(
        z,

        exact=(
            "e2",
            "e2_signal",
            "signal_e2",
            "topofusion_signal",
            "res_signal",
            "e2_climate_signal",
        ),

        token_groups=(
            ("e2", "signal"),
            ("topofusion", "signal"),
            ("res", "signal"),
            ("e2", "change"),
            ("topofusion", "change"),
        ),

        exclude=(
            "unet",
            "error",
            "bias",
            "diff",
        ),
    )


    signal_unet_key, signal_unet = pick_map(
        z,

        exact=(
            "unet",
            "unet_signal",
            "signal_unet",
            "unet_climate_signal",
        ),

        token_groups=(
            ("unet", "signal"),
            ("unet", "change"),
        ),

        exclude=(
            "error",
            "bias",
            "diff",
        ),
    )


valid_trend = (
    valid
    & np.isfinite(trend_rcm)
    & np.isfinite(trend_topo)
    & np.isfinite(trend_unet)
)


valid_signal = (
    valid
    & np.isfinite(signal_rcm)
    & np.isfinite(signal_topo)
    & np.isfinite(signal_unet)
)


trend_rcm = masked(
    trend_rcm,
    valid_trend,
)

trend_topo = masked(
    trend_topo,
    valid_trend,
)

trend_unet = masked(
    trend_unet,
    valid_trend,
)


signal_rcm = masked(
    signal_rcm,
    valid_signal,
)

signal_topo = masked(
    signal_topo,
    valid_signal,
)

signal_unet = masked(
    signal_unet,
    valid_signal,
)


# ============================================================
# ERROR DIFFERENCE
#
# Positive:
# TopoFusion-Res has smaller local absolute error.
#
# Negative:
# U-Net has smaller local absolute error.
# ============================================================

trend_abs_error_diff = (
    np.abs(
        trend_unet
        - trend_rcm
    )
    -
    np.abs(
        trend_topo
        - trend_rcm
    )
)


signal_abs_error_diff = (
    np.abs(
        signal_unet
        - signal_rcm
    )
    -
    np.abs(
        signal_topo
        - signal_rcm
    )
)


trend_lo, trend_hi = display_range(
    trend_rcm,
    trend_topo,
    trend_unet,
    mask=valid_trend,
)


signal_lo, signal_hi = display_range(
    signal_rcm,
    signal_topo,
    signal_unet,
    mask=valid_signal,
)


trend_diff_lim = symmetric_p98(
    trend_abs_error_diff,
    valid_trend,
)


signal_diff_lim = symmetric_p98(
    signal_abs_error_diff,
    valid_signal,
)


# ============================================================
# METRICS
# ============================================================

def error_metrics(
    pred,
    ref,
    mask,
):
    e = (
        pred
        - ref
    )[mask]

    e = e[
        np.isfinite(e)
    ]

    return {
        "bias":
            float(
                np.mean(e)
            ),

        "mae":
            float(
                np.mean(
                    np.abs(e)
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        e ** 2
                    )
                )
            ),
    }


trend_topo_metrics = error_metrics(
    trend_topo,
    trend_rcm,
    valid_trend,
)


trend_unet_metrics = error_metrics(
    trend_unet,
    trend_rcm,
    valid_trend,
)


signal_topo_metrics = error_metrics(
    signal_topo,
    signal_rcm,
    valid_signal,
)


signal_unet_metrics = error_metrics(
    signal_unet,
    signal_rcm,
    valid_signal,
)


print()
print("TREND")

print(
    "  TopoFusion:",
    trend_topo_metrics,
)

print(
    "  U-Net     :",
    trend_unet_metrics,
)

print()
print("CLIMATE SIGNAL")

print(
    "  TopoFusion:",
    signal_topo_metrics,
)

print(
    "  U-Net     :",
    signal_unet_metrics,
)


# ============================================================
# FIGURE 8 LAYOUT
# ============================================================

cmap_field = plt.get_cmap(
    "viridis"
).copy()

cmap_field.set_bad(
    "white"
)


cmap_error = plt.get_cmap(
    "RdBu_r"
).copy()

cmap_error.set_bad(
    "white"
)


fig = plt.figure(
    figsize=(
        8.8,
        6.75,
    )
)


gs = fig.add_gridspec(
    4,
    4,

    height_ratios=[
        1.0,
        0.065,
        1.0,
        0.065,
    ],

    left=0.055,
    right=0.985,

    bottom=0.07,
    top=0.955,

    wspace=0.16,
    hspace=0.28,
)


axes_top = [
    fig.add_subplot(
        gs[0, i]
    )
    for i in range(4)
]


axes_bottom = [
    fig.add_subplot(
        gs[2, i]
    )
    for i in range(4)
]


# ============================================================
# ROW (a) TREND
# ============================================================

im_t0 = axes_top[0].pcolormesh(
    LON,
    LAT,
    trend_rcm,

    shading="auto",
    rasterized=True,

    cmap=cmap_field,

    vmin=trend_lo,
    vmax=trend_hi,
)


im_t1 = axes_top[1].pcolormesh(
    LON,
    LAT,
    trend_topo,

    shading="auto",
    rasterized=True,

    cmap=cmap_field,

    vmin=trend_lo,
    vmax=trend_hi,
)


im_t2 = axes_top[2].pcolormesh(
    LON,
    LAT,
    trend_unet,

    shading="auto",
    rasterized=True,

    cmap=cmap_field,

    vmin=trend_lo,
    vmax=trend_hi,
)


trend_norm = TwoSlopeNorm(
    vmin=-trend_diff_lim,
    vcenter=0.0,
    vmax=trend_diff_lim,
)


im_t3 = axes_top[3].pcolormesh(
    LON,
    LAT,
    trend_abs_error_diff,

    shading="auto",
    rasterized=True,

    cmap=cmap_error,

    norm=trend_norm,
)


top_titles = [
    "(a1) CNRM-ALADIN64",
    "(a2) TopoFusion-Res",
    "(a3) U-Net CMIP-CMIP",
    "(a4) Absolute-error difference",
]


for i, ax in enumerate(
    axes_top
):

    ax.set_title(
        top_titles[i],
        fontweight="bold",
    )

    format_map(
        ax,
        show_ylabel=(i == 0),
    )

    add_topography(
        ax
    )


axes_top[3].text(
    0.035,
    0.035,

    (
        "RMSE\n"
        f"Topo {trend_topo_metrics['rmse']:.3f}\n"
        f"U-Net {trend_unet_metrics['rmse']:.3f}"
    ),

    transform=axes_top[3].transAxes,

    fontsize=5.7,

    ha="left",
    va="bottom",

    bbox=dict(
        boxstyle="round,pad=0.18",

        facecolor="white",
        edgecolor="#666666",

        linewidth=0.4,
        alpha=0.86,
    ),

    zorder=30,
)


# ============================================================
# TREND COLORBARS
# ============================================================

cax_t_field = fig.add_subplot(
    gs[
        1,
        0:3
    ]
)

cax_t_diff = fig.add_subplot(
    gs[
        1,
        3
    ]
)


cb_t_field = fig.colorbar(
    im_t2,

    cax=cax_t_field,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)

cb_t_field.set_label(
    r"Temperature trend ($^\circ$C decade$^{-1}$)"
)


cb_t_diff = fig.colorbar(
    im_t3,

    cax=cax_t_diff,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)

cb_t_diff.set_label(
    r"$|e_{\rm U-Net}|-|e_{\rm Topo}|$"
)


# ============================================================
# ROW (b) CLIMATE CHANGE SIGNAL
# ============================================================

im_s0 = axes_bottom[0].pcolormesh(
    LON,
    LAT,
    signal_rcm,

    shading="auto",
    rasterized=True,

    cmap=cmap_field,

    vmin=signal_lo,
    vmax=signal_hi,
)


im_s1 = axes_bottom[1].pcolormesh(
    LON,
    LAT,
    signal_topo,

    shading="auto",
    rasterized=True,

    cmap=cmap_field,

    vmin=signal_lo,
    vmax=signal_hi,
)


im_s2 = axes_bottom[2].pcolormesh(
    LON,
    LAT,
    signal_unet,

    shading="auto",
    rasterized=True,

    cmap=cmap_field,

    vmin=signal_lo,
    vmax=signal_hi,
)


signal_norm = TwoSlopeNorm(
    vmin=-signal_diff_lim,
    vcenter=0.0,
    vmax=signal_diff_lim,
)


im_s3 = axes_bottom[3].pcolormesh(
    LON,
    LAT,
    signal_abs_error_diff,

    shading="auto",
    rasterized=True,

    cmap=cmap_error,

    norm=signal_norm,
)


bottom_titles = [
    "(b1) CNRM-ALADIN64",
    "(b2) TopoFusion-Res",
    "(b3) U-Net CMIP-CMIP",
    "(b4) Absolute-error difference",
]


for i, ax in enumerate(
    axes_bottom
):

    ax.set_title(
        bottom_titles[i],
        fontweight="bold",
    )

    format_map(
        ax,
        show_ylabel=(i == 0),
    )

    add_topography(
        ax
    )


axes_bottom[3].text(
    0.035,
    0.035,

    (
        "RMSE\n"
        f"Topo {signal_topo_metrics['rmse']:.3f}\n"
        f"U-Net {signal_unet_metrics['rmse']:.3f}"
    ),

    transform=axes_bottom[3].transAxes,

    fontsize=5.7,

    ha="left",
    va="bottom",

    bbox=dict(
        boxstyle="round,pad=0.18",

        facecolor="white",
        edgecolor="#666666",

        linewidth=0.4,
        alpha=0.86,
    ),

    zorder=30,
)


# ============================================================
# SIGNAL COLORBARS
# ============================================================

cax_s_field = fig.add_subplot(
    gs[
        3,
        0:3
    ]
)

cax_s_diff = fig.add_subplot(
    gs[
        3,
        3
    ]
)


cb_s_field = fig.colorbar(
    im_s2,

    cax=cax_s_field,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)

cb_s_field.set_label(
    r"Climate-change signal ($^\circ$C)"
)


cb_s_diff = fig.colorbar(
    im_s3,

    cax=cax_s_diff,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)

cb_s_diff.set_label(
    r"$|e_{\rm U-Net}|-|e_{\rm Topo}|$"
)


# ============================================================
# SAVE FIGURE 8
# ============================================================

FIG8_PNG = (
    OUT
    / "Figure08_long_term_trend_signal.png"
)

FIG8_PDF = (
    OUT
    / "Figure08_long_term_trend_signal.pdf"
)


fig.savefig(
    FIG8_PNG,

    dpi=400,

    bbox_inches="tight",

    facecolor="white",
)

fig.savefig(
    FIG8_PDF,

    bbox_inches="tight",

    facecolor="white",
)

plt.close(fig)


meta8 = {

    "figure":
        "Figure 8",

    "trend": {

        "reference_key":
            trend_rcm_key,

        "topofusion_key":
            trend_topo_key,

        "unet_key":
            trend_unet_key,

        "units":
            "degC per decade",

        "TopoFusion-Res":
            trend_topo_metrics,

        "U-Net CMIP-CMIP":
            trend_unet_metrics,

        "difference_definition":
            (
                "abs(U-Net - RCM) minus "
                "abs(TopoFusion-Res - RCM). Positive values "
                "indicate smaller local absolute error "
                "for TopoFusion-Res; negative values indicate "
                "smaller local absolute error for U-Net."
            ),
    },

    "climate_change_signal": {

        "reference_key":
            signal_rcm_key,

        "topofusion_key":
            signal_topo_key,

        "unet_key":
            signal_unet_key,

        "definition":
            "2080-2100 mean minus 2015-2035 mean",

        "units":
            "degC",

        "TopoFusion-Res":
            signal_topo_metrics,

        "U-Net CMIP-CMIP":
            signal_unet_metrics,

        "difference_definition":
            (
                "abs(U-Net - RCM) minus "
                "abs(TopoFusion-Res - RCM). Positive values "
                "indicate smaller local absolute error "
                "for TopoFusion-Res; negative values indicate "
                "smaller local absolute error for U-Net."
            ),
    },

    "topographic_contours_m": [
        500,
        1500,
        3000,
    ],

    "outputs": {
        "png":
            str(FIG8_PNG),

        "pdf":
            str(FIG8_PDF),
    },
}


with open(
    META_OUT
    / "Figure08_long_term_trend_signal_metadata.json",
    "w",
) as f:

    json.dump(
        meta8,
        f,
        indent=2,
    )


# ============================================================
# FINAL AUDIT
# ============================================================

print()
print("=" * 90)
print("FINAL OUTPUTS")
print("=" * 90)

print()
print("FIGURE 7")
print(FIG7_PNG)
print(FIG7_PDF)

print()
print("FIGURE 8")
print(FIG8_PNG)
print(FIG8_PDF)

print()
print("Expected qualitative checks:")
print(
    "  Fig. 7: U-Net should be closer to "
    "the RCM in domain-mean lag-1."
)

print(
    "  Fig. 8: U-Net should show somewhat lower "
    "spatial RMSE/MAE for both trend and climate signal."
)

print()
print("DONE")
