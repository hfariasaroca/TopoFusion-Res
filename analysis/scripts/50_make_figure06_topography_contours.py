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

VAR_FILE = DIA / "06b_true_variance_ratio_maps.npz"
WASS_FILE = DIA / "07_wasserstein_distance_maps.npz"
RMSE_FILE = DIA / "38_final_seeded_spatial_rmse_maps.npz"

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

OUT = FINAL / "final_publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUT / "Figure06_variance_wasserstein.png"
PDF_OUT = OUT / "Figure06_variance_wasserstein.pdf"
JSON_OUT = OUT / "Figure06_variance_wasserstein_metadata.json"


# ============================================================
# INPUT AUDIT
# ============================================================

for p in [
    VAR_FILE,
    WASS_FILE,
    RMSE_FILE,
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
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,

    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,

    "axes.linewidth": 0.7,

    "figure.dpi": 150,
    "savefig.dpi": 400,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# HELPERS
# ============================================================

def find_key(keys, alternatives, exclude=()):
    keys = list(keys)

    for tokens in alternatives:
        for key in keys:

            kl = key.lower()

            if any(
                token in kl
                for token in exclude
            ):
                continue

            if all(
                token in kl
                for token in tokens
            ):
                return key

    raise RuntimeError(
        "Could not identify diagnostic key.\n"
        f"Available keys: {keys}\n"
        f"Patterns: {alternatives}"
    )


def masked(a, valid):
    out = np.asarray(
        a,
        dtype=np.float64,
    ).copy()

    out[~valid] = np.nan

    return out


def valid_values(a, valid):
    v = np.asarray(
        a,
        dtype=np.float64,
    )[valid]

    return v[
        np.isfinite(v)
    ]


def symmetric_p98(a, valid):
    vals = np.abs(
        valid_values(a, valid)
    )

    return max(
        float(
            np.percentile(
                vals,
                98,
            )
        ),
        1e-12,
    )


def common_p98(a, b, valid):
    vals = np.concatenate([
        valid_values(a, valid),
        valid_values(b, valid),
    ])

    return float(
        np.percentile(
            vals,
            98,
        )
    )


# ============================================================
# COMMON FINAL MASK / COORDINATES
# ============================================================

with np.load(
    RMSE_FILE,
    allow_pickle=False,
) as z:

    print(
        "RMSE keys:",
        z.files,
    )

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
        f"LAT shape mismatch: {LAT.shape}"
    )

if LON.shape != valid.shape:
    raise RuntimeError(
        f"LON shape mismatch: {LON.shape}"
    )


print()
print(
    "Common valid cells:",
    int(valid.sum()),
)


# ============================================================
# VARIANCE RATIO
# ============================================================

with np.load(
    VAR_FILE,
    allow_pickle=False,
) as z:

    print()
    print(
        "Variance keys:",
        z.files,
    )

    topo_key = find_key(
        z.files,
        alternatives=[
            ("e2", "variance", "ratio"),
            ("e2", "ratio"),
            ("topofusion", "ratio"),
        ],
        exclude=(
            "unet",
            "diff",
            "improvement",
        ),
    )

    unet_key = find_key(
        z.files,
        alternatives=[
            ("unet", "variance", "ratio"),
            ("unet", "ratio"),
        ],
    )

    ratio_topo = np.asarray(
        z[topo_key],
        dtype=np.float64,
    )

    ratio_unet = np.asarray(
        z[unet_key],
        dtype=np.float64,
    )


# ============================================================
# WASSERSTEIN
# ============================================================

with np.load(
    WASS_FILE,
    allow_pickle=False,
) as z:

    print()
    print(
        "Wasserstein keys:",
        z.files,
    )

    topo_w_key = find_key(
        z.files,
        alternatives=[
            ("e2",),
            ("topofusion",),
        ],
        exclude=(
            "unet",
            "diff",
            "improvement",
        ),
    )

    unet_w_key = find_key(
        z.files,
        alternatives=[
            ("unet",),
        ],
        exclude=(
            "diff",
            "improvement",
        ),
    )

    wd_topo = np.asarray(
        z[topo_w_key],
        dtype=np.float64,
    )

    wd_unet = np.asarray(
        z[unet_w_key],
        dtype=np.float64,
    )


for name, a in [
    ("ratio_topo", ratio_topo),
    ("ratio_unet", ratio_unet),
    ("wd_topo", wd_topo),
    ("wd_unet", wd_unet),
]:
    if a.shape != valid.shape:
        raise RuntimeError(
            f"{name} shape mismatch: {a.shape}"
        )


ratio_topo = masked(
    ratio_topo,
    valid,
)

ratio_unet = masked(
    ratio_unet,
    valid,
)

wd_topo = masked(
    wd_topo,
    valid,
)

wd_unet = masked(
    wd_unet,
    valid,
)


# ============================================================
# DIFFERENCE DEFINITIONS
# ============================================================

# Positive:
# TopoFusion-Res is closer to ideal 100%.
variance_fidelity_diff = (
    np.abs(
        ratio_unet
        - 100.0
    )
    -
    np.abs(
        ratio_topo
        - 100.0
    )
)


# Positive:
# TopoFusion-Res has lower Wasserstein distance.
wasserstein_diff = (
    wd_unet
    -
    wd_topo
)


# ============================================================
# PHYSICAL OROGRAPHY
# ============================================================

with open(
    STATIC_NAMES_FILE,
    "r",
) as f:

    static_names = json.load(f)


print()
print("=" * 90)
print("FINE-STATIC FEATURE AUDIT")
print("=" * 90)

for i, name in enumerate(
    static_names
):
    print(
        f"channel {i}: {name}"
    )


if len(static_names) != 6:
    raise RuntimeError(
        "Expected six fine-static channels."
    )


channel0_name = str(
    static_names[0]
).lower()


if (
    "orog" not in channel0_name
    and "elev" not in channel0_name
):
    raise RuntimeError(
        "Channel 0 is not identified as orography."
    )


fine_static = np.load(
    STATIC_FILE
).astype(
    np.float64
)


if fine_static.shape != (
    6,
    64,
    64,
):
    raise RuntimeError(
        f"Unexpected static array: {fine_static.shape}"
    )


orog_norm = fine_static[0]


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
    orog_stats[
        "mean"
    ]
)

orog_std = float(
    orog_stats[
        "std"
    ]
)


orography = (
    orog_norm
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


print()
print("=" * 90)
print("PHYSICAL OROGRAPHY")
print("=" * 90)

print(
    "Normalized source:",
    STATIC_FILE,
)

print(
    "Stats source     :",
    STATS_FILE,
)

print(
    "Mean [m]         :",
    orog_mean,
)

print(
    "Std [m]          :",
    orog_std,
)

print(
    "Valid topo range :",
    float(np.nanmin(orography_plot)),
    "..",
    float(np.nanmax(orography_plot)),
    "m",
)


# ============================================================
# TOPOGRAPHY
# ============================================================

CONTOUR_LEVELS = [
    500.0,
    1500.0,
    3000.0,
]


ALTITUDE_BANDS = [
    {
        "label": "< 500 m",
        "low": None,
        "high": 500.0,
    },

    {
        "label": "500–1500 m",
        "low": 500.0,
        "high": 1500.0,
    },

    {
        "label": "1500–3000 m",
        "low": 1500.0,
        "high": 3000.0,
    },

    {
        "label": "> 3000 m",
        "low": 3000.0,
        "high": None,
    },
]


def add_topographic_contours(ax):

    return ax.contour(
        LON,
        LAT,
        orography_plot,

        levels=CONTOUR_LEVELS,

        colors="#616b70",

        linewidths=0.65,

        alpha=0.85,

        zorder=10,
    )


def get_band_mask(low=None, high=None):

    m = (
        valid
        & np.isfinite(orography)
    )

    if low is not None:
        m &= (
            orography
            >= low
        )

    if high is not None:
        m &= (
            orography
            < high
        )

    return m


def representative_band_position(mask):

    ii, jj = np.where(mask)

    if ii.size == 0:
        return None

    lons = LON[ii, jj]
    lats = LAT[ii, jj]

    lon_med = float(
        np.nanmedian(lons)
    )

    lat_med = float(
        np.nanmedian(lats)
    )

    coslat = np.cos(
        np.deg2rad(lat_med)
    )

    d2 = (
        (
            (lons - lon_med)
            * coslat
        ) ** 2
        +
        (
            lats - lat_med
        ) ** 2
    )

    k = int(
        np.nanargmin(d2)
    )

    return (
        float(lons[k]),
        float(lats[k]),
    )


def add_altitude_band_labels(ax):

    positions = {}

    for band in ALTITUDE_BANDS:

        mask = get_band_mask(
            low=band["low"],
            high=band["high"],
        )

        pos = representative_band_position(
            mask
        )

        if pos is None:
            continue

        x, y = pos

        positions[
            band["label"]
        ] = {
            "lon": x,
            "lat": y,
            "n_cells": int(
                mask.sum()
            ),
        }

        ax.text(
            x,
            y,
            band["label"],

            fontsize=5.8,
            fontweight="semibold",

            ha="center",
            va="center",

            color="#303030",

            bbox=dict(
                boxstyle="round,pad=0.13",

                facecolor="white",
                edgecolor="#777777",

                linewidth=0.4,
                alpha=0.78,
            ),

            zorder=30,
            clip_on=True,
        )

    return positions


# ============================================================
# MAP GEOMETRY
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
        np.deg2rad(
            mean_lat
        )
    )
)


def format_geo_axis(
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
# DISPLAY SCALES
# ============================================================

VARIANCE_MIN = 60.0
VARIANCE_MAX = 120.0


VARIANCE_DIFF_LIM = symmetric_p98(
    variance_fidelity_diff,
    valid,
)


WASSERSTEIN_MAX = common_p98(
    wd_topo,
    wd_unet,
    valid,
)


WASSERSTEIN_DIFF_LIM = symmetric_p98(
    wasserstein_diff,
    valid,
)


print()
print("=" * 90)
print("DISPLAY SCALES")
print("=" * 90)

print(
    "Variance ratio   :",
    VARIANCE_MIN,
    "..",
    VARIANCE_MAX,
    "%",
)

print(
    "Variance Δ p98   :",
    VARIANCE_DIFF_LIM,
    "percentage points",
)

print(
    "Wasserstein p98  :",
    WASSERSTEIN_MAX,
    "degC",
)

print(
    "Wasserstein Δ p98:",
    WASSERSTEIN_DIFF_LIM,
    "degC",
)


# ============================================================
# STATISTICS USED IN ANNOTATIONS
# ============================================================

vr_diff_vals = valid_values(
    variance_fidelity_diff,
    valid,
)


w_diff_vals = valid_values(
    wasserstein_diff,
    valid,
)


VR_MEAN_ADV = float(
    np.mean(vr_diff_vals)
)

VR_POS_FRAC = float(
    np.mean(
        vr_diff_vals > 0
    )
)


W_MEAN_ADV = float(
    np.mean(w_diff_vals)
)

W_POS_FRAC = float(
    np.mean(
        w_diff_vals > 0
    )
)


# ============================================================
# COLORMAPS
# ============================================================

cmap_abs = plt.get_cmap(
    "viridis"
).copy()

cmap_abs.set_bad(
    "white"
)


cmap_diff = plt.get_cmap(
    "RdBu_r"
).copy()

cmap_diff.set_bad(
    "white"
)


# ============================================================
# FIGURE LAYOUT
#
# Climate-paper style:
#   row heading
#   3 maps
#   horizontal colorbars
#   row heading
#   3 maps
#   horizontal colorbars
# ============================================================

fig = plt.figure(
    figsize=(
        8.0,
        7.65,
    ),
)


gs = fig.add_gridspec(
    6,
    3,

    height_ratios=[
        0.085,
        1.00,
        0.075,
        0.085,
        1.00,
        0.075,
    ],

    left=0.065,
    right=0.985,

    bottom=0.060,
    top=0.985,

    wspace=0.18,
    hspace=0.12,
)


# ============================================================
# ROW HEADERS
# ============================================================

header_a = fig.add_subplot(
    gs[0, :]
)

header_a.axis(
    "off"
)

header_a.text(
    0.0,
    0.5,
    "(a)  Variance fidelity",

    ha="left",
    va="center",

    fontsize=10.0,
    fontweight="bold",
)


header_b = fig.add_subplot(
    gs[3, :]
)

header_b.axis(
    "off"
)

header_b.text(
    0.0,
    0.5,
    "(b)  Wasserstein distance",

    ha="left",
    va="center",

    fontsize=10.0,
    fontweight="bold",
)


# ============================================================
# MAP AXES
# ============================================================

ax00 = fig.add_subplot(
    gs[1, 0]
)

ax01 = fig.add_subplot(
    gs[1, 1]
)

ax02 = fig.add_subplot(
    gs[1, 2]
)


ax10 = fig.add_subplot(
    gs[4, 0]
)

ax11 = fig.add_subplot(
    gs[4, 1]
)

ax12 = fig.add_subplot(
    gs[4, 2]
)


# ============================================================
# TOP ROW — VARIANCE FIDELITY
# ============================================================

im00 = ax00.pcolormesh(
    LON,
    LAT,
    ratio_topo,

    shading="auto",
    rasterized=True,

    cmap=cmap_abs,

    vmin=VARIANCE_MIN,
    vmax=VARIANCE_MAX,
)


ax00.set_title(
    "TopoFusion-Res",
    fontweight="bold",
    pad=4,
)

format_geo_axis(
    ax00,
    show_ylabel=True,
)

add_topographic_contours(
    ax00
)


im01 = ax01.pcolormesh(
    LON,
    LAT,
    ratio_unet,

    shading="auto",
    rasterized=True,

    cmap=cmap_abs,

    vmin=VARIANCE_MIN,
    vmax=VARIANCE_MAX,
)


ax01.set_title(
    "U-Net CMIP-CMIP",
    fontweight="bold",
    pad=4,
)

format_geo_axis(
    ax01,
    show_ylabel=False,
)

add_topographic_contours(
    ax01
)


var_norm = TwoSlopeNorm(
    vmin=-VARIANCE_DIFF_LIM,
    vcenter=0.0,
    vmax=VARIANCE_DIFF_LIM,
)


im02 = ax02.pcolormesh(
    LON,
    LAT,
    variance_fidelity_diff,

    shading="auto",
    rasterized=True,

    cmap=cmap_diff,

    norm=var_norm,
)


ax02.set_title(
    "Difference toward 100%",
    fontweight="bold",
    pad=4,
)

format_geo_axis(
    ax02,
    show_ylabel=False,
)

add_topographic_contours(
    ax02
)


band_positions_c = add_altitude_band_labels(
    ax02
)


ax02.text(
    0.035,
    0.035,

    (
        f"mean Δ = {VR_MEAN_ADV:.2f} pp\n"
        f"Δ > 0: {100.0 * VR_POS_FRAC:.1f}%"
    ),

    transform=ax02.transAxes,

    ha="left",
    va="bottom",

    fontsize=6.4,

    bbox=dict(
        boxstyle="round,pad=0.22",

        facecolor="white",
        edgecolor="#666666",

        linewidth=0.45,
        alpha=0.86,
    ),

    zorder=40,
)


# ============================================================
# TOP COLORBARS
# ============================================================

cb_top_left_ax = fig.add_subplot(
    gs[2, 0:2]
)

cb_top_right_ax = fig.add_subplot(
    gs[2, 2]
)


cb_var = fig.colorbar(
    im01,

    cax=cb_top_left_ax,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)


cb_var.set_label(
    "Variance ratio (%)",
    labelpad=1.5,
)


cb_var.ax.tick_params(
    length=2.5,
    pad=1.5,
)


cb_var_diff = fig.colorbar(
    im02,

    cax=cb_top_right_ax,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)


cb_var_diff.set_label(
    "Fidelity difference (percentage points)",
    labelpad=1.5,
)


cb_var_diff.ax.tick_params(
    length=2.5,
    pad=1.5,
)


# ============================================================
# BOTTOM ROW — WASSERSTEIN
# ============================================================

im10 = ax10.pcolormesh(
    LON,
    LAT,
    wd_topo,

    shading="auto",
    rasterized=True,

    cmap=cmap_abs,

    vmin=0.0,
    vmax=WASSERSTEIN_MAX,
)


ax10.set_title(
    "TopoFusion-Res",
    fontweight="bold",
    pad=4,
)

format_geo_axis(
    ax10,
    show_ylabel=True,
)

add_topographic_contours(
    ax10
)


im11 = ax11.pcolormesh(
    LON,
    LAT,
    wd_unet,

    shading="auto",
    rasterized=True,

    cmap=cmap_abs,

    vmin=0.0,
    vmax=WASSERSTEIN_MAX,
)


ax11.set_title(
    "U-Net CMIP-CMIP",
    fontweight="bold",
    pad=4,
)

format_geo_axis(
    ax11,
    show_ylabel=False,
)

add_topographic_contours(
    ax11
)


w_norm = TwoSlopeNorm(
    vmin=-WASSERSTEIN_DIFF_LIM,
    vcenter=0.0,
    vmax=WASSERSTEIN_DIFF_LIM,
)


im12 = ax12.pcolormesh(
    LON,
    LAT,
    wasserstein_diff,

    shading="auto",
    rasterized=True,

    cmap=cmap_diff,

    norm=w_norm,
)


ax12.set_title(
    r"$\Delta$ Wasserstein",
    fontweight="bold",
    pad=4,
)

format_geo_axis(
    ax12,
    show_ylabel=False,
)

add_topographic_contours(
    ax12
)


band_positions_f = add_altitude_band_labels(
    ax12
)


ax12.text(
    0.035,
    0.035,

    (
        f"mean Δ = {W_MEAN_ADV:.3f} °C\n"
        f"Δ > 0: {100.0 * W_POS_FRAC:.1f}%"
    ),

    transform=ax12.transAxes,

    ha="left",
    va="bottom",

    fontsize=6.4,

    bbox=dict(
        boxstyle="round,pad=0.22",

        facecolor="white",
        edgecolor="#666666",

        linewidth=0.45,
        alpha=0.86,
    ),

    zorder=40,
)


# ============================================================
# BOTTOM COLORBARS
# ============================================================

cb_bottom_left_ax = fig.add_subplot(
    gs[5, 0:2]
)

cb_bottom_right_ax = fig.add_subplot(
    gs[5, 2]
)


cb_w = fig.colorbar(
    im11,

    cax=cb_bottom_left_ax,

    orientation="horizontal",

    extend="max",
    extendfrac=0.025,
)


cb_w.set_label(
    r"Wasserstein distance ($^\circ$C)",
    labelpad=1.5,
)


cb_w.ax.tick_params(
    length=2.5,
    pad=1.5,
)


cb_w_diff = fig.colorbar(
    im12,

    cax=cb_bottom_right_ax,

    orientation="horizontal",

    extend="both",
    extendfrac=0.025,
)


cb_w_diff.set_label(
    r"$\Delta$ Wasserstein ($^\circ$C)",
    labelpad=1.5,
)


cb_w_diff.ax.tick_params(
    length=2.5,
    pad=1.5,
)


# ============================================================
# OPTIONAL SEPARATOR
#
# Thin line between the two scientific blocks.
# ============================================================

separator = plt.Line2D(
    [0.055, 0.985],
    [0.515, 0.515],

    transform=fig.transFigure,

    color="#555555",
    linewidth=0.65,
)

fig.add_artist(
    separator
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PNG_OUT,

    dpi=400,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(
    PDF_OUT,

    bbox_inches="tight",

    facecolor="white",
)


plt.close(fig)


# ============================================================
# FINAL STATISTICS
# ============================================================

vr_topo = valid_values(
    ratio_topo,
    valid,
)

vr_unet = valid_values(
    ratio_unet,
    valid,
)

w_topo = valid_values(
    wd_topo,
    valid,
)

w_unet = valid_values(
    wd_unet,
    valid,
)


summary = {

    "figure":
        "Figure 6",

    "layout":
        "climate-paper publication layout",

    "support": {
        "n_common_cells":
            int(valid.sum()),
    },

    "variance_ratio": {

        "definition":
            (
                "100 * model variance / "
                "CNRM-ALADIN64 variance"
            ),

        "topofusion_mean_pct":
            float(
                np.mean(vr_topo)
            ),

        "topofusion_median_pct":
            float(
                np.median(vr_topo)
            ),

        "unet_mean_pct":
            float(
                np.mean(vr_unet)
            ),

        "unet_median_pct":
            float(
                np.median(vr_unet)
            ),

        "fidelity_difference_definition":
            (
                "abs(U-Net variance ratio - 100) minus "
                "abs(TopoFusion-Res variance ratio - 100)"
            ),

        "positive_means":
            (
                "TopoFusion-Res is closer to "
                "the ideal variance ratio of 100%."
            ),

        "mean_fidelity_advantage_pp":
            VR_MEAN_ADV,

        "fraction_positive":
            VR_POS_FRAC,
    },

    "wasserstein": {

        "topofusion_mean_C":
            float(
                np.mean(w_topo)
            ),

        "topofusion_median_C":
            float(
                np.median(w_topo)
            ),

        "unet_mean_C":
            float(
                np.mean(w_unet)
            ),

        "unet_median_C":
            float(
                np.median(w_unet)
            ),

        "difference_definition":
            (
                "U-Net Wasserstein distance minus "
                "TopoFusion-Res Wasserstein distance"
            ),

        "positive_means":
            (
                "TopoFusion-Res has lower "
                "Wasserstein distance."
            ),

        "mean_advantage_C":
            W_MEAN_ADV,

        "fraction_positive":
            W_POS_FRAC,
    },

    "topography": {

        "contours_m": [
            500,
            1500,
            3000,
        ],

        "all_panels":
            True,

        "band_labels_panels": [
            "variance difference",
            "wasserstein difference",
        ],

        "band_positions_variance":
            band_positions_c,

        "band_positions_wasserstein":
            band_positions_f,

        "normalized_source":
            str(STATIC_FILE),

        "normalization_stats_source":
            str(STATS_FILE),

        "valid_elevation_min_m":
            float(
                np.nanmin(orography_plot)
            ),

        "valid_elevation_max_m":
            float(
                np.nanmax(orography_plot)
            ),
    },

    "visualization": {

        "variance_ratio_range_pct": [
            VARIANCE_MIN,
            VARIANCE_MAX,
        ],

        "variance_difference_p98_pp":
            VARIANCE_DIFF_LIM,

        "wasserstein_p98_C":
            WASSERSTEIN_MAX,

        "wasserstein_difference_p98_C":
            WASSERSTEIN_DIFF_LIM,

        "colorbars":
            "horizontal",

        "global_title":
            False,
    },

    "outputs": {

        "png":
            str(PNG_OUT),

        "pdf":
            str(PDF_OUT),
    },
}


with open(
    JSON_OUT,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL AUDIT
# ============================================================

print()
print("=" * 90)
print("FIGURE 6 — CLIMATE-PAPER PUBLICATION VERSION")
print("=" * 90)

print(
    "Common cells       :",
    int(valid.sum()),
)

print(
    "Topo range [m]     :",
    f"{np.nanmin(orography_plot):.2f}",
    "..",
    f"{np.nanmax(orography_plot):.2f}",
)

print(
    "Contours           :",
    "500, 1500, 3000 m",
)

print()
print(
    "Variance Topo mean :",
    float(np.mean(vr_topo)),
)

print(
    "Variance U-Net mean:",
    float(np.mean(vr_unet)),
)

print(
    "Mean fidelity Δ    :",
    VR_MEAN_ADV,
    "pp",
)

print(
    "Fidelity Δ > 0     :",
    100.0 * VR_POS_FRAC,
    "%",
)

print()
print(
    "Wasserstein Topo   :",
    float(np.mean(w_topo)),
)

print(
    "Wasserstein U-Net  :",
    float(np.mean(w_unet)),
)

print(
    "Mean Wasserstein Δ :",
    W_MEAN_ADV,
    "degC",
)

print(
    "Wasserstein Δ > 0  :",
    100.0 * W_POS_FRAC,
    "%",
)

print()
print("Saved:")
print(PNG_OUT)
print(PDF_OUT)
print(JSON_OUT)

print()
print("DONE")
