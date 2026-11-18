#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parents[1]
ROOT = PAPER.parent

CUBE = (
    ROOT
    / "diagnostics"
    / "final_e2_vs_unet_advanced_diagnostics"
    / "prediction_cubes"
)

META_FILE = CUBE / "cube_metadata.json"
RCM_FILE = CUBE / "rcm.dat"
TOPO_FILE = CUBE / "e2_full_seeded.dat"
UNET_FILE = CUBE / "unet.dat"

RMSE_FILE = (
    PAPER
    / "diagnostics"
    / "38_final_seeded_spatial_rmse_maps.npz"
)

STATIC_FILE = (
    ROOT
    / "data"
    / "torch_dataset_model_E_physical_v1"
    / "static_features_64x64.npy"
)

STATS_FILE = (
    ROOT
    / "data"
    / "torch_dataset_model_E_physical_v1"
    / "normalization_stats.json"
)

OUT_DIR = PAPER / "final_publication_figures"
TABLE_DIR = PAPER / "tables"
DIAG_DIR = PAPER / "diagnostics"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = (
    OUT_DIR
    / "Figure09_distribution_by_orographic_band_v6.png"
)

OUT_PDF = (
    OUT_DIR
    / "Figure09_distribution_by_orographic_band_v6.pdf"
)

OUT_CSV = (
    TABLE_DIR
    / "62_distribution_by_orographic_band_v6_summary.csv"
)

OUT_JSON = (
    DIAG_DIR
    / "62_distribution_by_orographic_band_v6_metadata.json"
)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})

REF_COLOR = "#1f77b4"
TOPO_COLOR = "#ff7f0e"
UNET_COLOR = "#2ca02c"

RNG = np.random.default_rng(20261001)


# ============================================================
# COMMON SUPPORT
# ============================================================

with np.load(
    RMSE_FILE,
    allow_pickle=False,
) as z:

    valid = np.asarray(
        z["valid"]
    ).astype(bool)

    lat = np.asarray(
        z["lat"]
    )

    lon = np.asarray(
        z["lon"]
    )


H, W = valid.shape

if int(valid.sum()) != 1852:
    raise RuntimeError(
        f"Expected 1852 common cells, "
        f"got {int(valid.sum())}"
    )


if (
    lat.shape == (H, W)
    and lon.shape == (H, W)
):
    LAT = lat
    LON = lon

elif (
    lat.ndim == 1
    and lon.ndim == 1
):
    LON, LAT = np.meshgrid(
        lon,
        lat,
    )

else:
    raise RuntimeError(
        f"Unexpected coordinate shapes: "
        f"lat={lat.shape}, lon={lon.shape}"
    )


# ============================================================
# PHYSICAL OROGRAPHY
# ============================================================

static = np.load(
    STATIC_FILE
).astype(np.float64)

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
    encoding="utf-8",
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

orography = np.asarray(
    orography,
    dtype=np.float64,
)

# Only final common support.
orography[
    ~valid
] = np.nan


finite_orog = orography[
    np.isfinite(orography)
]


print("=" * 100)
print("FIGURE 09 V6 — DISTRIBUTIONAL FIDELITY BY OROGRAPHIC BAND")
print("=" * 100)

print(
    "Valid orography range:",
    f"{np.nanmin(finite_orog):.2f}",
    "..",
    f"{np.nanmax(finite_orog):.2f}",
    "m",
)


# ============================================================
# OFFICIAL FINAL SEEDED CUBES
# ============================================================

with open(
    META_FILE,
    "r",
    encoding="utf-8",
) as f:

    meta = json.load(f)


shape = tuple(
    meta["shape"]
)

if shape != (
    31390,
    64,
    64,
):
    raise RuntimeError(
        f"Unexpected cube shape: {shape}"
    )


N, _, _ = shape


rcm = np.memmap(
    RCM_FILE,
    dtype="float32",
    mode="r",
    shape=shape,
)

topo = np.memmap(
    TOPO_FILE,
    dtype="float32",
    mode="r",
    shape=shape,
)

unet = np.memmap(
    UNET_FILE,
    dtype="float32",
    mode="r",
    shape=shape,
)


print("Days:", N)
print("Common cells:", int(valid.sum()))
print("Pairs:", N * int(valid.sum()))
print()


# ============================================================
# OROGRAPHIC BANDS
# ============================================================

bands = [
    (
        "< 500 m",
        valid
        & (orography < 500.0),
    ),

    (
        "500–1500 m",
        valid
        & (orography >= 500.0)
        & (orography < 1500.0),
    ),

    (
        "1500–3000 m",
        valid
        & (orography >= 1500.0)
        & (orography < 3000.0),
    ),

    (
        "> 3000 m",
        valid
        & (orography >= 3000.0),
    ),
]


counts = [
    int(mask.sum())
    for _, mask in bands
]


if counts != [
    163,
    618,
    428,
    643,
]:
    raise RuntimeError(
        f"Unexpected band counts: {counts}"
    )


print("Orographic bands:")

for (name, _), n in zip(
    bands,
    counts,
):
    print(
        f"  {name:13s}: {n} cells"
    )

print()


# ============================================================
# HELPERS
# ============================================================

def extract_band(
    mask,
):
    iy, ix = np.where(
        mask
    )

    rr = np.asarray(
        rcm[
            :,
            iy,
            ix,
        ],
        dtype=np.float32,
    ).reshape(-1)

    tt = np.asarray(
        topo[
            :,
            iy,
            ix,
        ],
        dtype=np.float32,
    ).reshape(-1)

    uu = np.asarray(
        unet[
            :,
            iy,
            ix,
        ],
        dtype=np.float32,
    ).reshape(-1)

    good = (
        np.isfinite(rr)
        & np.isfinite(tt)
        & np.isfinite(uu)
    )

    return (
        rr[good],
        tt[good],
        uu[good],
    )


def sample_values(
    x,
    max_n,
):
    x = np.asarray(
        x
    )

    if x.size <= max_n:
        return x.astype(
            np.float64
        )

    idx = RNG.choice(
        x.size,
        size=max_n,
        replace=False,
    )

    return x[
        idx
    ].astype(
        np.float64
    )


def kde_values(
    x,
    grid,
):
    xs = sample_values(
        x,
        40000,
    )

    kde = gaussian_kde(
        xs
    )

    return kde(
        grid
    )


def calc_stats(
    ref,
    pred,
):
    ref = ref.astype(
        np.float64
    )

    pred = pred.astype(
        np.float64
    )

    p01_ref = float(
        np.percentile(
            ref,
            1,
        )
    )

    p99_ref = float(
        np.percentile(
            ref,
            99,
        )
    )

    p01_pred = float(
        np.percentile(
            pred,
            1,
        )
    )

    p99_pred = float(
        np.percentile(
            pred,
            99,
        )
    )

    dp01 = (
        p01_pred
        - p01_ref
    )

    dp99 = (
        p99_pred
        - p99_ref
    )

    return {
        "bias":
            float(
                np.mean(
                    pred
                    - ref
                )
            ),

        "p01_ref":
            p01_ref,

        "p99_ref":
            p99_ref,

        "p01_pred":
            p01_pred,

        "p99_pred":
            p99_pred,

        "dp01":
            dp01,

        "dp99":
            dp99,

        "abs_dp01":
            abs(dp01),

        "abs_dp99":
            abs(dp99),
    }


# ============================================================
# FULL STATISTICS
# ============================================================

summary_rows = []
plot_cache = {}


print("=" * 100)
print("FULL BAND STATISTICS")
print("=" * 100)


for name, mask in bands:

    rr, tt, uu = extract_band(
        mask
    )

    ts = calc_stats(
        rr,
        tt,
    )

    us = calc_stats(
        rr,
        uu,
    )

    row = {
        "band":
            name,

        "cells":
            int(
                mask.sum()
            ),

        "pairs":
            int(
                rr.size
            ),

        "topo_bias_C":
            ts["bias"],

        "topo_dp01_C":
            ts["dp01"],

        "topo_dp99_C":
            ts["dp99"],

        "topo_abs_dp01_C":
            ts["abs_dp01"],

        "topo_abs_dp99_C":
            ts["abs_dp99"],

        "unet_bias_C":
            us["bias"],

        "unet_dp01_C":
            us["dp01"],

        "unet_dp99_C":
            us["dp99"],

        "unet_abs_dp01_C":
            us["abs_dp01"],

        "unet_abs_dp99_C":
            us["abs_dp99"],

        "rcm_p01_C":
            ts["p01_ref"],

        "rcm_p99_C":
            ts["p99_ref"],

        "topo_p01_C":
            ts["p01_pred"],

        "topo_p99_C":
            ts["p99_pred"],

        "unet_p01_C":
            us["p01_pred"],

        "unet_p99_C":
            us["p99_pred"],
    }

    summary_rows.append(
        row
    )


    # Fixed reproducible plotting samples.
    plot_cache[
        name
    ] = {
        "ref":
            sample_values(
                rr,
                150000,
            ),

        "topo":
            sample_values(
                tt,
                150000,
            ),

        "unet":
            sample_values(
                uu,
                150000,
            ),
    }


    print()
    print(name)

    print(
        f"  cells={row['cells']} "
        f"pairs={row['pairs']:,}"
    )

    print(
        "  Topo "
        f"bias={row['topo_bias_C']:+.4f} "
        f"dp01={row['topo_dp01_C']:+.4f} "
        f"dp99={row['topo_dp99_C']:+.4f}"
    )

    print(
        "  U-Net "
        f"bias={row['unet_bias_C']:+.4f} "
        f"dp01={row['unet_dp01_C']:+.4f} "
        f"dp99={row['unet_dp99_C']:+.4f}"
    )


    del rr
    del tt
    del uu


# ============================================================
# SAVE TABLE
# ============================================================

with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        17.2,
        10.7,
    )
)

gs = fig.add_gridspec(
    2,
    3,
    width_ratios=[
        0.92,
        1.15,
        1.15,
    ],
    height_ratios=[
        1.0,
        1.0,
    ],
    left=0.055,
    right=0.985,
    bottom=0.07,
    top=0.925,
    wspace=0.21,
    hspace=0.27,
)


ax_map = fig.add_subplot(
    gs[:, 0]
)

axes = [
    fig.add_subplot(
        gs[0, 1]
    ),

    fig.add_subplot(
        gs[0, 2]
    ),

    fig.add_subplot(
        gs[1, 1]
    ),

    fig.add_subplot(
        gs[1, 2]
    ),
]


# ============================================================
# PANEL (a): CONTINUOUS REAL ELEVATION
# ============================================================

levels = np.linspace(
    0,
    6000,
    31,
)


cf = ax_map.contourf(
    LON,
    LAT,
    orography,
    levels=levels,
    cmap="terrain",
    extend="max",
)


cs = ax_map.contour(
    LON,
    LAT,
    orography,
    levels=[
        500,
        1500,
        3000,
    ],
    colors="0.25",
    linewidths=0.85,
)


contour_labels = ax_map.clabel(
    cs,
    inline=True,
    fontsize=8,
    fmt=lambda x: f"{int(x)} m",
)

for txt in contour_labels:
    txt.set_color("white")
    txt.set_fontsize(8)
    txt.set_fontweight("normal")
    txt.set_path_effects([
        pe.withStroke(
            linewidth=1.6,
            foreground="black",
            alpha=0.55,
        )
    ])


ax_map.set_title(
    "(a) Orographic bands"
)

ax_map.set_xlabel(
    "Longitude"
)

ax_map.set_ylabel(
    "Latitude"
)


# Keep successful V1-style labels.
label_offsets = {
    "< 500 m":
        (-0.30, -0.25),

    "500–1500 m":
        (0.20, -0.20),

    "1500–3000 m":
        (0.15, 0.08),

    "> 3000 m":
        (0.10, 0.15),
}


for name, mask in bands:

    x = float(
        np.nanmedian(
            LON[
                mask
            ]
        )
    )

    y = float(
        np.nanmedian(
            LAT[
                mask
            ]
        )
    )

    dx, dy = (
        label_offsets[
            name
        ]
    )

    ax_map.text(
        x + dx,
        y + dy,
        name,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="0.45",
            alpha=0.86,
        ),
    )


cbar = fig.colorbar(
    cf,
    ax=ax_map,
    orientation="horizontal",
    fraction=0.045,
    pad=0.085,
)


cbar.set_ticks(
    [
        0,
        500,
        1500,
        3000,
        4500,
        6000,
    ]
)

cbar.set_label(
    "Elevation (m)"
)


# ============================================================
# PANELS (b)-(e): HISTOGRAM + KDE
# ============================================================

letters = [
    "(b)",
    "(c)",
    "(d)",
    "(e)",
]


for ax, letter, row in zip(
    axes,
    letters,
    summary_rows,
):

    name = row[
        "band"
    ]

    d = plot_cache[
        name
    ]

    ref = d[
        "ref"
    ]

    tt = d[
        "topo"
    ]

    uu = d[
        "unet"
    ]


    all_vals = np.concatenate(
        [
            ref,
            tt,
            uu,
        ]
    )


    lo = float(
        np.percentile(
            all_vals,
            0.10,
        )
    )

    hi = float(
        np.percentile(
            all_vals,
            99.90,
        )
    )

    span = (
        hi
        - lo
    )

    lo -= (
        0.05
        * span
    )

    hi += (
        0.05
        * span
    )


    bins = np.linspace(
        lo,
        hi,
        45,
    )

    grid = np.linspace(
        lo,
        hi,
        500,
    )


    # --------------------------------------------------------
    # Light histograms — visual character of V1
    # --------------------------------------------------------

    ax.hist(
        ref,
        bins=bins,
        density=True,
        color=REF_COLOR,
        alpha=0.16,
        edgecolor="none",
    )

    ax.hist(
        tt,
        bins=bins,
        density=True,
        color=TOPO_COLOR,
        alpha=0.16,
        edgecolor="none",
    )

    ax.hist(
        uu,
        bins=bins,
        density=True,
        color=UNET_COLOR,
        alpha=0.16,
        edgecolor="none",
    )


    # --------------------------------------------------------
    # KDE
    # --------------------------------------------------------

    y_ref = kde_values(
        ref,
        grid,
    )

    y_topo = kde_values(
        tt,
        grid,
    )

    y_unet = kde_values(
        uu,
        grid,
    )


    ax.plot(
        grid,
        y_ref,
        color=REF_COLOR,
        linewidth=2.15,
    )

    ax.plot(
        grid,
        y_topo,
        color=TOPO_COLOR,
        linewidth=2.15,
    )

    ax.plot(
        grid,
        y_unet,
        color=UNET_COLOR,
        linewidth=2.15,
    )


    # --------------------------------------------------------
    # p01 / p99
    # --------------------------------------------------------

    for x, color in [
        (
            row[
                "rcm_p01_C"
            ],
            REF_COLOR,
        ),

        (
            row[
                "rcm_p99_C"
            ],
            REF_COLOR,
        ),

        (
            row[
                "topo_p01_C"
            ],
            TOPO_COLOR,
        ),

        (
            row[
                "topo_p99_C"
            ],
            TOPO_COLOR,
        ),

        (
            row[
                "unet_p01_C"
            ],
            UNET_COLOR,
        ),

        (
            row[
                "unet_p99_C"
            ],
            UNET_COLOR,
        ),
    ]:

        ax.axvline(
            x,
            color=color,
            linestyle=":",
            linewidth=1.1,
            alpha=0.78,
        )


    # --------------------------------------------------------
    # Compact quantitative box
    # --------------------------------------------------------

    txt = (
        r"$|\Delta p01|$ / $|\Delta p99|$ (°C)"
        "\n"
        f"Topo     "
        f"{row['topo_abs_dp01_C']:.2f} / "
        f"{row['topo_abs_dp99_C']:.2f}"
        "\n"
        f"U-Net    "
        f"{row['unet_abs_dp01_C']:.2f} / "
        f"{row['unet_abs_dp99_C']:.2f}"
    )


    ax.text(
        0.97,
        0.96,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="0.55",
            alpha=0.88,
        ),
    )


    ax.set_title(
        f"{letter} {name}"
    )

    ax.set_xlabel(
        "Temperature (°C)"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.grid(
        alpha=0.20,
    )


# ============================================================
# ONE COMMON LEGEND
# ============================================================

legend_handles = [
    Line2D(
        [0],
        [0],
        color=REF_COLOR,
        linewidth=2.3,
        label="CNRM-ALADIN64",
    ),

    Line2D(
        [0],
        [0],
        color=TOPO_COLOR,
        linewidth=2.3,
        label="TopoFusion-Res",
    ),

    Line2D(
        [0],
        [0],
        color=UNET_COLOR,
        linewidth=2.3,
        label="U-Net CMIP-CMIP",
    ),

    Line2D(
        [0],
        [0],
        color="0.4",
        linestyle=":",
        linewidth=1.2,
        label="p01 / p99",
    ),
]


fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.665, 0.985),
    ncol=4,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.6,
    borderaxespad=0.0,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "figure":
        "Figure09_distribution_by_orographic_band_v6",

    "scenario":
        "SSP3-7.0",

    "period":
        "2015-2100",

    "days":
        int(N),

    "common_cells":
        int(
            valid.sum()
        ),

    "orography_range_common_m": [
        float(
            np.nanmin(
                finite_orog
            )
        ),
        float(
            np.nanmax(
                finite_orog
            )
        ),
    ],

    "band_counts":
        counts,

    "reference_cube":
        str(
            RCM_FILE
        ),

    "topofusion_cube":
        str(
            TOPO_FILE
        ),

    "unet_cube":
        str(
            UNET_FILE
        ),

    "summary":
        summary_rows,

    "interpretation_note": (
        "The displayed PDFs/KDEs pool all common daily "
        "temperature values within each orographic band. "
        "The p01/p99 values shown here are band-pooled "
        "descriptive quantities and are distinct from the "
        "cell-wise percentile diagnostics used in the main "
        "temperature-tail analysis."
    ),

    "outputs": {
        "png":
            str(
                OUT_PNG
            ),

        "pdf":
            str(
                OUT_PDF
            ),

        "csv":
            str(
                OUT_CSV
            ),
    },
}


with open(
    OUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print(OUT_PNG)
print(OUT_PDF)
print(OUT_CSV)
print(OUT_JSON)

print()
print("DONE")
