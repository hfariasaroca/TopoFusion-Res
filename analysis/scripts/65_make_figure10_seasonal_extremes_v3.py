#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


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

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PNG = (
    OUT_DIR
    / "Figure10_seasonal_extremes_v3.png"
)

OUT_PDF = (
    OUT_DIR
    / "Figure10_seasonal_extremes_v3.pdf"
)

OUT_CSV = (
    TABLE_DIR
    / "65_figure10_seasonal_extremes_v3_timeseries.csv"
)

OUT_JSON = (
    DIAG_DIR
    / "65_figure10_seasonal_extremes_v3_metadata.json"
)


# ============================================================
# CONSTANTS
# ============================================================

START_YEAR = 2015
END_YEAR = 2100

EXPECTED_DAYS = 31390
EXPECTED_YEARS = 86
EXPECTED_COMMON_CELLS = 1852

COLOR_RCM = "#1f77b4"
COLOR_TOPO = "#ff7f0e"
COLOR_UNET = "#2ca02c"

HIGHLIGHT_YEARS = [
    2020,
    2040,
    2060,
    2080,
    2100,
]

ANNOTATE_YEARS = [
    2020,
    2060,
    2100,
]


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,

    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "legend.fontsize": 9,

    "axes.linewidth": 0.8,

    "savefig.dpi": 300,
})


# ============================================================
# HELPERS
# ============================================================

def print_block(title):
    print("=" * 100)
    print(title)
    print("=" * 100)


def restore_map(
    values,
    valid,
):
    out = np.full(
        valid.shape,
        np.nan,
        dtype=np.float64,
    )

    out[
        valid
    ] = values

    return out


def masked(
    values,
    valid,
):
    out = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    out[
        ~valid
    ] = np.nan

    return out


def symmetric_limit(
    *arrays,
):
    vals = []

    for arr in arrays:

        x = np.asarray(
            arr,
            dtype=np.float64,
        )

        x = np.abs(
            x[
                np.isfinite(x)
            ]
        )

        if x.size:
            vals.append(
                x
            )

    if not vals:
        return 1.0

    vals = np.concatenate(
        vals
    )

    lim = float(
        np.percentile(
            vals,
            98.0,
        )
    )

    return max(
        lim,
        1e-8,
    )


def find_year_index(
    years_array,
    year,
):
    idx = np.flatnonzero(
        years_array == year
    )

    if len(idx) != 1:
        return None

    return int(
        idx[0]
    )


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

n_common = int(
    valid.sum()
)

if n_common != EXPECTED_COMMON_CELLS:
    raise RuntimeError(
        f"Expected {EXPECTED_COMMON_CELLS} common cells, "
        f"got {n_common}"
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
        "Unexpected coordinate shapes: "
        f"lat={lat.shape}, lon={lon.shape}"
    )


# ============================================================
# PHYSICAL OROGRAPHY
# ============================================================

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

orography[
    ~valid
] = np.nan


# ============================================================
# FINAL SEEDED CUBES
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

if N != EXPECTED_DAYS:
    raise RuntimeError(
        f"Expected {EXPECTED_DAYS} days, got {N}"
    )


if N % 365 != 0:
    raise RuntimeError(
        "Expected complete 365-day years."
    )


NY = N // 365

if NY != EXPECTED_YEARS:
    raise RuntimeError(
        f"Expected {EXPECTED_YEARS} years, got {NY}"
    )


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


# ============================================================
# EXACT 365-DAY NO-LEAP CALENDAR
# ============================================================

MONTH_LENGTHS = np.array([
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
], dtype=int)


if int(
    MONTH_LENGTHS.sum()
) != 365:
    raise RuntimeError(
        "Month lengths do not sum to 365."
    )


month_of_doy = np.empty(
    365,
    dtype=np.int16,
)

cursor = 0

for month, ndays in enumerate(
    MONTH_LENGTHS,
    start=1,
):

    month_of_doy[
        cursor:
        cursor + ndays
    ] = month

    cursor += ndays


YEAR_VALUES = np.arange(
    START_YEAR,
    END_YEAR + 1,
    dtype=np.int16,
)


year_of_time = np.repeat(
    YEAR_VALUES,
    365,
)

month_of_time = np.tile(
    month_of_doy,
    NY,
)


if len(
    year_of_time
) != N:
    raise RuntimeError(
        "Calendar length does not match prediction cubes."
    )


# ============================================================
# SEASON DEFINITIONS
# ============================================================

DJF_YEARS = np.arange(
    2016,
    2101,
    dtype=int,
)

JJA_YEARS = np.arange(
    2015,
    2101,
    dtype=int,
)


if len(
    DJF_YEARS
) != 85:
    raise RuntimeError(
        "Expected 85 complete DJF seasons."
    )


if len(
    JJA_YEARS
) != 86:
    raise RuntimeError(
        "Expected 86 complete JJA seasons."
    )


def season_mask(
    season,
    season_year,
):
    if season == "DJF":

        return (
            (
                (year_of_time == season_year - 1)
                & (month_of_time == 12)
            )
            |
            (
                (year_of_time == season_year)
                &
                (
                    (month_of_time == 1)
                    | (month_of_time == 2)
                )
            )
        )

    if season == "JJA":

        return (
            (year_of_time == season_year)
            &
            (
                (month_of_time == 6)
                | (month_of_time == 7)
                | (month_of_time == 8)
            )
        )

    raise ValueError(
        season
    )


# ============================================================
# POOLED MONTH-OF-YEAR MASKS
#
# These reproduce the seasonal pooled diagnostics:
#
# DJF = all Dec + Jan + Feb values inside 2015-2100.
# JJA = all Jun + Jul + Aug values inside 2015-2100.
#
# The season-resolved series below uses complete seasons.
# ============================================================

DJF_POOLED_MASK = (
    (month_of_time == 12)
    | (month_of_time == 1)
    | (month_of_time == 2)
)

JJA_POOLED_MASK = (
    (month_of_time == 6)
    | (month_of_time == 7)
    | (month_of_time == 8)
)


expected_djf_pooled_days = (
    86
    * (
        31
        + 31
        + 28
    )
)

expected_jja_pooled_days = (
    86
    * (
        30
        + 31
        + 31
    )
)


if int(
    DJF_POOLED_MASK.sum()
) != expected_djf_pooled_days:
    raise RuntimeError(
        "Unexpected pooled DJF day count."
    )


if int(
    JJA_POOLED_MASK.sum()
) != expected_jja_pooled_days:
    raise RuntimeError(
        "Unexpected pooled JJA day count."
    )


# ============================================================
# COMPLETE-SEASON CALENDAR AUDIT
# ============================================================

for yy in DJF_YEARS:

    n = int(
        season_mask(
            "DJF",
            int(yy),
        ).sum()
    )

    if n != 90:
        raise RuntimeError(
            f"DJF {yy} has {n} days, expected 90."
        )


for yy in JJA_YEARS:

    n = int(
        season_mask(
            "JJA",
            int(yy),
        ).sum()
    )

    if n != 92:
        raise RuntimeError(
            f"JJA {yy} has {n} days, expected 92."
        )


# ============================================================
# DATA HELPERS
# ============================================================

def take_valid(
    cube,
    time_mask,
):
    arr = np.asarray(
        cube[
            time_mask
        ],
        dtype=np.float32,
    )

    return arr[
        :,
        valid
    ]


def percentile_map_flat(
    cube,
    time_mask,
    q,
):
    vals = take_valid(
        cube,
        time_mask,
    )

    result = np.percentile(
        vals,
        q,
        axis=0,
    )

    del vals

    return np.asarray(
        result,
        dtype=np.float64,
    )


def absolute_error_advantage(
    reference,
    topo_values,
    unet_values,
):
    """
    Positive value:
    TopoFusion-Res has lower absolute error than U-Net.
    """

    return (
        np.abs(
            unet_values
            - reference
        )
        -
        np.abs(
            topo_values
            - reference
        )
    )


def seasonal_domain_mean_percentile_series(
    cube,
    season,
    season_years,
    q,
):
    series = []

    for yy in season_years:

        mask = season_mask(
            season,
            int(yy),
        )

        vals = take_valid(
            cube,
            mask,
        )

        qmap = np.percentile(
            vals,
            q,
            axis=0,
        )

        series.append(
            float(
                np.mean(
                    qmap
                )
            )
        )

        del vals

    return np.asarray(
        series,
        dtype=np.float64,
    )


# ============================================================
# INPUT REPORT
# ============================================================

print_block(
    "FIGURE 10 V3 — SEASONAL THERMAL TAIL FIDELITY"
)

print(
    "Calendar          : 365-day no-leap"
)

print(
    f"Period            : {START_YEAR}-{END_YEAR}"
)

print(
    f"Days              : {N}"
)

print(
    f"Years             : {NY}"
)

print(
    f"Common cells      : {n_common}"
)

print(
    f"Total pairs       : {N * n_common}"
)

print()

print(
    "Pooled map support"
)

print(
    f"  DJF months       : "
    f"{int(DJF_POOLED_MASK.sum())} days"
)

print(
    f"  JJA months       : "
    f"{int(JJA_POOLED_MASK.sum())} days"
)

print()

print(
    "Complete-season series"
)

print(
    f"  DJF              : "
    f"{DJF_YEARS[0]}-{DJF_YEARS[-1]} "
    f"(n={len(DJF_YEARS)}, 90 days each)"
)

print(
    f"  JJA              : "
    f"{JJA_YEARS[0]}-{JJA_YEARS[-1]} "
    f"(n={len(JJA_YEARS)}, 92 days each)"
)

print()


# ============================================================
# POOLED SEASONAL MAPS
# ============================================================

print_block(
    "POOLED SEASONAL MAP DIAGNOSTICS"
)


# ------------------------------------------------------------
# DJF p99
# ------------------------------------------------------------

rcm_djf_p99 = percentile_map_flat(
    rcm,
    DJF_POOLED_MASK,
    99,
)

topo_djf_p99 = percentile_map_flat(
    topo,
    DJF_POOLED_MASK,
    99,
)

unet_djf_p99 = percentile_map_flat(
    unet,
    DJF_POOLED_MASK,
    99,
)


djf_topo_abs_error = np.abs(
    topo_djf_p99
    - rcm_djf_p99
)

djf_unet_abs_error = np.abs(
    unet_djf_p99
    - rcm_djf_p99
)


djf_adv_flat = (
    djf_unet_abs_error
    - djf_topo_abs_error
)

djf_adv_map = restore_map(
    djf_adv_flat,
    valid,
)


djf_topo_mae = float(
    np.mean(
        djf_topo_abs_error
    )
)

djf_unet_mae = float(
    np.mean(
        djf_unet_abs_error
    )
)

djf_mean_adv = float(
    np.mean(
        djf_adv_flat
    )
)

djf_fraction_positive = float(
    np.mean(
        djf_adv_flat > 0
    )
)


# ------------------------------------------------------------
# JJA p01
# ------------------------------------------------------------

rcm_jja_p01 = percentile_map_flat(
    rcm,
    JJA_POOLED_MASK,
    1,
)

topo_jja_p01 = percentile_map_flat(
    topo,
    JJA_POOLED_MASK,
    1,
)

unet_jja_p01 = percentile_map_flat(
    unet,
    JJA_POOLED_MASK,
    1,
)


jja_topo_abs_error = np.abs(
    topo_jja_p01
    - rcm_jja_p01
)

jja_unet_abs_error = np.abs(
    unet_jja_p01
    - rcm_jja_p01
)


jja_adv_flat = (
    jja_unet_abs_error
    - jja_topo_abs_error
)

jja_adv_map = restore_map(
    jja_adv_flat,
    valid,
)


jja_topo_mae = float(
    np.mean(
        jja_topo_abs_error
    )
)

jja_unet_mae = float(
    np.mean(
        jja_unet_abs_error
    )
)

jja_mean_adv = float(
    np.mean(
        jja_adv_flat
    )
)

jja_fraction_positive = float(
    np.mean(
        jja_adv_flat > 0
    )
)


print(
    "DJF p99"
)

print(
    f"  Topo spatial MAE = "
    f"{djf_topo_mae:.4f} C"
)

print(
    f"  U-Net spatial MAE= "
    f"{djf_unet_mae:.4f} C"
)

print(
    f"  mean advantage   = "
    f"{djf_mean_adv:.4f} C"
)

print(
    f"  Topo closer      = "
    f"{100*djf_fraction_positive:.2f}%"
)

print()


print(
    "JJA p01"
)

print(
    f"  Topo spatial MAE = "
    f"{jja_topo_mae:.4f} C"
)

print(
    f"  U-Net spatial MAE= "
    f"{jja_unet_mae:.4f} C"
)

print(
    f"  mean advantage   = "
    f"{jja_mean_adv:.4f} C"
)

print(
    f"  Topo closer      = "
    f"{100*jja_fraction_positive:.2f}%"
)

print()


# ============================================================
# SEASON-RESOLVED DOMAIN-MEAN SERIES
# ============================================================

print_block(
    "SEASON-RESOLVED DOMAIN-MEAN SERIES"
)


rcm_djf_series = (
    seasonal_domain_mean_percentile_series(
        rcm,
        "DJF",
        DJF_YEARS,
        99,
    )
)

topo_djf_series = (
    seasonal_domain_mean_percentile_series(
        topo,
        "DJF",
        DJF_YEARS,
        99,
    )
)

unet_djf_series = (
    seasonal_domain_mean_percentile_series(
        unet,
        "DJF",
        DJF_YEARS,
        99,
    )
)


rcm_jja_series = (
    seasonal_domain_mean_percentile_series(
        rcm,
        "JJA",
        JJA_YEARS,
        1,
    )
)

topo_jja_series = (
    seasonal_domain_mean_percentile_series(
        topo,
        "JJA",
        JJA_YEARS,
        1,
    )
)

unet_jja_series = (
    seasonal_domain_mean_percentile_series(
        unet,
        "JJA",
        JJA_YEARS,
        1,
    )
)


djf_series_mae_topo = float(
    np.mean(
        np.abs(
            topo_djf_series
            - rcm_djf_series
        )
    )
)

djf_series_mae_unet = float(
    np.mean(
        np.abs(
            unet_djf_series
            - rcm_djf_series
        )
    )
)


jja_series_mae_topo = float(
    np.mean(
        np.abs(
            topo_jja_series
            - rcm_jja_series
        )
    )
)

jja_series_mae_unet = float(
    np.mean(
        np.abs(
            unet_jja_series
            - rcm_jja_series
        )
    )
)


print(
    "DJF p99 — domain-mean series MAE"
)

print(
    f"  TopoFusion-Res = "
    f"{djf_series_mae_topo:.4f} C"
)

print(
    f"  U-Net          = "
    f"{djf_series_mae_unet:.4f} C"
)

print()


print(
    "JJA p01 — domain-mean series MAE"
)

print(
    f"  TopoFusion-Res = "
    f"{jja_series_mae_topo:.4f} C"
)

print(
    f"  U-Net          = "
    f"{jja_series_mae_unet:.4f} C"
)

print()


# ============================================================
# FIGURE
# ============================================================

lim = symmetric_limit(
    djf_adv_map,
    jja_adv_map,
)


norm = TwoSlopeNorm(
    vmin=-lim,
    vcenter=0.0,
    vmax=lim,
)


fig = plt.figure(
    figsize=(
        13.0,
        9.3,
    )
)


gs = fig.add_gridspec(
    3,
    2,

    height_ratios=[
        1.0,
        0.055,
        1.0,
    ],

    left=0.07,
    right=0.97,
    bottom=0.07,
    top=0.97,

    wspace=0.20,
    hspace=0.27,
)


ax1 = fig.add_subplot(
    gs[0, 0]
)

ax2 = fig.add_subplot(
    gs[0, 1]
)

cax = fig.add_subplot(
    gs[1, :]
)

ax3 = fig.add_subplot(
    gs[2, 0]
)

ax4 = fig.add_subplot(
    gs[2, 1]
)


# ============================================================
# MAPS
# ============================================================

im1 = ax1.pcolormesh(
    LON,
    LAT,
    masked(
        djf_adv_map,
        valid,
    ),

    shading="auto",

    cmap="RdBu_r",

    norm=norm,
)


im2 = ax2.pcolormesh(
    LON,
    LAT,
    masked(
        jja_adv_map,
        valid,
    ),

    shading="auto",

    cmap="RdBu_r",

    norm=norm,
)


for ax in [
    ax1,
    ax2,
]:

    ax.contour(
        LON,
        LAT,
        orography,

        levels=[
            500,
            1500,
            3000,
        ],

        colors="0.38",

        linewidths=0.65,

        alpha=0.65,
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )


ax1.set_title(
    "(a) DJF p99 absolute-error advantage",
    pad=7,
)


ax2.set_title(
    "(b) JJA p01 absolute-error advantage",
    pad=7,
)


ax1.text(
    0.025,
    0.035,

    (
        f"mean Δ = "
        f"{djf_mean_adv:.3f} °C\n"
        f"Topo closer = "
        f"{100*djf_fraction_positive:.1f}%"
    ),

    transform=ax1.transAxes,

    ha="left",
    va="bottom",

    fontsize=9,

    bbox=dict(
        facecolor="white",
        edgecolor="0.55",
        alpha=0.90,
        boxstyle="round,pad=0.25",
    ),
)


ax2.text(
    0.025,
    0.035,

    (
        f"mean Δ = "
        f"{jja_mean_adv:.3f} °C\n"
        f"Topo closer = "
        f"{100*jja_fraction_positive:.1f}%"
    ),

    transform=ax2.transAxes,

    ha="left",
    va="bottom",

    fontsize=9,

    bbox=dict(
        facecolor="white",
        edgecolor="0.55",
        alpha=0.90,
        boxstyle="round,pad=0.25",
    ),
)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(
    im2,
    cax=cax,
    orientation="horizontal",
)


cbar.set_label(
    (
        "Absolute-error advantage: "
        "U-Net − TopoFusion-Res "
        "(°C; positive = lower error for TopoFusion-Res)"
    )
)


# ============================================================
# TIME-SERIES STYLE HELPER
# ============================================================

def plot_season_series(
    ax,
    years,
    rcm_series,
    topo_series,
    unet_series,
    title,
    mae_topo,
    mae_unet,
):

    # --------------------------------------------------------
    # Continuous lines + small regular markers
    # --------------------------------------------------------

    ax.plot(
        years,
        rcm_series,

        color=COLOR_RCM,

        linewidth=1.45,

        marker="o",

        markersize=2.7,

        markevery=2,

        label="CNRM-ALADIN64",

        zorder=3,
    )


    ax.plot(
        years,
        topo_series,

        color=COLOR_TOPO,

        linewidth=1.45,

        marker="o",

        markersize=2.7,

        markevery=2,

        label="TopoFusion-Res",

        zorder=3,
    )


    ax.plot(
        years,
        unet_series,

        color=COLOR_UNET,

        linewidth=1.45,

        marker="o",

        markersize=2.7,

        markevery=2,

        label="U-Net CMIP-CMIP",

        zorder=3,
    )


    # --------------------------------------------------------
    # Larger highlighted markers every ~20 years
    # --------------------------------------------------------

    for yy in HIGHLIGHT_YEARS:

        idx = find_year_index(
            years,
            yy,
        )

        if idx is None:
            continue


        for series, color in [
            (
                rcm_series,
                COLOR_RCM,
            ),

            (
                topo_series,
                COLOR_TOPO,
            ),

            (
                unet_series,
                COLOR_UNET,
            ),
        ]:

            ax.scatter(
                [yy],
                [
                    series[
                        idx
                    ]
                ],

                s=34,

                color=color,

                edgecolors="white",

                linewidths=0.8,

                zorder=6,
            )


    # --------------------------------------------------------
    # Numeric annotations:
    # CNRM-ALADIN64 reference at 2020, 2060, 2100 only.
    # --------------------------------------------------------

    annotation_offsets = {
        2020:
            (
                12,
                10,
            ),

        2060:
            (
                0,
                11,
            ),

        2100:
            (
                -18,
                10,
            ),
    }


    for yy in ANNOTATE_YEARS:

        idx = find_year_index(
            years,
            yy,
        )

        if idx is None:
            continue


        dx, dy = (
            annotation_offsets[
                yy
            ]
        )


        value = float(
            rcm_series[
                idx
            ]
        )


        ax.annotate(
            f"{yy}\n{value:.1f} °C",

            xy=(
                yy,
                value,
            ),

            xytext=(
                dx,
                dy,
            ),

            textcoords="offset points",

            ha="center",
            va="bottom",

            fontsize=8,

            color=COLOR_RCM,

            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=COLOR_RCM,
                alpha=0.83,
                linewidth=0.7,
            ),

            arrowprops=dict(
                arrowstyle="-",
                color=COLOR_RCM,
                linewidth=0.6,
                alpha=0.65,
            ),

            zorder=8,
        )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_title(
        title,
        pad=7,
    )


    ax.set_xlabel(
        "Season year"
    )


    ax.set_ylabel(
        "Temperature (°C)"
    )


    ax.grid(
        alpha=0.20,
        linewidth=0.6,
    )


    # --------------------------------------------------------
    # Overall series error
    # --------------------------------------------------------

    ax.text(
        0.975,
        0.04,

        (
            "MAE of domain-mean series\n"
            f"Topo: {mae_topo:.3f} °C\n"
            f"U-Net: {mae_unet:.3f} °C"
        ),

        transform=ax.transAxes,

        ha="right",
        va="bottom",

        fontsize=8.7,

        bbox=dict(
            facecolor="white",
            edgecolor="0.55",
            alpha=0.90,
            boxstyle="round,pad=0.25",
        ),
    )


# ============================================================
# PANEL (c)
# ============================================================

plot_season_series(
    ax=ax3,

    years=DJF_YEARS,

    rcm_series=rcm_djf_series,

    topo_series=topo_djf_series,

    unet_series=unet_djf_series,

    title="(c) Domain-mean DJF p99 by season",

    mae_topo=djf_series_mae_topo,

    mae_unet=djf_series_mae_unet,
)


ax3.legend(
    loc="upper left",
    frameon=True,
)


# ============================================================
# PANEL (d)
# ============================================================

plot_season_series(
    ax=ax4,

    years=JJA_YEARS,

    rcm_series=rcm_jja_series,

    topo_series=topo_jja_series,

    unet_series=unet_jja_series,

    title="(d) Domain-mean JJA p01 by season",

    mae_topo=jja_series_mae_topo,

    mae_unet=jja_series_mae_unet,
)


# ============================================================
# SAVE FIGURE
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
# SAVE TABLE
# ============================================================

with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "diagnostic",
        "season_year",
        "CNRM_ALADIN64_C",
        "TopoFusion_Res_C",
        "UNet_CMIP_CMIP_C",
    ])


    for (
        yy,
        rr,
        tt,
        uu,
    ) in zip(
        DJF_YEARS,
        rcm_djf_series,
        topo_djf_series,
        unet_djf_series,
    ):

        writer.writerow([
            "DJF_p99",
            int(yy),
            float(rr),
            float(tt),
            float(uu),
        ])


    for (
        yy,
        rr,
        tt,
        uu,
    ) in zip(
        JJA_YEARS,
        rcm_jja_series,
        topo_jja_series,
        unet_jja_series,
    ):

        writer.writerow([
            "JJA_p01",
            int(yy),
            float(rr),
            float(tt),
            float(uu),
        ])


# ============================================================
# METADATA
# ============================================================

metadata = {
    "figure":
        "Figure10_seasonal_extremes_v3",

    "scenario":
        "SSP3-7.0",

    "period":
        "2015-2100",

    "calendar":
        "365-day no-leap",

    "days":
        int(N),

    "years":
        int(NY),

    "common_cells":
        int(n_common),

    "map_support": {
        "DJF_p99": {
            "definition":
                "All December, January and February days "
                "within 2015-2100.",

            "n_days":
                int(
                    DJF_POOLED_MASK.sum()
                ),

            "topofusion_spatial_mae_C":
                djf_topo_mae,

            "unet_spatial_mae_C":
                djf_unet_mae,

            "mean_advantage_C":
                djf_mean_adv,

            "fraction_cells_topofusion_closer":
                djf_fraction_positive,
        },

        "JJA_p01": {
            "definition":
                "All June, July and August days "
                "within 2015-2100.",

            "n_days":
                int(
                    JJA_POOLED_MASK.sum()
                ),

            "topofusion_spatial_mae_C":
                jja_topo_mae,

            "unet_spatial_mae_C":
                jja_unet_mae,

            "mean_advantage_C":
                jja_mean_adv,

            "fraction_cells_topofusion_closer":
                jja_fraction_positive,
        },
    },

    "season_resolved_series": {
        "DJF_p99": {
            "first_season":
                int(
                    DJF_YEARS[0]
                ),

            "last_season":
                int(
                    DJF_YEARS[-1]
                ),

            "n_seasons":
                int(
                    len(
                        DJF_YEARS
                    )
                ),

            "topofusion_mae_C":
                djf_series_mae_topo,

            "unet_mae_C":
                djf_series_mae_unet,
        },

        "JJA_p01": {
            "first_season":
                int(
                    JJA_YEARS[0]
                ),

            "last_season":
                int(
                    JJA_YEARS[-1]
                ),

            "n_seasons":
                int(
                    len(
                        JJA_YEARS
                    )
                ),

            "topofusion_mae_C":
                jja_series_mae_topo,

            "unet_mae_C":
                jja_series_mae_unet,
        },
    },

    "highlight_years":
        HIGHLIGHT_YEARS,

    "annotated_reference_years":
        ANNOTATE_YEARS,

    "important_note": (
        "The top-row maps reproduce the pooled seasonal "
        "percentile diagnostics using all calendar months "
        "belonging to DJF or JJA within 2015-2100. "
        "The lower panels use complete season-resolved "
        "percentiles: DJF 2016-2100 and JJA 2015-2100. "
        "Numeric callouts identify CNRM-ALADIN64 reference "
        "values at selected years. These curves are distinct "
        "from the season-resolved spatial MAE series used "
        "for moving-block bootstrap inference."
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


# ============================================================
# OUTPUT REPORT
# ============================================================

print()
print_block(
    "OUTPUTS"
)

print(
    OUT_PNG
)

print(
    OUT_PDF
)

print(
    OUT_CSV
)

print(
    OUT_JSON
)

print()
print(
    "DONE"
)
