from pathlib import Path
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# PATHS
#
# BASED DIRECTLY ON:
#   41_make_publication_figure03.py
# ============================================================

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[2]

FINAL = EXP / "paper_validation_final_seed20260813"
DIA = FINAL / "diagnostics"

IN_NPZ = (
    DIA
    / "38_final_seeded_spatial_rmse_maps.npz"
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

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = (
    OUT
    / "Figure03_residual_refinement_with_contours.png"
)

PDF_OUT = (
    OUT
    / "Figure03_residual_refinement_with_contours.pdf"
)

JSON_OUT = (
    OUT
    / "Figure03_residual_refinement_with_contours_metadata.json"
)


# ============================================================
# INPUT AUDIT
# ============================================================

for p in [
    IN_NPZ,
    STATIC_FILE,
    STATIC_NAMES_FILE,
    STATS_FILE,
]:
    if not p.exists():
        raise FileNotFoundError(
            p
        )


# ============================================================
# STYLE
#
# Preserved from the approved Figure 3.
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "figure.dpi": 120,
    "savefig.dpi": 300,

    "axes.linewidth": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# LOAD FINAL MAPS + EXACT FINAL COORDINATES
#
# lat/lon are already present in the final seeded diagnostic.
# This avoids reopening the U-Net NetCDF and guarantees exactly
# the same 64x64 coordinates used by the final diagnostics.
# ============================================================

with np.load(
    IN_NPZ,
    allow_pickle=False,
) as z:

    print(
        "RMSE keys:",
        z.files,
    )

    required = [
        "valid",
        "rmse_base_full",
        "rmse_res_full",
        "base_minus_res",
        "lat",
        "lon",
    ]

    missing = [
        k
        for k in required
        if k not in z.files
    ]

    if missing:
        raise RuntimeError(
            "Missing required keys:\n"
            + "\n".join(
                missing
            )
        )

    valid = np.asarray(
        z["valid"]
    ).astype(
        bool
    )

    rmse_base = np.asarray(
        z["rmse_base_full"],
        dtype=np.float64,
    )

    rmse_res = np.asarray(
        z["rmse_res_full"],
        dtype=np.float64,
    )

    delta = np.asarray(
        z["base_minus_res"],
        dtype=np.float64,
    )

    LAT = np.asarray(
        z["lat"],
        dtype=np.float64,
    )

    LON = np.asarray(
        z["lon"],
        dtype=np.float64,
    )


H, W = valid.shape


if valid.shape != (
    64,
    64,
):
    raise RuntimeError(
        f"Expected valid shape (64,64), got {valid.shape}"
    )

if LAT.shape != (
    H,
    W,
):
    raise RuntimeError(
        f"Unexpected LAT shape: {LAT.shape}"
    )

if LON.shape != (
    H,
    W,
):
    raise RuntimeError(
        f"Unexpected LON shape: {LON.shape}"
    )


print("=" * 90)
print("FIGURE 3 — RESIDUAL REFINEMENT + TOPOGRAPHIC CONTEXT")
print("=" * 90)

print(
    "RMSE source :",
    IN_NPZ,
)

print(
    "map shape   :",
    valid.shape,
)

print(
    "valid cells :",
    int(
        valid.sum()
    ),
)

print(
    "lat shape   :",
    LAT.shape,
)

print(
    "lon shape   :",
    LON.shape,
)


# ============================================================
# MASKING
# ============================================================

def masked(
    a,
):
    x = np.asarray(
        a,
        dtype=np.float64,
    ).copy()

    x[
        ~valid
    ] = np.nan

    return x


rmse_base_m = masked(
    rmse_base
)

rmse_res_m = masked(
    rmse_res
)

delta_m = masked(
    delta
)


# ============================================================
# DISPLAY SCALES
#
# EXACT SAME p98 PHILOSOPHY AS FINAL FIGURE 3.
# ============================================================

DISPLAY_PERCENTILE = 98.0


def finite_valid_values(
    a,
):
    v = np.asarray(
        a,
        dtype=np.float64,
    )[valid]

    return v[
        np.isfinite(
            v
        )
    ]


rmse_values = np.concatenate([
    finite_valid_values(
        rmse_base
    ),

    finite_valid_values(
        rmse_res
    ),
])


delta_values = finite_valid_values(
    delta
)

abs_delta_values = np.abs(
    delta_values
)


rmse_vmax = float(
    np.percentile(
        rmse_values,
        DISPLAY_PERCENTILE,
    )
)


delta_lim = float(
    np.percentile(
        abs_delta_values,
        DISPLAY_PERCENTILE,
    )
)


rmse_fraction_above = float(
    np.mean(
        rmse_values
        > rmse_vmax
    )
)


delta_fraction_beyond = float(
    np.mean(
        abs_delta_values
        > delta_lim
    )
)


print()
print(
    "Visualization scales (display only):"
)

print(
    f"  RMSE vmax (p{DISPLAY_PERCENTILE:g}) = "
    f"{rmse_vmax:.12f}"
)

print(
    f"  Delta |limit| (p{DISPLAY_PERCENTILE:g}) = "
    f"{delta_lim:.12f}"
)


# ============================================================
# RECONSTRUCT EXACT FINE-SCALE PHYSICAL OROGRAPHY
# ============================================================

with open(
    STATIC_NAMES_FILE,
    "r",
) as f:

    static_names = json.load(
        f
    )


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


if len(
    static_names
) != 6:

    raise RuntimeError(
        f"Expected six static channels, got {static_names}"
    )


channel0_name = str(
    static_names[0]
).lower()


if (
    "orog" not in channel0_name
    and "elev" not in channel0_name
):
    raise RuntimeError(
        "Channel 0 is not identified as fine-scale orography."
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
        "Expected static_features_64x64.npy "
        f"shape (6,64,64), got {fine_static.shape}"
    )


orog_norm = fine_static[0]


with open(
    STATS_FILE,
    "r",
) as f:

    normalization_stats = json.load(
        f
    )


try:
    orog_stats = (
        normalization_stats[
            "static_64x64"
        ][
            "orography_64"
        ]
    )

except Exception as exc:

    raise RuntimeError(
        "Could not locate "
        "static_64x64.orography_64 "
        "in normalization_stats.json"
    ) from exc


orog_mean = float(
    orog_stats["mean"]
)

orog_std = float(
    orog_stats["std"]
)


elevation = (
    orog_norm
    * orog_std
    + orog_mean
)


# Remove only tiny numerical negatives.
elevation[
    (elevation < 0)
    & (elevation > -1e-2)
] = 0.0


elevation_plot = np.asarray(
    elevation,
    dtype=np.float64,
).copy()

elevation_plot[
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
    "Stats block      :",
    "static_64x64.orography_64",
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
    float(
        np.nanmin(
            elevation_plot
        )
    ),
    "..",
    float(
        np.nanmax(
            elevation_plot
        )
    ),
    "m",
)


# ============================================================
# TOPOGRAPHIC CONVENTION
#
# EXACTLY SAME CONVENTION AS FINAL FIGURES 2 AND 4.
# ============================================================

CONTOUR_LEVELS = [
    500.0,
    1500.0,
    3000.0,
]


TOPO_CONTOUR_COLOR = "#2B2B2B"
TOPO_CONTOUR_WIDTH = 0.82
TOPO_CONTOUR_ALPHA = 0.80


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


def add_topographic_contours(
    ax,
):
    """
    Same geographic-reference contours used in Figures 2 and 4.
    """

    return ax.contour(
        LON,
        LAT,
        elevation_plot,

        levels=CONTOUR_LEVELS,

        colors=TOPO_CONTOUR_COLOR,

        linewidths=TOPO_CONTOUR_WIDTH,

        alpha=TOPO_CONTOUR_ALPHA,

        zorder=10,
    )


def get_band_mask(
    low=None,
    high=None,
):
    m = (
        valid
        & np.isfinite(
            elevation
        )
    )

    if low is not None:
        m &= (
            elevation
            >= low
        )

    if high is not None:
        m &= (
            elevation
            < high
        )

    return m


def representative_band_position(
    mask,
):
    ii, jj = np.where(
        mask
    )

    if ii.size == 0:
        return None

    lons = LON[
        ii,
        jj,
    ]

    lats = LAT[
        ii,
        jj,
    ]

    lon_med = float(
        np.nanmedian(
            lons
        )
    )

    lat_med = float(
        np.nanmedian(
            lats
        )
    )

    coslat = np.cos(
        np.deg2rad(
            lat_med
        )
    )

    dist2 = (
        (
            (
                lons
                - lon_med
            )
            * coslat
        ) ** 2
        +
        (
            lats
            - lat_med
        ) ** 2
    )

    k = int(
        np.nanargmin(
            dist2
        )
    )

    return (
        float(
            lons[k]
        ),

        float(
            lats[k]
        ),
    )


def add_altitudinal_band_labels(
    ax,
):
    """
    Same altitude-band annotations as final Figures 2 and 4.
    Labels are shown only on panel (c).
    """

    positions = {}

    for band in ALTITUDE_BANDS:

        band_mask = get_band_mask(
            low=band["low"],
            high=band["high"],
        )

        pos = representative_band_position(
            band_mask
        )

        if pos is None:
            continue

        x, y = pos

        positions[
            band["label"]
        ] = {
            "lon":
                x,

            "lat":
                y,

            "n_cells":
                int(
                    band_mask.sum()
                ),
        }

        ax.text(
            x,
            y,
            band["label"],

            ha="center",
            va="center",

            fontsize=6.5,
            fontweight="semibold",

            color="#222222",

            bbox=dict(
                boxstyle="round,pad=0.16",

                facecolor="white",
                edgecolor="#666666",

                linewidth=0.45,

                alpha=0.76,
            ),

            zorder=30,
            clip_on=True,
        )

    return positions


# ============================================================
# SUMMARY VALUES
# ============================================================

v = finite_valid_values(
    delta
)


summary = {
    "figure":
        "Figure 3",

    "comparison":
        (
            "Residual refinement ablation: "
            "TopoFusion-Base versus TopoFusion-Res"
        ),

    "support": {
        "n_common_cells":
            int(
                valid.sum()
            ),
    },

    "coordinate_source":
        str(
            IN_NPZ
        ),

    "sign_convention":
        (
            "Delta RMSE = TopoFusion-Base RMSE minus "
            "TopoFusion-Res RMSE. Positive values indicate "
            "lower RMSE after residual refinement."
        ),

    "spatial_delta_rmse_C": {
        "mean":
            float(
                np.mean(
                    v
                )
            ),

        "median":
            float(
                np.median(
                    v
                )
            ),

        "p10":
            float(
                np.percentile(
                    v,
                    10,
                )
            ),

        "p90":
            float(
                np.percentile(
                    v,
                    90,
                )
            ),

        "fraction_positive":
            float(
                np.mean(
                    v > 0
                )
            ),

        "fraction_negative":
            float(
                np.mean(
                    v < 0
                )
            ),
    },

    "visualization": {
        "display_percentile":
            DISPLAY_PERCENTILE,

        "rmse_vmax_C":
            rmse_vmax,

        "delta_symmetric_limit_C":
            delta_lim,

        "rmse_fraction_above_display_limit":
            rmse_fraction_above,

        "delta_fraction_beyond_display_limit":
            delta_fraction_beyond,

        "note":
            (
                "Color limits use the spatial 98th percentile "
                "for visualization only. Raw maps and all "
                "reported statistics use complete finite values "
                "without clipping."
            ),
    },
}


# ============================================================
# FINAL PUBLICATION LAYOUT
#
# DIRECTLY PRESERVED FROM 41_make_publication_figure03.py
# ============================================================

mean_lat = float(
    np.nanmean(
        LAT[
            valid
        ]
    )
)


map_aspect = float(
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
    show_ylabel=True,
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
        map_aspect
    )

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8,
    )

    ax.grid(
        False
    )


cmap_rmse = plt.get_cmap(
    "viridis"
).copy()

cmap_rmse.set_bad(
    "white"
)


cmap_delta = plt.get_cmap(
    "RdBu_r"
).copy()

cmap_delta.set_bad(
    "white"
)


fig = plt.figure(
    figsize=(
        14.8,
        5.15,
    )
)


gs = fig.add_gridspec(
    1,
    6,

    width_ratios=[
        1.0,
        1.0,
        0.045,
        0.10,
        1.0,
        0.045,
    ],

    left=0.055,
    right=0.965,

    bottom=0.13,
    top=0.91,

    wspace=0.20,
)


ax0 = fig.add_subplot(
    gs[0, 0]
)

ax1 = fig.add_subplot(
    gs[0, 1]
)

cax_rmse = fig.add_subplot(
    gs[0, 2]
)

spacer = fig.add_subplot(
    gs[0, 3]
)

ax2 = fig.add_subplot(
    gs[0, 4]
)

cax_delta = fig.add_subplot(
    gs[0, 5]
)


spacer.axis(
    "off"
)


# ============================================================
# PANEL (a): TopoFusion-Base
# ============================================================

im0 = ax0.pcolormesh(
    LON,
    LAT,
    rmse_base_m,

    shading="auto",

    rasterized=True,

    cmap=cmap_rmse,

    vmin=0.0,
    vmax=rmse_vmax,
)


ax0.set_title(
    "(a) TopoFusion-Base"
)

format_geo_axis(
    ax0,
    show_ylabel=True,
)

add_topographic_contours(
    ax0
)


# ============================================================
# PANEL (b): TopoFusion-Res
# ============================================================

im1 = ax1.pcolormesh(
    LON,
    LAT,
    rmse_res_m,

    shading="auto",

    rasterized=True,

    cmap=cmap_rmse,

    vmin=0.0,
    vmax=rmse_vmax,
)


ax1.set_title(
    "(b) TopoFusion-Res"
)

format_geo_axis(
    ax1,
    show_ylabel=False,
)

add_topographic_contours(
    ax1
)


# ============================================================
# SHARED RMSE COLORBAR
# ============================================================

cb_rmse = fig.colorbar(
    im1,

    cax=cax_rmse,

    extend="max",
    extendfrac=0.035,
)


cb_rmse.set_label(
    r"RMSE ($^\circ$C)"
)


# ============================================================
# PANEL (c): DELTA RMSE
# ============================================================

norm = TwoSlopeNorm(
    vmin=-delta_lim,
    vcenter=0.0,
    vmax=delta_lim,
)


im2 = ax2.pcolormesh(
    LON,
    LAT,
    delta_m,

    shading="auto",

    rasterized=True,

    cmap=cmap_delta,

    norm=norm,
)


ax2.set_title(
    r"(c) $\Delta$RMSE = TopoFusion-Base $-$ TopoFusion-Res"
)


# Harmonized with final Figure 2:
# Latitude title only appears once, on panel (a).
format_geo_axis(
    ax2,
    show_ylabel=False,
)


add_topographic_contours(
    ax2
)


band_label_positions = (
    add_altitudinal_band_labels(
        ax2
    )
)


# ============================================================
# DELTA COLORBAR
# ============================================================

cb_delta = fig.colorbar(
    im2,

    cax=cax_delta,

    extend="both",
    extendfrac=0.035,
)


cb_delta.set_label(
    r"$\Delta$RMSE ($^\circ$C)"
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PNG_OUT,

    dpi=300,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(
    PDF_OUT,

    bbox_inches="tight",

    facecolor="white",
)


plt.close(
    fig
)


# ============================================================
# EXTEND METADATA WITH TOPOGRAPHIC CONTEXT
# ============================================================

summary[
    "topography_overlay"
] = {

    "panels": [
        "a",
        "b",
        "c",
    ],

    "physical_field":
        (
            "Fine-scale physical orography used to construct "
            "the Model-E static inputs."
        ),

    "normalized_source":
        str(
            STATIC_FILE
        ),

    "normalization_stats_source":
        str(
            STATS_FILE
        ),

    "normalization_block":
        "static_64x64.orography_64",

    "normalization_mean_m":
        orog_mean,

    "normalization_std_m":
        orog_std,

    "valid_elevation_min_m":
        float(
            np.nanmin(
                elevation_plot
            )
        ),

    "valid_elevation_max_m":
        float(
            np.nanmax(
                elevation_plot
            )
        ),

    "contour_levels_m": [
        500,
        1500,
        3000,
    ],

    "altitudinal_band_labels": [
        band["label"]
        for band in ALTITUDE_BANDS
    ],

    "band_labels_panel":
        "c",

    "band_label_positions":
        band_label_positions,

    "interpretation":
        (
            "Elevation contours and altitude ranges are "
            "shown as geographic context. The controlled "
            "ablation establishes the contribution of the "
            "residual-refinement stage; the topographic "
            "overlay is not itself evidence that orography "
            "causes the residual improvement."
        ),
}


summary[
    "outputs"
] = {

    "png":
        str(
            PNG_OUT
        ),

    "pdf":
        str(
            PDF_OUT
        ),
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
print("FIGURE 3 — FINAL TOPOGRAPHIC CONTEXT VERSION")
print("=" * 90)

print(
    "RMSE source :",
    IN_NPZ,
)

print(
    "Topo source :",
    STATIC_FILE,
)

print(
    "Stats source:",
    STATS_FILE,
)

print(
    "Topo range  :",
    f"{np.nanmin(elevation_plot):.2f}",
    "..",
    f"{np.nanmax(elevation_plot):.2f}",
    "m",
)

print(
    "Contours    :",
    "500 m, 1500 m, 3000 m",
)

print(
    "Band labels :",
    ", ".join(
        band["label"]
        for band in ALTITUDE_BANDS
    ),
)

print(
    "Label panel : (c)"
)


for label, info in band_label_positions.items():

    print(
        f"  {label:14s} -> "
        f"lon={info['lon']:.3f}, "
        f"lat={info['lat']:.3f}, "
        f"cells={info['n_cells']}"
    )


print()
print(
    "RMSE p98    :",
    rmse_vmax,
)

print(
    "Delta p98   :",
    delta_lim,
)

print()
print(
    "Mean ΔRMSE  :",
    summary[
        "spatial_delta_rmse_C"
    ][
        "mean"
    ],
)

print(
    "Median      :",
    summary[
        "spatial_delta_rmse_C"
    ][
        "median"
    ],
)

print(
    "Fraction ΔRMSE > 0:",
    summary[
        "spatial_delta_rmse_C"
    ][
        "fraction_positive"
    ],
)

print()
print("Saved:")
print(PNG_OUT)
print(PDF_OUT)
print(JSON_OUT)

print()
print("DONE")
