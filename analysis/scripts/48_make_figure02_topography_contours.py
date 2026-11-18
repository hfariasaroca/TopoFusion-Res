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
EXP = SCRIPT.parents[2]

FINAL = EXP / "analysis"
DIA = FINAL / "diagnostics"

RMSE_FILE = (
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

PNG = (
    OUT
    / "Figure02_spatial_rmse_with_contours.png"
)

PDF = (
    OUT
    / "Figure02_spatial_rmse_with_contours.pdf"
)

META = (
    OUT
    / "Figure02_spatial_rmse_with_contours_metadata.json"
)


# ============================================================
# INPUT AUDIT
# ============================================================

for p in [
    RMSE_FILE,
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
# Preserve the final Figure-2 publication style.
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    "figure.dpi": 120,
    "savefig.dpi": 300,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# LOAD FINAL SEEDED SPATIAL RMSE DIAGNOSTICS
# ============================================================

with np.load(
    RMSE_FILE,
    allow_pickle=False,
) as z:

    print(
        "RMSE keys:",
        z.files,
    )

    required = [
        "valid",
        "rmse_unet",
        "rmse_res_full",
        "unet_minus_res_full",
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
            "Missing keys in final RMSE diagnostics:\n"
            + "\n".join(
                missing
            )
        )

    valid = np.asarray(
        z["valid"]
    ).astype(
        bool
    )

    rmse_unet = np.asarray(
        z["rmse_unet"],
        dtype=np.float64,
    )

    rmse_full = np.asarray(
        z["rmse_res_full"],
        dtype=np.float64,
    )

    delta_unet = np.asarray(
        z["unet_minus_res_full"],
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
        f"Expected 64x64 valid mask, got {valid.shape}"
    )


if LAT.shape != (
    H,
    W,
):
    raise RuntimeError(
        f"Unexpected latitude shape: {LAT.shape}"
    )


if LON.shape != (
    H,
    W,
):
    raise RuntimeError(
        f"Unexpected longitude shape: {LON.shape}"
    )


print()
print(
    "Final common cells:",
    int(
        valid.sum()
    ),
)

print(
    "Latitude shape    :",
    LAT.shape,
)

print(
    "Longitude shape   :",
    LON.shape,
)


# ============================================================
# RECONSTRUCT EXACT PHYSICAL OROGRAPHY USED BY MODEL E
# ============================================================

with open(
    STATIC_NAMES_FILE,
    "r",
) as f:

    static_names = json.load(
        f
    )


if len(
    static_names
) != 6:

    raise RuntimeError(
        "Expected six fine-static feature names, "
        f"got {static_names}"
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


name0 = str(
    static_names[0]
).lower()


if (
    "orog" not in name0
    and "elev" not in name0
):
    raise RuntimeError(
        "Fine-static channel 0 is not identified "
        f"as orography/elevation: {static_names[0]}"
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
        "normalization_stats['static_64x64']['orography_64']"
    ) from exc


orog_mean = float(
    orog_stats["mean"]
)

orog_std = float(
    orog_stats["std"]
)


if (
    not np.isfinite(
        orog_mean
    )
    or not np.isfinite(
        orog_std
    )
    or orog_std <= 0
):
    raise RuntimeError(
        "Invalid physical-orography normalization statistics."
    )


elevation = (
    orog_norm
    * orog_std
    + orog_mean
)


# Remove negligible numerical negatives.
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
# FINAL TOPOGRAPHIC CONVENTION
#
# Same convention as final Figure 4:
#   500 m
#   1500 m
#   3000 m
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


# ============================================================
# HELPERS
# ============================================================

def masked(
    array,
):
    out = np.asarray(
        array,
        dtype=np.float64,
    ).copy()

    out[
        ~valid
    ] = np.nan

    return out


def common_rmse_vmax(
    a,
    b,
):
    vals = np.concatenate([
        np.asarray(
            a
        )[valid],

        np.asarray(
            b
        )[valid],
    ])

    vals = vals[
        np.isfinite(
            vals
        )
    ]

    return float(
        np.percentile(
            vals,
            98,
        )
    )


def symmetric_limit(
    a,
):
    vals = np.abs(
        np.asarray(
            a
        )[valid]
    )

    vals = vals[
        np.isfinite(
            vals
        )
    ]

    lim = float(
        np.percentile(
            vals,
            98,
        )
    )

    return max(
        lim,
        1e-8,
    )


def finish_map_axis(
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
        # Preserve latitude tick labels but avoid repeating the
        # axis title in panels (b) and (c). This leaves more
        # room around the shared RMSE colorbar.
        ax.set_ylabel(
            ""
        )


def add_topographic_contours(
    ax,
):
    """
    Same physical elevation contours on every panel.
    No line-value labels are drawn on the maps.
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
    mask = (
        valid
        & np.isfinite(
            elevation
        )
    )

    if low is not None:
        mask &= (
            elevation
            >= low
        )

    if high is not None:
        mask &= (
            elevation
            < high
        )

    return mask


def representative_band_position(
    mask,
):
    """
    Same positioning principle used for final Figure 4:
    median geographical position, snapped to an actual
    cell belonging to the altitude band.
    """

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
# DISPLAY SCALES
#
# Preserve exactly the publication logic already approved:
# p98 for visualization only.
# ============================================================

RMSE_VMAX = common_rmse_vmax(
    rmse_full,
    rmse_unet,
)

DELTA_LIM = symmetric_limit(
    delta_unet
)


print()
print(
    "RMSE p98  :",
    RMSE_VMAX,
)

print(
    "Delta p98 :",
    DELTA_LIM,
)


# ============================================================
# FIGURE
#
# Preserve the original Figure-2 layout.
# ============================================================

fig, axes = plt.subplots(
    1,
    3,

    figsize=(
        12.6,
        4.6,
    ),
)


# ============================================================
# PANEL (a)
# ============================================================

im0 = axes[0].pcolormesh(
    LON,
    LAT,
    masked(
        rmse_full
    ),

    shading="auto",

    cmap="viridis",

    vmin=0,
    vmax=RMSE_VMAX,
)


axes[0].set_title(
    "(a) TopoFusion-Res"
)

finish_map_axis(
    axes[0],
    show_ylabel=True,
)

add_topographic_contours(
    axes[0]
)


# ============================================================
# PANEL (b)
# ============================================================

im1 = axes[1].pcolormesh(
    LON,
    LAT,
    masked(
        rmse_unet
    ),

    shading="auto",

    cmap="viridis",

    vmin=0,
    vmax=RMSE_VMAX,
)


axes[1].set_title(
    "(b) U-Net CMIP-CMIP"
)

finish_map_axis(
    axes[1],
    show_ylabel=False,
)

add_topographic_contours(
    axes[1]
)


# ============================================================
# PANEL (c)
# ============================================================

norm = TwoSlopeNorm(
    vmin=-DELTA_LIM,
    vcenter=0,
    vmax=DELTA_LIM,
)


im2 = axes[2].pcolormesh(
    LON,
    LAT,
    masked(
        delta_unet
    ),

    shading="auto",

    cmap="RdBu_r",

    norm=norm,
)


axes[2].set_title(
    r"(c) $\Delta$RMSE = U-Net $-$ TopoFusion-Res"
)

finish_map_axis(
    axes[2],
    show_ylabel=False,
)

add_topographic_contours(
    axes[2]
)


band_label_positions = (
    add_altitudinal_band_labels(
        axes[2]
    )
)


# ============================================================
# COLORBARS
#
# Preserve original layout.
# ============================================================

cb1 = fig.colorbar(
    im0,

    ax=axes[:2],

    fraction=0.030,
    pad=0.035,
)

cb1.set_label(
    r"RMSE ($^\circ$C)"
)


cb2 = fig.colorbar(
    im2,

    ax=axes[2],

    fraction=0.050,
    pad=0.025,
)

cb2.set_label(
    r"$\Delta$RMSE ($^\circ$C)"
)


fig.subplots_adjust(
    left=0.065,
    right=0.955,

    bottom=0.12,
    top=0.90,

    # Slightly larger spacing prevents the shared RMSE
    # colorbar label from competing visually with panel (c).
    wspace=0.32,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PNG,

    dpi=300,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(
    PDF,

    bbox_inches="tight",

    facecolor="white",
)


plt.close(
    fig
)


# ============================================================
# SPATIAL STATISTICS
# ============================================================

delta_valid = delta_unet[
    valid
]

delta_valid = delta_valid[
    np.isfinite(
        delta_valid
    )
]


fraction_positive = float(
    np.mean(
        delta_valid
        > 0
    )
)

fraction_negative = float(
    np.mean(
        delta_valid
        < 0
    )
)

mean_delta = float(
    np.mean(
        delta_valid
    )
)

median_delta = float(
    np.median(
        delta_valid
    )
)

p10_delta = float(
    np.percentile(
        delta_valid,
        10,
    )
)

p90_delta = float(
    np.percentile(
        delta_valid,
        90,
    )
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "figure":
        "Figure 2",

    "comparison":
        "TopoFusion-Res versus U-Net CMIP-CMIP",

    "support": {
        "n_common_cells":
            int(
                valid.sum()
            ),
    },

    "sign_convention":
        (
            "Delta RMSE = U-Net RMSE minus "
            "TopoFusion-Res RMSE. Positive values "
            "indicate lower RMSE for TopoFusion-Res."
        ),

    "spatial_delta_rmse_C": {
        "mean":
            mean_delta,

        "median":
            median_delta,

        "p10":
            p10_delta,

        "p90":
            p90_delta,

        "fraction_positive":
            fraction_positive,

        "fraction_negative":
            fraction_negative,
    },

    "topography_overlay": {
        "panels": [
            "a",
            "b",
            "c",
        ],

        "physical_field":
            "fine-scale orography used by Model E",

        "normalized_source":
            str(
                STATIC_FILE
            ),

        "feature_names_source":
            str(
                STATIC_NAMES_FILE
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
                "Topographic contours and altitude-band labels "
                "are shown only as geographic reference. "
                "They do not modify the RMSE diagnostics and "
                "do not by themselves establish a causal "
                "relationship between topography and the "
                "TopoFusion-Res versus U-Net difference."
            ),
    },

    "visualization": {
        "display_percentile":
            98.0,

        "rmse_vmax_C":
            RMSE_VMAX,

        "delta_symmetric_limit_C":
            DELTA_LIM,

        "note":
            (
                "The 98th percentile is used only for "
                "visual color limits. All reported statistics "
                "use the complete finite common support."
            ),
    },

    "outputs": {
        "png":
            str(
                PNG
            ),

        "pdf":
            str(
                PDF
            ),
    },
}


with open(
    META,
    "w",
) as f:

    json.dump(
        metadata,
        f,

        indent=2,
    )


# ============================================================
# FINAL AUDIT
# ============================================================

print()
print("=" * 90)
print("FIGURE 2 — FINAL TOPOGRAPHIC CONTEXT VERSION")
print("=" * 90)

print(
    "RMSE source :",
    RMSE_FILE,
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
    RMSE_VMAX,
)

print(
    "Delta p98   :",
    DELTA_LIM,
)

print()
print(
    "Mean ΔRMSE  :",
    mean_delta,
)

print(
    "Median      :",
    median_delta,
)

print(
    "Fraction ΔRMSE > 0:",
    fraction_positive,
)

print()
print("Saved:")
print(PNG)
print(PDF)
print(META)

print()
print("DONE")
