from pathlib import Path
import json
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[2]

# EXP:
# EXP/ROOT now point to the repository root.
ROOT = EXP

CUBE = (
    EXP
    / "diagnostics"
    / "final_e2_vs_unet_advanced_diagnostics"
    / "prediction_cubes"
)

FINAL = EXP / "analysis"

OUT_FIG = (
    FINAL
    / "figures"
    / "final_paper"
)

OUT_DIA = (
    FINAL
    / "diagnostics"
)

OUT_FIG.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIA.mkdir(
    parents=True,
    exist_ok=True,
)


# Exact coordinate source already used by the final validation
# pipeline (02_complete_mirror_metrics.py / 03_daily_persistence.py).
COORD_FILE = (
    ROOT
    / "inputs"
    / "unet"
    / "tas_cmip_cmip_output.nc"
)


# ============================================================
# METADATA
# ============================================================

with open(
    CUBE / "cube_metadata.json"
) as f:
    meta = json.load(f)

shape = tuple(
    meta["shape"]
)

N, H, W = shape


print("=" * 100)
print("FINAL SEEDED SPATIAL FIGURES — PUBLICATION VERSION")
print("=" * 100)

print("Experiment :", EXP)
print("Cube dir   :", CUBE)
print("Shape      :", shape)
print("Coords     :", COORD_FILE)


if shape != (
    31390,
    64,
    64,
):
    raise RuntimeError(
        f"Unexpected cube shape: {shape}"
    )


if not COORD_FILE.exists():
    raise FileNotFoundError(
        COORD_FILE
    )


# ============================================================
# INPUT FILES
# ============================================================

paths = {
    "RCM":
        CUBE
        / "rcm.dat",

    "U-Net":
        CUBE
        / "unet.dat",

    "TopoFusion-Res":
        CUBE
        / "e2_full_seeded.dat",

    "TopoFusion-Base":
        CUBE
        / "phase1_full_seeded.dat",

    "TopoFusion-Res without fine topography":
        CUBE
        / "e2_no_fine_topo_seeded.dat",

    "Mask":
        CUBE
        / "mask.dat",
}


print("\nInputs")

for name, path in paths.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name}: {path}"
        )

    print(
        f"{name:42s}: "
        f"{path.name}"
    )


# ============================================================
# LATITUDE / LONGITUDE
# ============================================================

ds = xr.open_dataset(
    COORD_FILE
)

lat = np.asarray(
    ds["lat"].values
)

lon = np.asarray(
    ds["lon"].values
)

ds.close()


print("\nCoordinates")
print("lat shape :", lat.shape)
print("lon shape :", lon.shape)


if (
    lat.shape != (H, W)
    or
    lon.shape != (H, W)
):
    raise RuntimeError(
        "Expected 2-D coordinates with shape "
        f"{(H, W)}, found "
        f"lat={lat.shape}, lon={lon.shape}"
    )


if not np.all(
    np.isfinite(
        lat[
            np.isfinite(lat)
        ]
    )
):
    raise RuntimeError(
        "Non-finite latitude coordinate problem."
    )


if not np.all(
    np.isfinite(
        lon[
            np.isfinite(lon)
        ]
    )
):
    raise RuntimeError(
        "Non-finite longitude coordinate problem."
    )


# ============================================================
# MEMMAPS
# ============================================================

rcm = np.memmap(
    paths["RCM"],
    dtype="float32",
    mode="r",
    shape=shape,
)

unet = np.memmap(
    paths["U-Net"],
    dtype="float32",
    mode="r",
    shape=shape,
)

res_full = np.memmap(
    paths["TopoFusion-Res"],
    dtype="float32",
    mode="r",
    shape=shape,
)

base_full = np.memmap(
    paths["TopoFusion-Base"],
    dtype="float32",
    mode="r",
    shape=shape,
)

res_notopo = np.memmap(
    paths[
        "TopoFusion-Res without fine topography"
    ],
    dtype="float32",
    mode="r",
    shape=shape,
)

mask = np.memmap(
    paths["Mask"],
    dtype="bool",
    mode="r",
    shape=shape,
)


# ============================================================
# CELL-WISE RMSE
# ============================================================

chunk = 128

count = np.zeros(
    (H, W),
    dtype=np.float64,
)

sse_unet = np.zeros(
    (H, W),
    dtype=np.float64,
)

sse_res = np.zeros(
    (H, W),
    dtype=np.float64,
)

sse_base = np.zeros(
    (H, W),
    dtype=np.float64,
)

sse_notopo = np.zeros(
    (H, W),
    dtype=np.float64,
)


for s in range(
    0,
    N,
    chunk,
):

    e = min(
        s + chunk,
        N,
    )

    # copy=True is intentional because the memmap mask
    # itself is read-only.
    m = np.array(
        mask[s:e],
        dtype=bool,
        copy=True,
    )

    y = np.asarray(
        rcm[s:e],
        dtype=np.float64,
    )

    p_unet = np.asarray(
        unet[s:e],
        dtype=np.float64,
    )

    p_res = np.asarray(
        res_full[s:e],
        dtype=np.float64,
    )

    p_base = np.asarray(
        base_full[s:e],
        dtype=np.float64,
    )

    p_notopo = np.asarray(
        res_notopo[s:e],
        dtype=np.float64,
    )


    # Exact common finite support
    m &= np.isfinite(y)
    m &= np.isfinite(p_unet)
    m &= np.isfinite(p_res)
    m &= np.isfinite(p_base)
    m &= np.isfinite(p_notopo)


    w = m.astype(
        np.float64
    )

    count += w.sum(
        axis=0
    )


    sse_unet += np.where(
        m,
        (p_unet - y) ** 2,
        0.0,
    ).sum(
        axis=0
    )


    sse_res += np.where(
        m,
        (p_res - y) ** 2,
        0.0,
    ).sum(
        axis=0
    )


    sse_base += np.where(
        m,
        (p_base - y) ** 2,
        0.0,
    ).sum(
        axis=0
    )


    sse_notopo += np.where(
        m,
        (p_notopo - y) ** 2,
        0.0,
    ).sum(
        axis=0
    )


    if (
        e == N
        or
        (s // chunk + 1) % 25 == 0
    ):
        print(
            f"Processed {e:5d}/{N} days"
        )


# ============================================================
# COMMON SUPPORT
# ============================================================

valid = (
    count > 0
)

n_valid = int(
    valid.sum()
)


print(
    "\nCommon valid cells:",
    n_valid,
)


if n_valid != 1852:

    raise RuntimeError(
        "Expected 1852 common cells, "
        f"found {n_valid}"
    )


# ============================================================
# RMSE MAPS
# ============================================================

def rmse_map(
    sse,
):

    out = np.full(
        (H, W),
        np.nan,
        dtype=np.float64,
    )

    out[valid] = np.sqrt(
        sse[valid]
        /
        count[valid]
    )

    return out


rmse_unet = rmse_map(
    sse_unet
)

rmse_res = rmse_map(
    sse_res
)

rmse_base = rmse_map(
    sse_base
)

rmse_notopo = rmse_map(
    sse_notopo
)


# ============================================================
# DIFFERENCE MAPS
#
# Positive always means improvement from the component/model
# highlighted by the comparison:
#
# Fig 2:
#     U-Net - TopoFusion-Res
#
# Fig 3:
#     TopoFusion-Base - TopoFusion-Res
#
# Fig 4:
#     No-fine-topography - Full
# ============================================================

delta_unet_minus_res = (
    rmse_unet
    -
    rmse_res
)

delta_base_minus_res = (
    rmse_base
    -
    rmse_res
)

delta_notopo_minus_full = (
    rmse_notopo
    -
    rmse_res
)


# ============================================================
# SAVE RAW MAP ARRAYS
#
# IMPORTANT:
# These arrays are never clipped.
# Plot clipping is visual only.
# ============================================================

npz_path = (
    OUT_DIA
    / "38_final_seeded_spatial_rmse_maps.npz"
)

np.savez_compressed(
    npz_path,

    valid=valid,
    count=count,

    rmse_unet=rmse_unet,
    rmse_res_full=rmse_res,
    rmse_base_full=rmse_base,
    rmse_res_no_fine_topo=rmse_notopo,

    unet_minus_res_full=
        delta_unet_minus_res,

    base_minus_res=
        delta_base_minus_res,

    no_fine_topo_minus_full=
        delta_notopo_minus_full,

    lat=lat,
    lon=lon,
)


print(
    "Saved:",
    npz_path,
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def delta_summary(
    arr,
):

    v = np.asarray(
        arr
    )[valid]

    v = v[
        np.isfinite(v)
    ]

    return {
        "mean_delta_rmse":
            float(
                np.mean(v)
            ),

        "median_delta_rmse":
            float(
                np.median(v)
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

        "fraction_cells_positive":
            float(
                np.mean(
                    v > 0
                )
            ),

        "fraction_cells_negative":
            float(
                np.mean(
                    v < 0
                )
            ),

        "n_cells":
            int(
                v.size
            ),
    }


# ============================================================
# PLOTTING SCALE HELPERS
#
# Preserve the ORIGINAL p98 logic.
#
# This is only a display transformation:
# - all statistics above use the complete arrays;
# - raw arrays remain unchanged;
# - extended colorbar ends explicitly indicate saturation.
# ============================================================

PLOT_PERCENTILE = 98.0


def common_rmse_vmax(
    a,
    b,
):

    values = np.concatenate([
        np.asarray(a)[valid],
        np.asarray(b)[valid],
    ])

    values = values[
        np.isfinite(values)
    ]

    return float(
        np.percentile(
            values,
            PLOT_PERCENTILE,
        )
    )


def delta_lim(
    d,
):

    values = np.abs(
        np.asarray(d)[valid]
    )

    values = values[
        np.isfinite(values)
    ]

    lim = float(
        np.percentile(
            values,
            PLOT_PERCENTILE,
        )
    )

    return max(
        lim,
        1e-6,
    )


def fraction_rmse_saturated(
    a,
    b,
    vmax,
):

    values = np.concatenate([
        np.asarray(a)[valid],
        np.asarray(b)[valid],
    ])

    values = values[
        np.isfinite(values)
    ]

    return float(
        np.mean(
            values > vmax
        )
    )


def fraction_delta_saturated(
    d,
    lim,
):

    values = np.abs(
        np.asarray(d)[valid]
    )

    values = values[
        np.isfinite(values)
    ]

    return float(
        np.mean(
            values > lim
        )
    )


# ============================================================
# PRECOMPUTE PLOT LIMITS
# ============================================================

plot_scales = {
    "figure_2": {
        "rmse_vmax":
            common_rmse_vmax(
                rmse_res,
                rmse_unet,
            ),

        "delta_lim":
            delta_lim(
                delta_unet_minus_res
            ),
    },

    "figure_3": {
        "rmse_vmax":
            common_rmse_vmax(
                rmse_base,
                rmse_res,
            ),

        "delta_lim":
            delta_lim(
                delta_base_minus_res
            ),
    },

    "figure_4": {
        "rmse_vmax":
            common_rmse_vmax(
                rmse_notopo,
                rmse_res,
            ),

        "delta_lim":
            delta_lim(
                delta_notopo_minus_full
            ),
    },
}


plot_scales["figure_2"].update({
    "rmse_fraction_above_vmax":
        fraction_rmse_saturated(
            rmse_res,
            rmse_unet,
            plot_scales[
                "figure_2"
            ]["rmse_vmax"],
        ),

    "delta_fraction_beyond_limit":
        fraction_delta_saturated(
            delta_unet_minus_res,
            plot_scales[
                "figure_2"
            ]["delta_lim"],
        ),
})


plot_scales["figure_3"].update({
    "rmse_fraction_above_vmax":
        fraction_rmse_saturated(
            rmse_base,
            rmse_res,
            plot_scales[
                "figure_3"
            ]["rmse_vmax"],
        ),

    "delta_fraction_beyond_limit":
        fraction_delta_saturated(
            delta_base_minus_res,
            plot_scales[
                "figure_3"
            ]["delta_lim"],
        ),
})


plot_scales["figure_4"].update({
    "rmse_fraction_above_vmax":
        fraction_rmse_saturated(
            rmse_notopo,
            rmse_res,
            plot_scales[
                "figure_4"
            ]["rmse_vmax"],
        ),

    "delta_fraction_beyond_limit":
        fraction_delta_saturated(
            delta_notopo_minus_full,
            plot_scales[
                "figure_4"
            ]["delta_lim"],
        ),
})


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "support": {
        "n_days":
            int(N),

        "n_common_cells":
            int(n_valid),
    },

    "coordinate_source":
        str(
            COORD_FILE
        ),

    "figure_2_topofusion_vs_unet":
        delta_summary(
            delta_unet_minus_res
        ),

    "figure_3_residual_refinement":
        delta_summary(
            delta_base_minus_res
        ),

    "figure_4_fine_topography":
        delta_summary(
            delta_notopo_minus_full
        ),

    "sign_convention": {

        "figure_2":
            (
                "U-Net RMSE minus TopoFusion-Res RMSE; "
                "positive means TopoFusion-Res has "
                "lower RMSE."
            ),

        "figure_3":
            (
                "TopoFusion-Base RMSE minus "
                "TopoFusion-Res RMSE; positive means "
                "residual refinement lowers RMSE."
            ),

        "figure_4":
            (
                "TopoFusion-Res without fine topography "
                "RMSE minus full TopoFusion-Res RMSE; "
                "positive means fine topography lowers "
                "RMSE."
            ),
    },

    "plotting": {
        "display_percentile":
            PLOT_PERCENTILE,

        "note":
            (
                "Plot color limits use the 98th percentile "
                "for visualization only. Raw maps and all "
                "reported statistics use all finite values "
                "without clipping."
            ),

        "scales":
            plot_scales,
    },
}


summary_path = (
    OUT_DIA
    / "38_final_seeded_spatial_rmse_summary.json"
)


with open(
    summary_path,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "Saved:",
    summary_path,
)


# ============================================================
# PLOTTING STYLE
# ============================================================

plt.rcParams.update({

    "font.size":
        10,

    "axes.titlesize":
        11,

    "axes.labelsize":
        10,

    "xtick.labelsize":
        9,

    "ytick.labelsize":
        9,

    "axes.linewidth":
        0.8,

    "figure.dpi":
        120,

    "savefig.dpi":
        300,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ============================================================
# PLOT HELPERS
# ============================================================

def masked(
    a,
):

    z = np.asarray(
        a,
        dtype=np.float64,
    ).copy()

    z[
        ~valid
    ] = np.nan

    return z


def geo_aspect():

    mean_lat = float(
        np.nanmean(
            lat[valid]
        )
    )

    return float(
        1.0
        /
        np.cos(
            np.deg2rad(
                mean_lat
            )
        )
    )


MAP_ASPECT = geo_aspect()


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
        length=3.5,
        width=0.8,
    )

    ax.grid(
        False
    )


# ============================================================
# THREE-PANEL PUBLICATION FIGURE
#
# Layout:
#
# [ panel a ][ panel b ][ RMSE CB ][ gap ][ panel c ][ Δ CB ]
#
# This prevents the shared colorbar from colliding with
# panel (c).
# ============================================================

def save_three_panel(
    first,
    second,
    delta,

    title_first,
    title_second,
    title_delta,

    basename,
):

    vmax = common_rmse_vmax(
        first,
        second,
    )

    lim = delta_lim(
        delta
    )


    print(
        "\n" + "-" * 100
    )

    print(
        basename
    )

    print(
        f"RMSE display p{PLOT_PERCENTILE:g}: "
        f"0.000000 .. {vmax:.6f}"
    )

    print(
        f"Delta display p{PLOT_PERCENTILE:g}: "
        f"{-lim:.6f} .. {lim:.6f}"
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
            0.075,
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

    gap_ax = fig.add_subplot(
        gs[0, 3]
    )

    gap_ax.axis(
        "off"
    )

    ax2 = fig.add_subplot(
        gs[0, 4]
    )

    cax_delta = fig.add_subplot(
        gs[0, 5]
    )


    # --------------------------------------------------------
    # Colormaps
    # --------------------------------------------------------

    cmap_rmse = plt.get_cmap(
        "viridis"
    ).copy()

    cmap_rmse.set_bad(
        color="white"
    )


    cmap_delta = plt.get_cmap(
        "RdBu_r"
    ).copy()

    cmap_delta.set_bad(
        color="white"
    )


    # --------------------------------------------------------
    # Panel (a)
    # --------------------------------------------------------

    im0 = ax0.pcolormesh(

        lon,
        lat,
        masked(first),

        shading="auto",
        rasterized=True,

        cmap=cmap_rmse,

        vmin=0.0,
        vmax=vmax,
    )


    ax0.set_title(
        title_first
    )

    format_geo_axis(
        ax0,
        show_ylabel=True,
    )


    # --------------------------------------------------------
    # Panel (b)
    # --------------------------------------------------------

    im1 = ax1.pcolormesh(

        lon,
        lat,
        masked(second),

        shading="auto",
        rasterized=True,

        cmap=cmap_rmse,

        vmin=0.0,
        vmax=vmax,
    )


    ax1.set_title(
        title_second
    )

    format_geo_axis(
        ax1,
        show_ylabel=False,
    )


    # --------------------------------------------------------
    # Shared RMSE colorbar for (a) and (b)
    # --------------------------------------------------------

    cb_rmse = fig.colorbar(

        im1,

        cax=cax_rmse,

        extend="max",
        extendfrac=0.035,
    )


    cb_rmse.set_label(
        r"RMSE ($^\circ$C)"
    )


    # --------------------------------------------------------
    # Panel (c)
    # --------------------------------------------------------

    norm = TwoSlopeNorm(

        vmin=-lim,

        vcenter=0.0,

        vmax=lim,
    )


    im2 = ax2.pcolormesh(

        lon,
        lat,
        masked(delta),

        shading="auto",
        rasterized=True,

        cmap=cmap_delta,

        norm=norm,
    )


    ax2.set_title(
        title_delta
    )


    format_geo_axis(
        ax2,
        show_ylabel=False,
    )


    # --------------------------------------------------------
    # Delta colorbar
    # --------------------------------------------------------

    cb_delta = fig.colorbar(

        im2,

        cax=cax_delta,

        extend="both",
        extendfrac=0.035,
    )


    cb_delta.set_label(
        r"$\Delta$RMSE ($^\circ$C)"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    png = (
        OUT_FIG
        /
        f"{basename}.png"
    )

    pdf = (
        OUT_FIG
        /
        f"{basename}.pdf"
    )


    fig.savefig(

        png,

        dpi=300,

        bbox_inches="tight",

        facecolor="white",
    )


    fig.savefig(

        pdf,

        bbox_inches="tight",

        facecolor="white",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        png,
    )

    print(
        "Saved:",
        pdf,
    )


# ============================================================
# FIGURE 2
# Spatial RMSE:
# TopoFusion-Res vs U-Net
# ============================================================

save_three_panel(

    first=
        rmse_res,

    second=
        rmse_unet,

    delta=
        delta_unet_minus_res,


    title_first=
        "(a) TopoFusion-Res",

    title_second=
        "(b) U-Net CMIP-CMIP",

    title_delta=
        (
            r"(c) $\Delta$RMSE = "
            r"U-Net $-$ TopoFusion-Res"
        ),


    basename=
        "02_spatial_rmse_topofusion_vs_unet_publication",
)


# ============================================================
# FIGURE 3
# Controlled residual-refinement ablation
# ============================================================

save_three_panel(

    first=
        rmse_base,

    second=
        rmse_res,

    delta=
        delta_base_minus_res,


    title_first=
        "(a) TopoFusion-Base",

    title_second=
        "(b) TopoFusion-Res",

    title_delta=
        (
            r"(c) $\Delta$RMSE = "
            r"Base $-$ Residual"
        ),


    basename=
        "03_residual_refinement_spatial_rmse_publication",
)


# ============================================================
# FIGURE 4
# Controlled fine-topography ablation
# ============================================================

save_three_panel(

    first=
        rmse_notopo,

    second=
        rmse_res,

    delta=
        delta_notopo_minus_full,


    title_first=
        (
            "(a) TopoFusion-Res without "
            "fine topography"
        ),

    title_second=
        "(b) TopoFusion-Res",

    title_delta=
        (
            r"(c) $\Delta$RMSE = "
            r"No fine topography $-$ Full"
        ),


    basename=
        "04_fine_topography_spatial_rmse_publication",
)


# ============================================================
# FINISH
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "DONE"
)

print(
    "=" * 100
)


print(
    "\nOutput figures:"
)

print(
    OUT_FIG
)


print(
    "\nSpatial summary:"
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)
