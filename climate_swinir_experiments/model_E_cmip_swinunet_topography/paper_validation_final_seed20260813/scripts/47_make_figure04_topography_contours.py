from pathlib import Path
import json
import warnings

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

ROOT = EXP.parents[1]
# EXP  = .../Normalizacion/climate_swinir_experiments/model_E_...
# ROOT = .../Normalizacion

FINAL = EXP / "paper_validation_final_seed20260813"
DIA = FINAL / "diagnostics"

RMSE_FILE = DIA / "38_final_seeded_spatial_rmse_maps.npz"

COORD_FILE = (
    ROOT
    / "data_tomas"
    / "unet_outputs"
    / "tas_cmip_cmip_output.nc"
)

OUT = FINAL / "final_publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "Figure04_fine_topography_with_contours.png"
PDF = OUT / "Figure04_fine_topography_with_contours.pdf"
META = OUT / "Figure04_fine_topography_with_contours_metadata.json"


# ============================================================
# CHECK INPUTS
# ============================================================

if not RMSE_FILE.exists():
    raise FileNotFoundError(RMSE_FILE)

if not COORD_FILE.exists():
    raise FileNotFoundError(COORD_FILE)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,

    "axes.titlesize": 12,
    "axes.labelsize": 11,

    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,

    "axes.linewidth": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 300,
})


# ============================================================
# LOAD FINAL SEEDED RMSE MAPS
# ============================================================

with np.load(RMSE_FILE) as z:

    print("RMSE keys:", z.files)

    required_keys = [
        "valid",
        "rmse_res_no_fine_topo",
        "rmse_res_full",
        "no_fine_topo_minus_full",
    ]

    missing = [
        k
        for k in required_keys
        if k not in z.files
    ]

    if missing:
        raise KeyError(
            "Missing keys in RMSE NPZ:\n"
            + "\n".join(missing)
        )

    valid = np.asarray(
        z["valid"]
    ).astype(bool)

    rmse_notopo = np.asarray(
        z["rmse_res_no_fine_topo"],
        dtype=np.float64,
    )

    rmse_full = np.asarray(
        z["rmse_res_full"],
        dtype=np.float64,
    )

    delta = np.asarray(
        z["no_fine_topo_minus_full"],
        dtype=np.float64,
    )

    lat = np.asarray(
        z["lat"],
        dtype=np.float64,
    )

    lon = np.asarray(
        z["lon"],
        dtype=np.float64,
    )


if valid.shape != (64, 64):
    raise RuntimeError(
        f"Expected valid shape (64,64), got {valid.shape}"
    )


# ============================================================
# LOAD LAT/LON — FROM FINAL SEEDED RMSE NPZ
#
# The final RMSE diagnostic already stores the exact 64x64
# latitude/longitude arrays used by the publication figures.
# This avoids reopening the U-Net NetCDF file and removes an
# unnecessary NetCDF backend dependency.
# ============================================================

if lat.shape != (64, 64):
    raise RuntimeError(
        f"Expected lat shape (64,64), got {lat.shape}"
    )

if lon.shape != (64, 64):
    raise RuntimeError(
        f"Expected lon shape (64,64), got {lon.shape}"
    )

LAT = lat
LON = lon

print("Coordinate source:", RMSE_FILE)
print("lat shape:", LAT.shape)
print("lon shape:", LON.shape)


# ============================================================
# MASKED PLOT ARRAYS
# ============================================================

def masked(a):
    x = np.asarray(
        a,
        dtype=np.float64,
    ).copy()

    x[~valid] = np.nan

    return x


rmse_notopo_plot = masked(
    rmse_notopo
)

rmse_full_plot = masked(
    rmse_full
)

delta_plot = masked(
    delta
)


# ============================================================
# DISPLAY SCALES
#
# Exactly the same principle used in the approved Figure 4:
# spatial p98 for visualization only.
# Raw statistics remain unchanged.
# ============================================================

DISPLAY_PERCENTILE = 98.0


rmse_values = np.concatenate([
    rmse_notopo[valid],
    rmse_full[valid],
])

rmse_values = rmse_values[
    np.isfinite(rmse_values)
]

RMSE_VMAX = float(
    np.percentile(
        rmse_values,
        DISPLAY_PERCENTILE,
    )
)


delta_values = np.abs(
    delta[valid]
)

delta_values = delta_values[
    np.isfinite(delta_values)
]

DELTA_LIM = float(
    np.percentile(
        delta_values,
        DISPLAY_PERCENTILE,
    )
)


if not np.isclose(
    RMSE_VMAX,
    2.0326396167343166,
    rtol=0,
    atol=1e-6,
):
    warnings.warn(
        "RMSE p98 differs from previously recorded "
        f"Figure 4 value: {RMSE_VMAX}"
    )


if not np.isclose(
    DELTA_LIM,
    0.14231422660892754,
    rtol=0,
    atol=1e-6,
):
    warnings.warn(
        "Delta p98 differs from previously recorded "
        f"Figure 4 value: {DELTA_LIM}"
    )


# ============================================================
# FIND PHYSICAL 64x64 ELEVATION FIELD
#
# IMPORTANT:
# We accept only physically plausible elevation in metres.
# Normalized/z-scored arrays are rejected.
# ============================================================

NAME_HINTS = (
    "orog",
    "orography",
    "elev",
    "elevation",
    "topo",
    "topography",
    "height",
)


def plausible_physical_elevation(a):
    """
    Strict check intended to reject normalized static fields.

    Regional elevation should:
      - be 64x64;
      - contain many finite values;
      - have a range > 1000 m;
      - reach > 1500 m;
      - remain within terrestrial elevation bounds.
    """

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    if a.shape != (64, 64):
        return False

    x = a[np.isfinite(a)]

    if x.size < 1000:
        return False

    amin = float(
        np.nanmin(x)
    )

    amax = float(
        np.nanmax(x)
    )

    arange = amax - amin

    if amax < 1500:
        return False

    if arange < 1000:
        return False

    if amin < -1000:
        return False

    if amax > 10000:
        return False

    return True


def candidate_score(path, key=None, units=None):
    """
    Rank likely physical-orography candidates.
    Higher is better.
    """

    text = (
        str(path).lower()
        + " "
        + str(key or "").lower()
    )

    score = 0

    if "elevation" in text:
        score += 10

    if "orog" in text:
        score += 10

    if "topo" in text:
        score += 8

    if "height" in text:
        score += 4

    if units:
        u = str(units).lower()

        if u in {
            "m",
            "meter",
            "meters",
            "metre",
            "metres",
        }:
            score += 12

    if "norm" in text:
        score -= 20

    if "standard" in text:
        score -= 20

    if "scaled" in text:
        score -= 20

    return score


def register_candidate(
    candidates,
    array,
    source,
    key=None,
    units=None,
):
    """
    Add candidate only if it is a plausible physical elevation
    field on the 64x64 target grid.
    """

    try:
        a = np.asarray(
            array,
            dtype=np.float64,
        )

        # Direct 64x64 field
        if a.shape == (64, 64):

            if plausible_physical_elevation(a):

                candidates.append({
                    "score":
                        candidate_score(
                            source,
                            key,
                            units,
                        ),

                    "source":
                        str(source),

                    "key":
                        key,

                    "units":
                        units,

                    "array":
                        a.copy(),
                })

            return


        # Channel-first 3-D arrays:
        # elevation is known to be channel 0 in S^f.
        if (
            a.ndim == 3
            and a.shape[1:] == (64, 64)
        ):

            channel0 = a[0]

            if plausible_physical_elevation(
                channel0
            ):

                candidates.append({
                    "score":
                        candidate_score(
                            source,
                            key,
                            units,
                        ),

                    "source":
                        str(source),

                    "key":
                        (
                            f"{key}[0]"
                            if key
                            else "channel_0"
                        ),

                    "units":
                        units,

                    "array":
                        channel0.copy(),
                })

            return


        # Channel-last 3-D arrays
        if (
            a.ndim == 3
            and a.shape[:2] == (64, 64)
        ):

            channel0 = a[..., 0]

            if plausible_physical_elevation(
                channel0
            ):

                candidates.append({
                    "score":
                        candidate_score(
                            source,
                            key,
                            units,
                        ),

                    "source":
                        str(source),

                    "key":
                        (
                            f"{key}[...,0]"
                            if key
                            else "channel_0"
                        ),

                    "units":
                        units,

                    "array":
                        channel0.copy(),
                })

    except Exception:
        pass


def find_elevation():
    """
    Search likely model/data locations.

    It does NOT accept normalized fields.

    If several equivalent physical elevation fields are found,
    highest-scoring candidate is selected and all candidates
    are printed for audit.
    """

    candidates = []

    search_roots = [
        EXP / "data",
        ROOT / "data_tomas",
        ROOT / "Models",
    ]

    search_roots = [
        p
        for p in search_roots
        if p.exists()
    ]


    # --------------------------------------------------------
    # NPY / NPZ
    # --------------------------------------------------------

    for root in search_roots:

        for p in root.rglob("*"):

            if not p.is_file():
                continue

            name = p.name.lower()

            # Only inspect likely static/elevation files.
            if not any(
                hint in name
                for hint in (
                    *NAME_HINTS,
                    "static",
                )
            ):
                continue


            if p.suffix.lower() == ".npy":

                try:
                    a = np.load(
                        p,
                        mmap_mode="r",
                    )

                    register_candidate(
                        candidates,
                        a,
                        p,
                    )

                except Exception:
                    pass


            elif p.suffix.lower() == ".npz":

                try:

                    with np.load(p) as z:

                        for key in z.files:

                            key_l = key.lower()

                            if not any(
                                hint in key_l
                                for hint in (
                                    *NAME_HINTS,
                                    "static",
                                )
                            ):
                                continue

                            register_candidate(
                                candidates,
                                z[key],
                                p,
                                key=key,
                            )

                except Exception:
                    pass


    # --------------------------------------------------------
    # NETCDF
    # --------------------------------------------------------

    for root in search_roots:

        for pattern in (
            "*.nc",
            "*.nc4",
            "*.netcdf",
        ):

            for p in root.rglob(
                pattern
            ):

                try:

                    ds = xr.open_dataset(
                        p,
                        decode_times=False,
                    )

                except Exception:
                    continue

                try:

                    for name, da in ds.data_vars.items():

                        name_l = name.lower()

                        if not any(
                            hint in name_l
                            for hint in NAME_HINTS
                        ):
                            continue

                        units = da.attrs.get(
                            "units"
                        )

                        a = np.asarray(
                            da.values
                        )

                        # Remove singleton leading dimensions.
                        while (
                            a.ndim > 2
                            and a.shape[0] == 1
                        ):
                            a = a[0]

                        register_candidate(
                            candidates,
                            a,
                            p,
                            key=name,
                            units=units,
                        )

                finally:
                    ds.close()


    if not candidates:

        raise RuntimeError(
            "\n"
            "Could not find a physical 64x64 elevation field.\n"
            "No normalized field was used.\n\n"
            "Search roots were:\n  "
            + "\n  ".join(
                str(p)
                for p in search_roots
            )
            + "\n\n"
            "Before drawing contours we must identify "
            "the exact physical elevation source."
        )


    # Highest-confidence first
    candidates.sort(
        key=lambda d: d["score"],
        reverse=True,
    )


    print()
    print("=" * 90)
    print("PHYSICAL ELEVATION CANDIDATES")
    print("=" * 90)

    for i, c in enumerate(
        candidates[:20],
        start=1,
    ):

        arr = c["array"]

        print(
            f"[{i:02d}] score={c['score']:3d}  "
            f"min={np.nanmin(arr):8.2f}  "
            f"max={np.nanmax(arr):8.2f}"
        )

        print(
            "     source:",
            c["source"],
        )

        if c["key"] is not None:
            print(
                "     key   :",
                c["key"],
            )

        if c["units"] is not None:
            print(
                "     units :",
                c["units"],
            )


    # --------------------------------------------------------
    # Guard against ambiguity:
    #
    # If top two candidates are physically different and tied,
    # stop instead of silently choosing one.
    # --------------------------------------------------------

    if len(candidates) >= 2:

        c0 = candidates[0]
        c1 = candidates[1]

        same_score = (
            c0["score"]
            == c1["score"]
        )

        same_field = np.allclose(
            c0["array"],
            c1["array"],
            equal_nan=True,
            atol=1e-4,
            rtol=1e-6,
        )

        if same_score and not same_field:

            raise RuntimeError(
                "\nAmbiguous elevation source: "
                "the two highest-scoring candidates differ.\n"
                "I will not choose one automatically.\n"
                "Use the candidate list above to identify "
                "the field used by the model."
            )


    selected = candidates[0]

    print()
    print("SELECTED PHYSICAL ELEVATION:")
    print(" source:", selected["source"])
    print(" key   :", selected["key"])
    print(
        " range :",
        float(
            np.nanmin(
                selected["array"]
            )
        ),
        "..",
        float(
            np.nanmax(
                selected["array"]
            )
        ),
        "m",
    )

    return selected


# ============================================================
# RECONSTRUCT THE EXACT PHYSICAL OROGRAPHY USED BY MODEL E
#
# static_features_64x64.npy channel 0 is:
#     normalized fine-scale orography
#
# During dataset preparation:
#     orog_norm = (orog - mean) / std
#
# Therefore:
#     orog [m] = orog_norm * std + mean
#
# This avoids reopening the NetCDF source and guarantees that
# the contour field corresponds exactly to the field supplied
# to TopoFusion-Res.
# ============================================================

DATASET_DIR = (
    EXP
    / "data"
    / "torch_dataset_model_E_physical_v1"
)

STATIC_FILE = (
    DATASET_DIR
    / "static_features_64x64.npy"
)

STATS_FILE = (
    DATASET_DIR
    / "normalization_stats.json"
)

CONFIG_FILE = (
    EXP
    / "configs"
    / "config_model_E.json"
)


for required_file in [
    STATIC_FILE,
    STATS_FILE,
    CONFIG_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            required_file
        )


# ------------------------------------------------------------
# Load normalized fine-static tensor
# ------------------------------------------------------------

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


print()
print("=" * 90)
print("RECONSTRUCTING PHYSICAL OROGRAPHY")
print("=" * 90)

print(
    "Normalized source :",
    STATIC_FILE,
)

print(
    "Normalized range  :",
    float(
        np.nanmin(
            orog_norm
        )
    ),
    "..",
    float(
        np.nanmax(
            orog_norm
        )
    ),
)


# ------------------------------------------------------------
# Load normalization statistics
# ------------------------------------------------------------

with open(
    STATS_FILE,
    "r",
) as f:
    normalization_stats = json.load(
        f
    )


def find_orography_stats(
    obj,
    path="root",
):
    """
    Recursively locate the statistics associated with
    'orography_64'.

    Dataset preparation stores:
        static_stats["orography_64"] = st

    but this function is deliberately robust to an additional
    nesting level in normalization_stats.json.
    """

    matches = []

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            key_lower = str(
                key
            ).lower()

            current_path = (
                f"{path}.{key}"
            )

            if (
                key_lower
                == "orography_64"
            ):
                matches.append(
                    (
                        current_path,
                        value,
                    )
                )

            if (
                isinstance(
                    value,
                    dict,
                )
                and str(
                    value.get(
                        "name",
                        "",
                    )
                ).lower()
                == "orography_64"
            ):
                matches.append(
                    (
                        current_path,
                        value,
                    )
                )

            matches.extend(
                find_orography_stats(
                    value,
                    current_path,
                )
            )

    elif isinstance(
        obj,
        list,
    ):

        for i, value in enumerate(
            obj
        ):
            matches.extend(
                find_orography_stats(
                    value,
                    f"{path}[{i}]",
                )
            )

    return matches


matches = find_orography_stats(
    normalization_stats
)


# remove duplicated paths if recursive matching found the same object twice
unique = {}

for path_name, value in matches:
    unique[path_name] = value

matches = list(
    unique.items()
)


if not matches:
    print()
    print(
        "Top-level normalization-stat keys:"
    )

    if isinstance(
        normalization_stats,
        dict,
    ):
        for key in normalization_stats:
            print(
                "  ",
                key,
            )

    raise RuntimeError(
        "Could not locate 'orography_64' "
        "inside normalization_stats.json"
    )


print()
print(
    "Candidate orography-stat blocks:"
)

for path_name, value in matches:
    print(
        " ",
        path_name,
    )

    if isinstance(
        value,
        dict,
    ):
        print(
            "    keys:",
            list(
                value.keys()
            ),
        )


def numeric_scalar(
    value,
    label,
):
    a = np.asarray(
        value,
        dtype=np.float64,
    )

    if a.size != 1:
        raise RuntimeError(
            f"{label} is not scalar: "
            f"shape={a.shape}"
        )

    return float(
        a.reshape(-1)[0]
    )


selected_path = None
selected_stats = None
orog_mean = None
orog_std = None


for path_name, st in matches:

    if not isinstance(
        st,
        dict,
    ):
        continue

    # Normal expected names
    mean_key = None
    std_key = None

    for candidate in [
        "mean",
        "mu",
        "avg",
        "average",
    ]:
        if candidate in st:
            mean_key = candidate
            break

    for candidate in [
        "std",
        "sigma",
        "stdev",
        "standard_deviation",
    ]:
        if candidate in st:
            std_key = candidate
            break

    if (
        mean_key is None
        or std_key is None
    ):
        continue

    try:

        mean_value = numeric_scalar(
            st[mean_key],
            f"{path_name}.{mean_key}",
        )

        std_value = numeric_scalar(
            st[std_key],
            f"{path_name}.{std_key}",
        )

    except Exception:
        continue

    if not np.isfinite(
        mean_value
    ):
        continue

    if (
        not np.isfinite(
            std_value
        )
        or std_value <= 0
    ):
        continue

    selected_path = path_name
    selected_stats = st
    orog_mean = mean_value
    orog_std = std_value

    break


if selected_stats is None:
    raise RuntimeError(
        "Found orography_64 statistics, "
        "but could not identify scalar mean/std."
    )


print()
print(
    "Selected stats     :",
    selected_path,
)

print(
    "Orography mean [m] :",
    orog_mean,
)

print(
    "Orography std  [m] :",
    orog_std,
)


# ------------------------------------------------------------
# Exact inverse z-score transformation
# ------------------------------------------------------------

elevation = (
    orog_norm
    * orog_std
    + orog_mean
)


# ------------------------------------------------------------
# Physical sanity audit
# ------------------------------------------------------------

finite_elevation = elevation[
    np.isfinite(
        elevation
    )
]


if finite_elevation.size == 0:
    raise RuntimeError(
        "Reconstructed elevation contains "
        "no finite values."
    )


elev_min = float(
    np.nanmin(
        finite_elevation
    )
)

elev_max = float(
    np.nanmax(
        finite_elevation
    )
)

elev_p01 = float(
    np.nanpercentile(
        finite_elevation,
        1,
    )
)

elev_p50 = float(
    np.nanpercentile(
        finite_elevation,
        50,
    )
)

elev_p99 = float(
    np.nanpercentile(
        finite_elevation,
        99,
    )
)


print()
print(
    "Reconstructed range:",
    elev_min,
    "..",
    elev_max,
    "m",
)

print(
    "p01 / p50 / p99   :",
    elev_p01,
    elev_p50,
    elev_p99,
    "m",
)


# Chilean regional terrain should contain high Andes.
# These are broad sanity checks, not data manipulation.
if elev_max < 1500:
    raise RuntimeError(
        "Reconstructed elevation maximum "
        f"is implausibly low: {elev_max:.2f} m"
    )

if elev_max > 10000:
    raise RuntimeError(
        "Reconstructed elevation maximum "
        f"is implausibly high: {elev_max:.2f} m"
    )

if elev_min < -1000:
    raise RuntimeError(
        "Reconstructed elevation minimum "
        f"is implausibly low: {elev_min:.2f} m"
    )


# ------------------------------------------------------------
# Read original physical source from Model-E configuration
# only for provenance / metadata.
# ------------------------------------------------------------

with open(
    CONFIG_FILE,
    "r",
) as f:
    config = json.load(
        f
    )


def find_key_recursive(
    obj,
    wanted_key,
):
    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            if key == wanted_key:
                return value

            result = find_key_recursive(
                value,
                wanted_key,
            )

            if result is not None:
                return result

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            result = find_key_recursive(
                value,
                wanted_key,
            )

            if result is not None:
                return result

    return None


original_orography_path = find_key_recursive(
    config,
    "orography_path",
)


print()
print(
    "Original Model-E orography source:",
    original_orography_path,
)


# ------------------------------------------------------------
# Object expected by the remainder of this plotting script
# ------------------------------------------------------------

topo = {
    "array":
        elevation,

    "source":
        str(
            STATIC_FILE
        ),

    "key":
        (
            "channel 0 denormalized with "
            f"{selected_path}"
        ),

    "units":
        "m",

    "original_source":
        original_orography_path,

    "normalization_stats_file":
        str(
            STATS_FILE
        ),

    "normalization_mean_m":
        orog_mean,

    "normalization_std_m":
        orog_std,
}


# ============================================================
# APPLY COMMON SUPPORT FOR DISPLAY
# ============================================================

elevation_plot = elevation.copy()

elevation_plot[
    ~valid
] = np.nan


# ============================================================
# CONTOUR LEVELS
#
# 500 and 1500 m correspond directly to the elevation-band
# boundaries already used in the model diagnostics.
#
# 3000 m is added only when represented in the domain.
# ============================================================

ELEV_MAX = float(
    np.nanmax(
        elevation_plot
    )
)

CONTOUR_LEVELS = [
    500.0,
    1500.0,
]

if ELEV_MAX >= 3000.0:
    CONTOUR_LEVELS.append(
        3000.0
    )

CONTOUR_LEVELS = [
    lev
    for lev in CONTOUR_LEVELS
    if lev < ELEV_MAX
]


if not CONTOUR_LEVELS:
    raise RuntimeError(
        "No requested contour level lies inside "
        "the physical elevation range."
    )


# ============================================================
# GEOGRAPHIC ASPECT
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


def format_axis(
    ax,
    ylabel=True,
):

    ax.set_xlabel(
        "Longitude"
    )

    if ylabel:
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


# ============================================================
# COMMON TOPOGRAPHIC OVERLAY
#
# The same physical-elevation contours are shown in panels
# (a), (b), and (c) strictly as geographic reference.
#
# IMPORTANT:
# In panel (a), these contours are NOT model inputs.
# The no-fine-topography configuration excludes channels
# elevation, local orographic gradient, and roughness.
# ============================================================

TOPO_CONTOUR_COLOR = "#2B2B2B"
TOPO_CONTOUR_WIDTH = 0.82
TOPO_CONTOUR_ALPHA = 0.80


def add_topography_contours(ax):
    """
    Add the same physical elevation contours to a map panel.

    Contours are intentionally left unlabeled inside the plot
    to preserve readability. Elevation levels are documented
    in the figure caption and metadata.
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


# ============================================================
# ALTITUDINAL-BAND LABELS
#
# Contours are shown on all three panels.
# Band labels are shown only on panel (c) to avoid repetition.
#
# Their positions are derived automatically from the actual
# physical elevation field and common valid support.
# ============================================================

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


def get_band_mask(elevation, valid_mask, low=None, high=None):
    """
    Return valid cells belonging to one physical-elevation band.
    """

    m = (
        valid_mask
        & np.isfinite(elevation)
    )

    if low is not None:
        m &= elevation >= low

    if high is not None:
        m &= elevation < high

    return m


def representative_band_position(mask):
    """
    Pick a robust representative grid cell for one elevation band.

    Strategy:
      1. Compute median lon/lat of all cells in the band.
      2. Select the actual band cell closest to that median.

    This guarantees that the annotation is placed on a cell that
    genuinely belongs to the corresponding elevation band.
    """

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

    # Geographic approximation is sufficient at this domain size.
    # Longitude is adjusted by cos(latitude).
    coslat = np.cos(
        np.deg2rad(lat_med)
    )

    dist2 = (
        ((lons - lon_med) * coslat) ** 2
        + (lats - lat_med) ** 2
    )

    k = int(
        np.nanargmin(dist2)
    )

    return (
        float(lons[k]),
        float(lats[k]),
    )


def add_altitudinal_band_labels(ax):
    """
    Add one unobtrusive label per physical-elevation range.

    Labels are geographic annotations only; they are not an
    additional model field and do not modify plotted statistics.
    """

    positions = {}

    for band in ALTITUDE_BANDS:

        mask_band = get_band_mask(
            elevation=elevation,
            valid_mask=valid,
            low=band["low"],
            high=band["high"],
        )

        pos = representative_band_position(
            mask_band
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
                mask_band.sum()
            ),
        }

        ax.text(
            x,
            y,
            band["label"],

            ha="center",
            va="center",

            fontsize=7.2,
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
# COLORMAPS
# ============================================================

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


# ============================================================
# FIGURE LAYOUT
# ============================================================

fig = plt.figure(
    figsize=(14.8, 5.35)
)

gs = fig.add_gridspec(
    1,
    6,

    width_ratios=[
        1.0,
        1.0,
        0.045,
        0.11,
        1.0,
        0.045,
    ],

    left=0.055,
    right=0.965,
    bottom=0.11,
    top=0.92,

    wspace=0.20,
)


ax_a = fig.add_subplot(
    gs[0, 0]
)

ax_b = fig.add_subplot(
    gs[0, 1]
)

cax_rmse = fig.add_subplot(
    gs[0, 2]
)

gap = fig.add_subplot(
    gs[0, 3]
)

gap.axis(
    "off"
)

ax_c = fig.add_subplot(
    gs[0, 4]
)

cax_delta = fig.add_subplot(
    gs[0, 5]
)


# ============================================================
# PANEL (a)
# ============================================================

im_a = ax_a.pcolormesh(
    LON,
    LAT,
    rmse_notopo_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_rmse,

    vmin=0.0,
    vmax=RMSE_VMAX,
)

ax_a.set_title(
    "(a) TopoFusion-Res without fine topography"
)

format_axis(
    ax_a,
    ylabel=True,
)

contours_a = add_topography_contours(
    ax_a
)


# ============================================================
# PANEL (b)
# ============================================================

im_b = ax_b.pcolormesh(
    LON,
    LAT,
    rmse_full_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_rmse,

    vmin=0.0,
    vmax=RMSE_VMAX,
)

ax_b.set_title(
    "(b) TopoFusion-Res"
)

format_axis(
    ax_b,
    ylabel=False,
)

contours_b = add_topography_contours(
    ax_b
)


cb_rmse = fig.colorbar(
    im_b,
    cax=cax_rmse,

    extend="max",
    extendfrac=0.035,
)

cb_rmse.set_label(
    r"RMSE ($^\circ$C)"
)


# ============================================================
# PANEL (c)
# ============================================================

delta_norm = TwoSlopeNorm(
    vmin=-DELTA_LIM,
    vcenter=0.0,
    vmax=DELTA_LIM,
)

im_c = ax_c.pcolormesh(
    LON,
    LAT,
    delta_plot,

    shading="auto",
    rasterized=True,

    cmap=cmap_delta,
    norm=delta_norm,
)

ax_c.set_title(
    r"(c) $\Delta$RMSE = No fine topography $-$ Full"
)

format_axis(
    ax_c,
    ylabel=False,
)

contours_c = add_topography_contours(
    ax_c
)

band_label_positions = add_altitudinal_band_labels(
    ax_c
)


# ============================================================
# TOPOGRAPHIC CONTOURS
#
# Already overlaid consistently on panels (a), (b), and (c).
# No in-map contour labels are used to avoid visual clutter.
# ============================================================

cb_delta = fig.colorbar(
    im_c,
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
# SUMMARY / METADATA
# ============================================================

v = delta[
    valid
]

v = v[
    np.isfinite(v)
]


summary = {
    "figure":
        "Figure 4",

    "comparison":
        (
            "Fine-topography ablation: "
            "TopoFusion-Res without fine topography "
            "versus full TopoFusion-Res"
        ),

    "support": {
        "n_common_cells":
            int(
                valid.sum()
            ),
    },

    "sign_convention":
        (
            "Delta RMSE = TopoFusion-Res without fine "
            "topography RMSE minus full TopoFusion-Res "
            "RMSE; positive values indicate lower RMSE "
            "after including fine-scale topography."
        ),

    "spatial_delta_rmse_C": {
        "mean":
            float(
                np.mean(v)
            ),

        "median":
            float(
                np.median(v)
            ),

        "p10":
            float(
                np.percentile(v, 10)
            ),

        "p90":
            float(
                np.percentile(v, 90)
            ),

        "fraction_positive":
            float(
                np.mean(v > 0)
            ),

        "fraction_negative":
            float(
                np.mean(v < 0)
            ),
    },

    "topography_overlay": {
        "panels": [
            "a",
            "b",
            "c",
        ],

        "meaning":
            (
                "Physical elevation contours from the "
                "fine-scale topographic field, displayed "
                "in all three panels as geographic reference."
            ),

        "source":
            topo["source"],

        "source_key":
            topo["key"],

        "source_units":
            topo["units"],

        "original_orography_source":
            topo.get(
                "original_source"
            ),

        "normalization_stats_file":
            topo.get(
                "normalization_stats_file"
            ),

        "normalization_mean_m":
            topo.get(
                "normalization_mean_m"
            ),

        "normalization_std_m":
            topo.get(
                "normalization_std_m"
            ),

        "elevation_min_m":
            float(
                np.nanmin(
                    elevation_plot
                )
            ),

        "elevation_max_m":
            float(
                np.nanmax(
                    elevation_plot
                )
            ),

        "contour_levels_m": [
            float(x)
            for x in CONTOUR_LEVELS
        ],

        "altitudinal_band_labels": [
            "< 500 m",
            "500–1500 m",
            "1500–3000 m",
            "> 3000 m",
        ],

        "band_labels_panel":
            "c",

        "band_label_positions":
            band_label_positions,

        "diagnostic_band_relation":
            (
                "500 m and 1500 m correspond to the "
                "elevation-band boundaries used in the "
                "topographic diagnostics."
            ),

        "note":
            (
                "Contours are displayed only for spatial "
                "interpretation and do not alter RMSE maps "
                "or any reported statistics. In panel (a), "
                "the contours are geographic reference only "
                "and are not inputs to the no-fine-topography "
                "model configuration."
            ),
    },

    "visualization": {
        "display_percentile":
            DISPLAY_PERCENTILE,

        "rmse_vmax_C":
            RMSE_VMAX,

        "delta_symmetric_limit_C":
            DELTA_LIM,

        "note":
            (
                "Color limits use the spatial 98th "
                "percentile for visualization only. "
                "Raw maps and reported statistics use "
                "all finite values without clipping."
            ),
    },
}


with open(
    META,
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
print("FIGURE 4 — TOPOGRAPHIC CONTOUR VERSION")
print("=" * 90)

print(
    "RMSE source :",
    RMSE_FILE,
)

print(
    "Coord source:",
    RMSE_FILE,
)

print(
    "Topo source :",
    topo["source"],
)

print(
    "Topo key    :",
    topo["key"],
)

print(
    "Original DEM:",
    topo.get(
        "original_source"
    ),
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
    ", ".join(
        f"{int(x)} m"
        for x in CONTOUR_LEVELS
    ),
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
    "Fraction ΔRMSE > 0:",
    float(
        np.mean(v > 0)
    ),
)

print()

print("Saved:")
print(PNG)
print(PDF)
print(META)

print()
print("DONE")
