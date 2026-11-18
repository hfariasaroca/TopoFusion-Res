from pathlib import Path
import csv
import json

import numpy as np


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
MASK_FILE = CUBE / "mask.dat"

OLD_TAIL_FILE = (
    ROOT
    / "diagnostics"
    / "final_e2_vs_unet_advanced_diagnostics"
    / "final_extreme_p01_p99_maps_e2_unet_rcm.npz"
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

OUT_DIA = PAPER / "diagnostics"
OUT_TAB = PAPER / "tables"

OUT_DIA.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIA / "54_final_seeded_temperature_tail_maps.npz"
OUT_JSON = OUT_DIA / "54_final_seeded_temperature_tail_summary.json"
OUT_CSV = OUT_TAB / "54_final_seeded_temperature_tail_summary.csv"
OUT_BANDS = OUT_TAB / "54_final_seeded_temperature_tail_by_altitude.csv"


# ============================================================
# INPUT AUDIT
# ============================================================

required = [
    META_FILE,
    RCM_FILE,
    TOPO_FILE,
    UNET_FILE,
    MASK_FILE,
    STATIC_FILE,
    STATS_FILE,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(p)

with open(META_FILE, "r") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])

if shape != (31390, 64, 64):
    raise RuntimeError(
        f"Unexpected cube shape: {shape}"
    )

N, H, W = shape

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

mask = np.memmap(
    MASK_FILE,
    dtype="bool",
    mode="r",
    shape=shape,
)

common = np.all(mask, axis=0)

cell_idx = np.flatnonzero(
    common.ravel()
)

NC = len(cell_idx)

if NC != 1852:
    raise RuntimeError(
        f"Expected 1852 common cells, got {NC}"
    )

print("=" * 90)
print("FINAL SEEDED TEMPERATURE-TAIL AUDIT")
print("=" * 90)
print("Shape        :", shape)
print("Days         :", N)
print("Common cells :", NC)
print("Pairs        :", N * NC)
print()
print("TopoFusion cube:")
print(" ", TOPO_FILE)

if OLD_TAIL_FILE.exists():
    print()
    print("NOTE:")
    print("Old p01/p99 file exists but is intentionally NOT used:")
    print(" ", OLD_TAIL_FILE)
    print("This audit is recomputed from e2_full_seeded.dat.")


# ============================================================
# HELPERS
# ============================================================

def metric_summary(pred, ref):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    err = pred - ref
    ae = np.abs(err)

    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_error": float(np.median(ae)),
        "p90_abs_error": float(np.percentile(ae, 90)),
    }


def compare_abs_errors(topo_pred, unet_pred, ref):
    et = np.abs(
        np.asarray(topo_pred, dtype=np.float64)
        - np.asarray(ref, dtype=np.float64)
    )

    eu = np.abs(
        np.asarray(unet_pred, dtype=np.float64)
        - np.asarray(ref, dtype=np.float64)
    )

    delta = eu - et

    return {
        "mean_advantage_topofusion": float(np.mean(delta)),
        "median_advantage_topofusion": float(np.median(delta)),
        "fraction_topofusion_closer": float(np.mean(delta > 0)),
        "fraction_unet_closer": float(np.mean(delta < 0)),
        "fraction_equal": float(np.mean(delta == 0)),
    }


# ============================================================
# P01/P99 MAPS
#
# Chunk over cells so we never load the complete cubes.
# ============================================================

def compute_tail_quantiles(arr, name, block_cells=64):
    flat = arr.reshape(N, H * W)

    q01 = np.full(
        H * W,
        np.nan,
        dtype=np.float64,
    )

    q99 = np.full(
        H * W,
        np.nan,
        dtype=np.float64,
    )

    nblocks = (
        len(cell_idx)
        + block_cells - 1
    ) // block_cells

    for ib, start in enumerate(
        range(0, len(cell_idx), block_cells),
        start=1,
    ):
        ids = cell_idx[
            start:start + block_cells
        ]

        x = np.asarray(
            flat[:, ids],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(x)):
            raise RuntimeError(
                f"Non-finite values in {name}, "
                f"cell block {ib}"
            )

        q = np.percentile(
            x,
            [1.0, 99.0],
            axis=0,
        )

        q01[ids] = q[0]
        q99[ids] = q[1]

        print(
            f"{name:18s} "
            f"quantile block "
            f"{ib:02d}/{nblocks:02d}"
        )

    return (
        q01.reshape(H, W),
        q99.reshape(H, W),
    )


print()
print("=" * 90)
print("COMPUTING CELL-WISE P01 / P99")
print("=" * 90)

rcm_p01, rcm_p99 = compute_tail_quantiles(
    rcm,
    "CNRM-ALADIN64",
)

topo_p01, topo_p99 = compute_tail_quantiles(
    topo,
    "TopoFusion-Res",
)

unet_p01, unet_p99 = compute_tail_quantiles(
    unet,
    "U-Net",
)


# ============================================================
# VALID-CELL VECTORS
# ============================================================

v = common

r01 = rcm_p01[v]
t01 = topo_p01[v]
u01 = unet_p01[v]

r99 = rcm_p99[v]
t99 = topo_p99[v]
u99 = unet_p99[v]


# ============================================================
# QUANTILE-VALUE FIDELITY
# ============================================================

p01_topo_stats = metric_summary(
    t01,
    r01,
)

p01_unet_stats = metric_summary(
    u01,
    r01,
)

p99_topo_stats = metric_summary(
    t99,
    r99,
)

p99_unet_stats = metric_summary(
    u99,
    r99,
)

p01_comparison = compare_abs_errors(
    t01,
    u01,
    r01,
)

p99_comparison = compare_abs_errors(
    t99,
    u99,
    r99,
)


# ============================================================
# P01--P99 CENTRAL 98% RANGE
# ============================================================

range_rcm = r99 - r01
range_topo = t99 - t01
range_unet = u99 - u01

range_topo_stats = metric_summary(
    range_topo,
    range_rcm,
)

range_unet_stats = metric_summary(
    range_unet,
    range_rcm,
)

range_ratio_topo = (
    100.0
    * range_topo
    / range_rcm
)

range_ratio_unet = (
    100.0
    * range_unet
    / range_rcm
)


# ============================================================
# EXTREME-DAY CONDITIONED ANALYSIS
#
# Warm event:
#   RCM >= its cell-specific p99
#
# Cold event:
#   RCM <= its cell-specific p01
#
# We assess:
#   - model error on those SAME RCM extreme days
#   - frequency relative to RCM threshold
#   - precision/recall for threshold crossings
# ============================================================

acc = {}

for model in [
    "TopoFusion-Res",
    "U-Net",
]:
    acc[model] = {}

    for tail in [
        "cold_p01",
        "warm_p99",
    ]:
        acc[model][tail] = {
            "target_count": 0,
            "model_event_count": 0,
            "intersection_count": 0,
            "sum_error": 0.0,
            "sum_abs_error": 0.0,
            "sum_sq_error": 0.0,
        }


rcm_flat = rcm.reshape(N, H * W)
topo_flat = topo.reshape(N, H * W)
unet_flat = unet.reshape(N, H * W)

TIME_CHUNK = 365

print()
print("=" * 90)
print("EXTREME-DAY CONDITIONED ANALYSIS")
print("=" * 90)

for t0 in range(0, N, TIME_CHUNK):
    t1 = min(
        t0 + TIME_CHUNK,
        N,
    )

    rr = np.asarray(
        rcm_flat[
            t0:t1,
            cell_idx,
        ],
        dtype=np.float64,
    )

    tt = np.asarray(
        topo_flat[
            t0:t1,
            cell_idx,
        ],
        dtype=np.float64,
    )

    uu = np.asarray(
        unet_flat[
            t0:t1,
            cell_idx,
        ],
        dtype=np.float64,
    )

    if not (
        np.all(np.isfinite(rr))
        and np.all(np.isfinite(tt))
        and np.all(np.isfinite(uu))
    ):
        raise RuntimeError(
            f"Non-finite value in time block "
            f"{t0}:{t1}"
        )

    cold_ref = (
        rr
        <= r01[None, :]
    )

    warm_ref = (
        rr
        >= r99[None, :]
    )

    for model_name, pred in [
        ("TopoFusion-Res", tt),
        ("U-Net", uu),
    ]:

        err = pred - rr

        model_cold = (
            pred
            <= r01[None, :]
        )

        model_warm = (
            pred
            >= r99[None, :]
        )

        for tail_name, ref_event, model_event in [
            (
                "cold_p01",
                cold_ref,
                model_cold,
            ),
            (
                "warm_p99",
                warm_ref,
                model_warm,
            ),
        ]:

            a = acc[
                model_name
            ][
                tail_name
            ]

            a["target_count"] += int(
                ref_event.sum()
            )

            a["model_event_count"] += int(
                model_event.sum()
            )

            a["intersection_count"] += int(
                (
                    ref_event
                    & model_event
                ).sum()
            )

            e = err[
                ref_event
            ]

            a["sum_error"] += float(
                e.sum()
            )

            a["sum_abs_error"] += float(
                np.abs(e).sum()
            )

            a["sum_sq_error"] += float(
                np.square(e).sum()
            )

    print(
        f"days {t0:05d}:{t1:05d}"
    )


TOTAL_PAIRS = N * NC


def finalize_event_stats(a):
    nt = a["target_count"]
    nm = a["model_event_count"]
    ni = a["intersection_count"]

    precision = (
        ni / nm
        if nm > 0
        else np.nan
    )

    recall = (
        ni / nt
        if nt > 0
        else np.nan
    )

    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
    ):
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = np.nan

    return {
        "n_reference_extreme_points":
            int(nt),

        "n_model_threshold_crossings":
            int(nm),

        "reference_frequency_pct":
            100.0 * nt / TOTAL_PAIRS,

        "model_frequency_pct":
            100.0 * nm / TOTAL_PAIRS,

        "bias_on_reference_extreme_days":
            a["sum_error"] / nt,

        "mae_on_reference_extreme_days":
            a["sum_abs_error"] / nt,

        "rmse_on_reference_extreme_days":
            np.sqrt(
                a["sum_sq_error"] / nt
            ),

        "threshold_precision":
            float(precision),

        "threshold_recall":
            float(recall),

        "threshold_f1":
            float(f1),
    }


event_stats = {
    model: {
        tail: finalize_event_stats(
            acc[model][tail]
        )
        for tail in [
            "cold_p01",
            "warm_p99",
        ]
    }
    for model in acc
}


# ============================================================
# ALTITUDE BANDS
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

with open(STATS_FILE, "r") as f:
    stats = json.load(f)

orog_stats = (
    stats[
        "static_64x64"
    ][
        "orography_64"
    ]
)

orography = (
    static[0]
    * float(orog_stats["std"])
    + float(orog_stats["mean"])
)

orog_valid = orography[v]

bands = [
    (
        "<500 m",
        orog_valid < 500.0,
    ),
    (
        "500-1500 m",
        (
            (orog_valid >= 500.0)
            & (orog_valid < 1500.0)
        ),
    ),
    (
        "1500-3000 m",
        (
            (orog_valid >= 1500.0)
            & (orog_valid < 3000.0)
        ),
    ),
    (
        ">3000 m",
        orog_valid >= 3000.0,
    ),
]


band_rows = []

for band_name, bm in bands:

    for tail_name, refq, tq, uq in [
        (
            "p01",
            r01,
            t01,
            u01,
        ),
        (
            "p99",
            r99,
            t99,
            u99,
        ),
    ]:

        et = np.abs(
            tq[bm]
            - refq[bm]
        )

        eu = np.abs(
            uq[bm]
            - refq[bm]
        )

        delta = eu - et

        band_rows.append({
            "band":
                band_name,

            "n_cells":
                int(bm.sum()),

            "tail":
                tail_name,

            "topofusion_mae_C":
                float(
                    np.mean(et)
                ),

            "unet_mae_C":
                float(
                    np.mean(eu)
                ),

            "mean_advantage_topofusion_C":
                float(
                    np.mean(delta)
                ),

            "fraction_topofusion_closer":
                float(
                    np.mean(delta > 0)
                ),
        })


# ============================================================
# SAVE MAPS
# ============================================================

np.savez_compressed(
    OUT_NPZ,

    common_mask=common,

    orography_m=orography,

    rcm_p01=rcm_p01,
    topofusion_p01=topo_p01,
    unet_p01=unet_p01,

    rcm_p99=rcm_p99,
    topofusion_p99=topo_p99,
    unet_p99=unet_p99,

    p01_error_topofusion=(
        topo_p01
        - rcm_p01
    ),

    p01_error_unet=(
        unet_p01
        - rcm_p01
    ),

    p01_abs_error_advantage_topofusion=(
        np.abs(
            unet_p01
            - rcm_p01
        )
        -
        np.abs(
            topo_p01
            - rcm_p01
        )
    ),

    p99_error_topofusion=(
        topo_p99
        - rcm_p99
    ),

    p99_error_unet=(
        unet_p99
        - rcm_p99
    ),

    p99_abs_error_advantage_topofusion=(
        np.abs(
            unet_p99
            - rcm_p99
        )
        -
        np.abs(
            topo_p99
            - rcm_p99
        )
    ),
)


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {
    "source": {
        "scenario":
            "SSP3-7.0",

        "period":
            "2015-2100",

        "days":
            int(N),

        "common_cells":
            int(NC),

        "pairs":
            int(N * NC),

        "topofusion_cube":
            str(TOPO_FILE),

        "unet_cube":
            str(UNET_FILE),

        "reference_cube":
            str(RCM_FILE),

        "old_tail_file_ignored":
            str(OLD_TAIL_FILE),
    },

    "p01": {
        "TopoFusion-Res":
            p01_topo_stats,

        "U-Net":
            p01_unet_stats,

        "comparison":
            p01_comparison,
    },

    "p99": {
        "TopoFusion-Res":
            p99_topo_stats,

        "U-Net":
            p99_unet_stats,

        "comparison":
            p99_comparison,
    },

    "central_98pct_range": {
        "rcm_mean_C":
            float(
                np.mean(range_rcm)
            ),

        "topofusion_mean_C":
            float(
                np.mean(range_topo)
            ),

        "unet_mean_C":
            float(
                np.mean(range_unet)
            ),

        "topofusion_mean_ratio_pct":
            float(
                np.mean(
                    range_ratio_topo
                )
            ),

        "unet_mean_ratio_pct":
            float(
                np.mean(
                    range_ratio_unet
                )
            ),

        "TopoFusion-Res":
            range_topo_stats,

        "U-Net":
            range_unet_stats,
    },

    "reference_extreme_days":
        event_stats,

    "altitude_bands":
        band_rows,
}

with open(
    OUT_JSON,
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# CSV SUMMARY
# ============================================================

rows = []

for tail_name, model_stats in [
    (
        "p01",
        {
            "TopoFusion-Res":
                p01_topo_stats,
            "U-Net":
                p01_unet_stats,
        },
    ),
    (
        "p99",
        {
            "TopoFusion-Res":
                p99_topo_stats,
            "U-Net":
                p99_unet_stats,
        },
    ),
]:

    for model, st in model_stats.items():

        rows.append({
            "analysis":
                tail_name,

            "model":
                model,

            **st,
        })


with open(
    OUT_CSV,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "analysis",
            "model",
            "bias",
            "mae",
            "rmse",
            "median_abs_error",
            "p90_abs_error",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


with open(
    OUT_BANDS,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "band",
            "n_cells",
            "tail",
            "topofusion_mae_C",
            "unet_mae_C",
            "mean_advantage_topofusion_C",
            "fraction_topofusion_closer",
        ],
    )

    writer.writeheader()
    writer.writerows(band_rows)


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_model_stats(label, st):
    print(
        f"{label:18s} "
        f"bias={st['bias']:+.4f} C  "
        f"MAE={st['mae']:.4f} C  "
        f"RMSE={st['rmse']:.4f} C  "
        f"median|e|={st['median_abs_error']:.4f} C"
    )


print()
print("=" * 90)
print("P01 — COLD DISTRIBUTION TAIL")
print("=" * 90)

print_model_stats(
    "TopoFusion-Res",
    p01_topo_stats,
)

print_model_stats(
    "U-Net",
    p01_unet_stats,
)

print(
    "Mean |error| advantage TopoFusion:",
    f"{p01_comparison['mean_advantage_topofusion']:+.4f} C",
)

print(
    "Cells TopoFusion closer:",
    f"{100*p01_comparison['fraction_topofusion_closer']:.2f}%",
)

print(
    "Cells U-Net closer:",
    f"{100*p01_comparison['fraction_unet_closer']:.2f}%",
)


print()
print("=" * 90)
print("P99 — WARM DISTRIBUTION TAIL")
print("=" * 90)

print_model_stats(
    "TopoFusion-Res",
    p99_topo_stats,
)

print_model_stats(
    "U-Net",
    p99_unet_stats,
)

print(
    "Mean |error| advantage TopoFusion:",
    f"{p99_comparison['mean_advantage_topofusion']:+.4f} C",
)

print(
    "Cells TopoFusion closer:",
    f"{100*p99_comparison['fraction_topofusion_closer']:.2f}%",
)

print(
    "Cells U-Net closer:",
    f"{100*p99_comparison['fraction_unet_closer']:.2f}%",
)


print()
print("=" * 90)
print("CENTRAL 98% TEMPERATURE RANGE = P99 - P01")
print("=" * 90)

print(
    "RCM mean range       :",
    f"{np.mean(range_rcm):.4f} C",
)

print(
    "TopoFusion mean range:",
    f"{np.mean(range_topo):.4f} C",
    f"({np.mean(range_ratio_topo):.2f}% of RCM)",
)

print(
    "U-Net mean range     :",
    f"{np.mean(range_unet):.4f} C",
    f"({np.mean(range_ratio_unet):.2f}% of RCM)",
)


print()
print("=" * 90)
print("ERROR ON ACTUAL RCM EXTREME DAYS")
print("=" * 90)

for tail in [
    "cold_p01",
    "warm_p99",
]:

    print()
    print(tail.upper())

    for model in [
        "TopoFusion-Res",
        "U-Net",
    ]:

        s = event_stats[
            model
        ][
            tail
        ]

        print(
            f"{model:18s} "
            f"bias={s['bias_on_reference_extreme_days']:+.4f} C  "
            f"MAE={s['mae_on_reference_extreme_days']:.4f} C  "
            f"RMSE={s['rmse_on_reference_extreme_days']:.4f} C"
        )

        print(
            f"{'':18s} "
            f"frequency={s['model_frequency_pct']:.3f}%  "
            f"precision={s['threshold_precision']:.3f}  "
            f"recall={s['threshold_recall']:.3f}  "
            f"F1={s['threshold_f1']:.3f}"
        )


print()
print("=" * 90)
print("ALTITUDE-BAND TAIL MAE")
print("=" * 90)

for row in band_rows:

    print(
        f"{row['band']:12s} "
        f"{row['tail']:3s} "
        f"n={row['n_cells']:4d}  "
        f"Topo={row['topofusion_mae_C']:.4f} C  "
        f"U-Net={row['unet_mae_C']:.4f} C  "
        f"adv={row['mean_advantage_topofusion_C']:+.4f} C  "
        f"Topo closer={100*row['fraction_topofusion_closer']:.1f}%"
    )


print()
print("=" * 90)
print("OUTPUTS")
print("=" * 90)

print(OUT_NPZ)
print(OUT_JSON)
print(OUT_CSV)
print(OUT_BANDS)

print()
print("DONE")
