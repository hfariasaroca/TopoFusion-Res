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

POOLED_SUMMARY = (
    PAPER
    / "diagnostics"
    / "54_final_seeded_temperature_tail_summary.json"
)

OUT_DIA = PAPER / "diagnostics"
OUT_TAB = PAPER / "tables"

OUT_DIA.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

OUT_JSON = (
    OUT_DIA
    / "55_monthly_seasonal_temperature_tail_summary.json"
)

OUT_MONTH = (
    OUT_TAB
    / "55_monthly_temperature_tail_summary.csv"
)

OUT_SEASON = (
    OUT_TAB
    / "55_seasonal_temperature_tail_summary.csv"
)

OUT_EVENT = (
    OUT_TAB
    / "55_monthly_seasonal_extreme_day_metrics.csv"
)


# ============================================================
# INPUT AUDIT
# ============================================================

for p in [
    META_FILE,
    RCM_FILE,
    TOPO_FILE,
    UNET_FILE,
    MASK_FILE,
]:
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

if N % 365 != 0:
    raise RuntimeError(
        "Expected complete 365-day years."
    )

NY = N // 365

if NY != 86:
    raise RuntimeError(
        f"Expected 86 years, got {NY}"
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

mask = np.memmap(
    MASK_FILE,
    dtype="bool",
    mode="r",
    shape=shape,
)

common = np.all(
    mask,
    axis=0,
)

cell_idx = np.flatnonzero(
    common.ravel()
)

NC = len(cell_idx)

if NC != 1852:
    raise RuntimeError(
        f"Expected 1852 common cells, got {NC}"
    )


print("=" * 100)
print("MONTHLY / SEASONAL TEMPERATURE-TAIL SENSITIVITY")
print("=" * 100)

print("Shape        :", shape)
print("Years        :", NY)
print("Days         :", N)
print("Common cells :", NC)
print("Pairs        :", N * NC)
print()
print("TopoFusion cube:")
print(" ", TOPO_FILE)


# ============================================================
# 365-DAY CALENDAR
# ============================================================

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MONTH_LENGTHS = np.array(
    [
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
    ],
    dtype=int,
)

if int(MONTH_LENGTHS.sum()) != 365:
    raise RuntimeError(
        "Month lengths do not sum to 365."
    )

month_of_doy = np.empty(
    365,
    dtype=np.int16,
)

cursor = 0

for im, nd in enumerate(
    MONTH_LENGTHS,
    start=1,
):
    month_of_doy[
        cursor:cursor + nd
    ] = im

    cursor += nd


month_of_time = np.tile(
    month_of_doy,
    NY,
)


# Southern Hemisphere seasons
SEASON_DEFS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

SEASON_LABELS = {
    "DJF": "DJF (austral summer)",
    "MAM": "MAM (austral autumn)",
    "JJA": "JJA (austral winter)",
    "SON": "SON (austral spring)",
}


# ============================================================
# PERIOD DEFINITIONS
# ============================================================

periods = []

for im, name in enumerate(
    MONTH_NAMES,
    start=1,
):
    tids = np.flatnonzero(
        month_of_time == im
    )

    periods.append({
        "kind":
            "month",

        "name":
            name,

        "order":
            im,

        "months":
            (im,),

        "time_ids":
            tids,
    })


for order, (
    season,
    months,
) in enumerate(
    SEASON_DEFS.items(),
    start=1,
):
    tids = np.flatnonzero(
        np.isin(
            month_of_time,
            months,
        )
    )

    periods.append({
        "kind":
            "season",

        "name":
            season,

        "label":
            SEASON_LABELS[season],

        "order":
            order,

        "months":
            months,

        "time_ids":
            tids,
    })


print()
print("PERIOD SUPPORT")

for p in periods:
    print(
        f"{p['kind']:7s} "
        f"{p['name']:3s}  "
        f"n_days={len(p['time_ids']):5d}"
    )


# ============================================================
# STORAGE FOR QUANTILES
#
# keys:
#   quantiles[period_key][model]["p01"/"p99"]
# each vector has NC cells.
# ============================================================

quantiles = {}

for p in periods:
    key = (
        p["kind"],
        p["name"],
    )

    quantiles[key] = {
        "RCM": {
            "p01":
                np.full(
                    NC,
                    np.nan,
                    dtype=np.float64,
                ),

            "p99":
                np.full(
                    NC,
                    np.nan,
                    dtype=np.float64,
                ),
        },

        "TopoFusion-Res": {
            "p01":
                np.full(
                    NC,
                    np.nan,
                    dtype=np.float64,
                ),

            "p99":
                np.full(
                    NC,
                    np.nan,
                    dtype=np.float64,
                ),
        },

        "U-Net": {
            "p01":
                np.full(
                    NC,
                    np.nan,
                    dtype=np.float64,
                ),

            "p99":
                np.full(
                    NC,
                    np.nan,
                    dtype=np.float64,
                ),
        },
    }


# ============================================================
# QUANTILES
#
# Read each cell block only once per model.
# Then calculate all month/season quantiles from that block.
# ============================================================

def compute_all_period_quantiles(
    arr,
    model_name,
    block_cells=64,
):
    flat = arr.reshape(
        N,
        H * W,
    )

    nblocks = (
        NC
        + block_cells - 1
    ) // block_cells

    for ib, start in enumerate(
        range(
            0,
            NC,
            block_cells,
        ),
        start=1,
    ):
        stop = min(
            start + block_cells,
            NC,
        )

        ids_global = cell_idx[
            start:stop
        ]

        x = np.asarray(
            flat[
                :,
                ids_global,
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(x)
        ):
            raise RuntimeError(
                f"Non-finite values in "
                f"{model_name}, block {ib}"
            )

        for p in periods:
            key = (
                p["kind"],
                p["name"],
            )

            xp = x[
                p["time_ids"],
                :
            ]

            q = np.percentile(
                xp,
                [1.0, 99.0],
                axis=0,
            )

            quantiles[
                key
            ][
                model_name
            ][
                "p01"
            ][start:stop] = q[0]

            quantiles[
                key
            ][
                model_name
            ][
                "p99"
            ][start:stop] = q[1]

        print(
            f"{model_name:18s} "
            f"block {ib:02d}/{nblocks:02d}"
        )


print()
print("=" * 100)
print("COMPUTING MONTHLY / SEASONAL P01-P99")
print("=" * 100)

compute_all_period_quantiles(
    rcm,
    "RCM",
)

compute_all_period_quantiles(
    topo,
    "TopoFusion-Res",
)

compute_all_period_quantiles(
    unet,
    "U-Net",
)


# ============================================================
# HELPERS
# ============================================================

def metric_summary(
    pred,
    ref,
):
    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    ref = np.asarray(
        ref,
        dtype=np.float64,
    )

    err = pred - ref
    ae = np.abs(err)

    return {
        "bias_C":
            float(
                np.mean(err)
            ),

        "mae_C":
            float(
                np.mean(ae)
            ),

        "rmse_C":
            float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            ),

        "median_abs_error_C":
            float(
                np.median(ae)
            ),
    }


def comparison_summary(
    topo_pred,
    unet_pred,
    ref,
):
    et = np.abs(
        topo_pred - ref
    )

    eu = np.abs(
        unet_pred - ref
    )

    delta = eu - et

    return {
        "mean_advantage_topofusion_C":
            float(
                np.mean(delta)
            ),

        "median_advantage_topofusion_C":
            float(
                np.median(delta)
            ),

        "fraction_topofusion_closer":
            float(
                np.mean(
                    delta > 0
                )
            ),

        "fraction_unet_closer":
            float(
                np.mean(
                    delta < 0
                )
            ),
    }


# ============================================================
# QUANTILE FIDELITY SUMMARY
# ============================================================

quantile_rows = []

for p in periods:

    key = (
        p["kind"],
        p["name"],
    )

    q = quantiles[key]

    for tail in [
        "p01",
        "p99",
    ]:

        ref = q["RCM"][tail]

        topo_q = q[
            "TopoFusion-Res"
        ][tail]

        unet_q = q[
            "U-Net"
        ][tail]

        topo_stats = metric_summary(
            topo_q,
            ref,
        )

        unet_stats = metric_summary(
            unet_q,
            ref,
        )

        comp = comparison_summary(
            topo_q,
            unet_q,
            ref,
        )

        for model, st in [
            (
                "TopoFusion-Res",
                topo_stats,
            ),
            (
                "U-Net",
                unet_stats,
            ),
        ]:
            quantile_rows.append({
                "kind":
                    p["kind"],

                "period":
                    p["name"],

                "order":
                    p["order"],

                "tail":
                    tail,

                "model":
                    model,

                "n_days":
                    len(
                        p["time_ids"]
                    ),

                **st,

                "mean_advantage_topofusion_C":
                    comp[
                        "mean_advantage_topofusion_C"
                    ],

                "fraction_topofusion_closer":
                    comp[
                        "fraction_topofusion_closer"
                    ],
            })


# ============================================================
# P01-P99 RANGE FIDELITY
# ============================================================

range_rows = []

for p in periods:

    key = (
        p["kind"],
        p["name"],
    )

    q = quantiles[key]

    r_range = (
        q["RCM"]["p99"]
        - q["RCM"]["p01"]
    )

    t_range = (
        q["TopoFusion-Res"]["p99"]
        - q["TopoFusion-Res"]["p01"]
    )

    u_range = (
        q["U-Net"]["p99"]
        - q["U-Net"]["p01"]
    )

    range_rows.append({
        "kind":
            p["kind"],

        "period":
            p["name"],

        "order":
            p["order"],

        "rcm_mean_range_C":
            float(
                np.mean(r_range)
            ),

        "topofusion_mean_range_C":
            float(
                np.mean(t_range)
            ),

        "unet_mean_range_C":
            float(
                np.mean(u_range)
            ),

        "topofusion_ratio_pct":
            float(
                100.0
                * np.mean(
                    t_range / r_range
                )
            ),

        "unet_ratio_pct":
            float(
                100.0
                * np.mean(
                    u_range / r_range
                )
            ),
    })


# ============================================================
# EVENT-CONDITIONED ANALYSIS
#
# For each period:
#  cold = RCM <= period-specific p01
#  warm = RCM >= period-specific p99
#
# Threshold crossings in each model are evaluated against
# the SAME RCM-defined threshold.
# ============================================================

acc = {}

for p in periods:
    key = (
        p["kind"],
        p["name"],
    )

    acc[key] = {}

    for model in [
        "TopoFusion-Res",
        "U-Net",
    ]:
        acc[key][model] = {}

        for tail in [
            "p01",
            "p99",
        ]:
            acc[key][model][tail] = {
                "reference_count":
                    0,

                "model_count":
                    0,

                "intersection_count":
                    0,

                "sum_error":
                    0.0,

                "sum_abs_error":
                    0.0,

                "sum_sq_error":
                    0.0,
            }


rcm_flat = rcm.reshape(
    N,
    H * W,
)

topo_flat = topo.reshape(
    N,
    H * W,
)

unet_flat = unet.reshape(
    N,
    H * W,
)


print()
print("=" * 100)
print("EVENT-CONDITIONED MONTHLY / SEASONAL ANALYSIS")
print("=" * 100)


# Process complete years.
for iy in range(NY):

    t0 = iy * 365
    t1 = t0 + 365

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
            f"Non-finite values in year {iy}"
        )

    months_year = month_of_doy

    for p in periods:

        key = (
            p["kind"],
            p["name"],
        )

        local = np.flatnonzero(
            np.isin(
                months_year,
                p["months"],
            )
        )

        rrp = rr[local]
        ttp = tt[local]
        uup = uu[local]

        q = quantiles[key]

        ref_p01 = q["RCM"]["p01"]
        ref_p99 = q["RCM"]["p99"]

        ref_cold = (
            rrp
            <= ref_p01[None, :]
        )

        ref_warm = (
            rrp
            >= ref_p99[None, :]
        )

        for model_name, pred in [
            (
                "TopoFusion-Res",
                ttp,
            ),
            (
                "U-Net",
                uup,
            ),
        ]:

            err = pred - rrp

            model_cold = (
                pred
                <= ref_p01[None, :]
            )

            model_warm = (
                pred
                >= ref_p99[None, :]
            )

            for tail, ref_event, model_event in [
                (
                    "p01",
                    ref_cold,
                    model_cold,
                ),
                (
                    "p99",
                    ref_warm,
                    model_warm,
                ),
            ]:

                a = acc[
                    key
                ][
                    model_name
                ][
                    tail
                ]

                a[
                    "reference_count"
                ] += int(
                    ref_event.sum()
                )

                a[
                    "model_count"
                ] += int(
                    model_event.sum()
                )

                a[
                    "intersection_count"
                ] += int(
                    (
                        ref_event
                        & model_event
                    ).sum()
                )

                e = err[
                    ref_event
                ]

                a[
                    "sum_error"
                ] += float(
                    e.sum()
                )

                a[
                    "sum_abs_error"
                ] += float(
                    np.abs(e).sum()
                )

                a[
                    "sum_sq_error"
                ] += float(
                    np.square(e).sum()
                )

    print(
        f"year {2015 + iy}"
    )


# ============================================================
# FINALIZE EVENT METRICS
# ============================================================

event_rows = []

for p in periods:

    key = (
        p["kind"],
        p["name"],
    )

    period_pairs = (
        len(p["time_ids"])
        * NC
    )

    for model in [
        "TopoFusion-Res",
        "U-Net",
    ]:

        for tail in [
            "p01",
            "p99",
        ]:

            a = acc[
                key
            ][
                model
            ][
                tail
            ]

            nr = a[
                "reference_count"
            ]

            nm = a[
                "model_count"
            ]

            ni = a[
                "intersection_count"
            ]

            precision = (
                ni / nm
                if nm > 0
                else np.nan
            )

            recall = (
                ni / nr
                if nr > 0
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
                    / (
                        precision
                        + recall
                    )
                )
            else:
                f1 = np.nan

            event_rows.append({
                "kind":
                    p["kind"],

                "period":
                    p["name"],

                "order":
                    p["order"],

                "tail":
                    tail,

                "model":
                    model,

                "n_reference":
                    nr,

                "reference_frequency_pct":
                    100.0
                    * nr
                    / period_pairs,

                "model_frequency_pct":
                    100.0
                    * nm
                    / period_pairs,

                "bias_C":
                    a[
                        "sum_error"
                    ] / nr,

                "mae_C":
                    a[
                        "sum_abs_error"
                    ] / nr,

                "rmse_C":
                    np.sqrt(
                        a[
                            "sum_sq_error"
                        ] / nr
                    ),

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,
            })


# ============================================================
# SAVE CSV
# ============================================================

month_rows = [
    r
    for r in quantile_rows
    if r["kind"] == "month"
]

season_rows = [
    r
    for r in quantile_rows
    if r["kind"] == "season"
]


fieldnames_quant = [
    "kind",
    "period",
    "order",
    "tail",
    "model",
    "n_days",
    "bias_C",
    "mae_C",
    "rmse_C",
    "median_abs_error_C",
    "mean_advantage_topofusion_C",
    "fraction_topofusion_closer",
]


with open(
    OUT_MONTH,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames_quant,
    )

    writer.writeheader()
    writer.writerows(
        month_rows
    )


with open(
    OUT_SEASON,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames_quant,
    )

    writer.writeheader()
    writer.writerows(
        season_rows
    )


with open(
    OUT_EVENT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "kind",
            "period",
            "order",
            "tail",
            "model",
            "n_reference",
            "reference_frequency_pct",
            "model_frequency_pct",
            "bias_C",
            "mae_C",
            "rmse_C",
            "precision",
            "recall",
            "f1",
        ],
    )

    writer.writeheader()
    writer.writerows(
        event_rows
    )


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {
    "analysis":
        "monthly_and_seasonal_p01_p99_sensitivity",

    "source": {
        "scenario":
            "SSP3-7.0",

        "period":
            "2015-2100",

        "calendar":
            "365-day no-leap",

        "days":
            N,

        "years":
            NY,

        "common_cells":
            NC,

        "topofusion_cube":
            str(TOPO_FILE),

        "reference_cube":
            str(RCM_FILE),

        "unet_cube":
            str(UNET_FILE),
    },

    "interpretation_note": (
        "Monthly and seasonal conditioning removes "
        "mixing across the annual cycle, but does not "
        "remove secular warming within 2015-2100."
    ),

    "quantile_fidelity":
        quantile_rows,

    "central_98pct_range":
        range_rows,

    "extreme_day_metrics":
        event_rows,
}


if POOLED_SUMMARY.exists():

    with open(
        POOLED_SUMMARY,
        "r",
    ) as f:

        summary[
            "pooled_2015_2100_reference"
        ] = json.load(f)


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
# CONSOLE REPORT HELPERS
# ============================================================

def get_quant_row(
    kind,
    period,
    tail,
    model,
):
    for r in quantile_rows:
        if (
            r["kind"] == kind
            and r["period"] == period
            and r["tail"] == tail
            and r["model"] == model
        ):
            return r

    raise KeyError(
        (
            kind,
            period,
            tail,
            model,
        )
    )


def get_event_row(
    kind,
    period,
    tail,
    model,
):
    for r in event_rows:
        if (
            r["kind"] == kind
            and r["period"] == period
            and r["tail"] == tail
            and r["model"] == model
        ):
            return r

    raise KeyError(
        (
            kind,
            period,
            tail,
            model,
        )
    )


# ============================================================
# CONSOLE: MONTHS
# ============================================================

print()
print("=" * 100)
print("MONTHLY QUANTILE FIDELITY")
print("=" * 100)

for month in MONTH_NAMES:

    print()
    print(month)

    for tail in [
        "p01",
        "p99",
    ]:

        t = get_quant_row(
            "month",
            month,
            tail,
            "TopoFusion-Res",
        )

        u = get_quant_row(
            "month",
            month,
            tail,
            "U-Net",
        )

        print(
            f"  {tail.upper()}  "
            f"Topo MAE={t['mae_C']:.4f} C  "
            f"U-Net MAE={u['mae_C']:.4f} C  "
            f"adv={t['mean_advantage_topofusion_C']:+.4f} C  "
            f"Topo closer="
            f"{100*t['fraction_topofusion_closer']:.1f}%"
        )


# ============================================================
# CONSOLE: SEASONS
# ============================================================

print()
print("=" * 100)
print("SEASONAL QUANTILE FIDELITY")
print("=" * 100)

for season in SEASON_DEFS:

    print()
    print(
        SEASON_LABELS[
            season
        ]
    )

    for tail in [
        "p01",
        "p99",
    ]:

        t = get_quant_row(
            "season",
            season,
            tail,
            "TopoFusion-Res",
        )

        u = get_quant_row(
            "season",
            season,
            tail,
            "U-Net",
        )

        print(
            f"  {tail.upper()}  "
            f"Topo bias={t['bias_C']:+.4f} C  "
            f"MAE={t['mae_C']:.4f} C  |  "
            f"U-Net bias={u['bias_C']:+.4f} C  "
            f"MAE={u['mae_C']:.4f} C"
        )

        print(
            f"       advantage="
            f"{t['mean_advantage_topofusion_C']:+.4f} C  "
            f"Topo closer="
            f"{100*t['fraction_topofusion_closer']:.2f}%"
        )


# ============================================================
# CONSOLE: SEASONAL EXTREME DAYS
# ============================================================

print()
print("=" * 100)
print("SEASONAL RCM-DEFINED EXTREME DAYS")
print("=" * 100)

for season in SEASON_DEFS:

    print()
    print(
        SEASON_LABELS[
            season
        ]
    )

    for tail in [
        "p01",
        "p99",
    ]:

        print(
            f"  {tail.upper()}"
        )

        for model in [
            "TopoFusion-Res",
            "U-Net",
        ]:

            r = get_event_row(
                "season",
                season,
                tail,
                model,
            )

            print(
                f"    {model:18s} "
                f"bias={r['bias_C']:+.4f} C  "
                f"MAE={r['mae_C']:.4f} C  "
                f"RMSE={r['rmse_C']:.4f} C  "
                f"freq={r['model_frequency_pct']:.3f}%  "
                f"recall={r['recall']:.3f}  "
                f"F1={r['f1']:.3f}"
            )


# ============================================================
# CONSISTENCY / ROBUSTNESS COUNTS
# ============================================================

print()
print("=" * 100)
print("ROBUSTNESS ACROSS CALENDAR PERIODS")
print("=" * 100)

for kind, names in [
    (
        "month",
        MONTH_NAMES,
    ),
    (
        "season",
        list(
            SEASON_DEFS.keys()
        ),
    ),
]:

    for tail in [
        "p01",
        "p99",
    ]:

        wins_mae = 0
        wins_cells = 0
        advantages = []

        for period in names:

            t = get_quant_row(
                kind,
                period,
                tail,
                "TopoFusion-Res",
            )

            u = get_quant_row(
                kind,
                period,
                tail,
                "U-Net",
            )

            if (
                t["mae_C"]
                < u["mae_C"]
            ):
                wins_mae += 1

            if (
                t[
                    "fraction_topofusion_closer"
                ]
                > 0.5
            ):
                wins_cells += 1

            advantages.append(
                t[
                    "mean_advantage_topofusion_C"
                ]
            )

        print(
            f"{kind:6s} {tail.upper()}  "
            f"Topo lower MAE: "
            f"{wins_mae}/{len(names)}  "
            f"Topo closer in majority cells: "
            f"{wins_cells}/{len(names)}  "
            f"advantage range: "
            f"{min(advantages):+.4f} to "
            f"{max(advantages):+.4f} C"
        )


# ============================================================
# MONTHLY WARM-TAIL RECALL
# ============================================================

print()
print("=" * 100)
print("MONTHLY WARM-TAIL THRESHOLD RECALL")
print("=" * 100)

for month in MONTH_NAMES:

    t = get_event_row(
        "month",
        month,
        "p99",
        "TopoFusion-Res",
    )

    u = get_event_row(
        "month",
        month,
        "p99",
        "U-Net",
    )

    print(
        f"{month:3s}  "
        f"Topo freq={t['model_frequency_pct']:.3f}% "
        f"recall={t['recall']:.3f}  |  "
        f"U-Net freq={u['model_frequency_pct']:.3f}% "
        f"recall={u['recall']:.3f}"
    )


print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print(OUT_JSON)
print(OUT_MONTH)
print(OUT_SEASON)
print(OUT_EVENT)

print()
print("DONE")
