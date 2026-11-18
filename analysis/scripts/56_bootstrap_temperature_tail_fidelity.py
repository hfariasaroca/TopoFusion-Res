from pathlib import Path
import csv
import json
import math

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

POOLED_54 = (
    PAPER
    / "diagnostics"
    / "54_final_seeded_temperature_tail_summary.json"
)

SEASONAL_55 = (
    PAPER
    / "diagnostics"
    / "55_monthly_seasonal_temperature_tail_summary.json"
)

OUT_DIA = PAPER / "diagnostics"
OUT_TAB = PAPER / "tables"

OUT_DIA.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

OUT_UNIT = (
    OUT_TAB
    / "56_tail_fidelity_by_year_or_season.csv"
)

OUT_BOOT = (
    OUT_TAB
    / "56_tail_fidelity_block_bootstrap_summary.csv"
)

OUT_DIST = (
    OUT_DIA
    / "56_tail_fidelity_block_bootstrap_distributions.npz"
)

OUT_JSON = (
    OUT_DIA
    / "56_tail_fidelity_block_bootstrap_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

N_BOOT = 2000
BLOCK_SIZES = [3, 5, 10]
SEED = 20260813

START_YEAR = 2015
END_YEAR = 2100

TAILS = [
    "p01",
    "p99",
]

METRICS = [
    "mae",
    "rmse",
]


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
print("FINAL TEMPERATURE-TAIL MOVING-BLOCK BOOTSTRAP")
print("=" * 100)

print("Shape             :", shape)
print("Years             :", NY)
print("Common cells      :", NC)
print("Bootstrap reps    :", N_BOOT)
print("Block lengths     :", BLOCK_SIZES)
print("Seed              :", SEED)
print()

print("IMPORTANT:")
print(
    "Bootstrap is applied to year-/season-resolved "
    "paired tail-fidelity metrics."
)
print(
    "It measures temporal robustness of the "
    "TopoFusion-vs-U-Net advantage."
)
print(
    "The pooled 2015-2100 p01/p99 metrics remain "
    "the descriptive full-period estimates."
)


# ============================================================
# CALENDAR
# ============================================================

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

month_start = {}
month_end = {}

cursor = 0

for month, nd in enumerate(
    MONTH_LENGTHS,
    start=1,
):
    month_start[month] = cursor
    month_end[month] = cursor + int(nd)
    cursor += int(nd)


# JJA = June 1 through August 31
JJA_START = month_start[6]
JJA_END = month_end[8]

# DJF will be:
# Dec(Y-1) + Jan-Feb(Y)
DEC_START = month_start[12]
DEC_END = month_end[12]

JANFEB_START = month_start[1]
JANFEB_END = month_end[2]


# ============================================================
# FLATTENED VIEWS
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def quantile_pair(x):
    q = np.percentile(
        x,
        [1.0, 99.0],
        axis=0,
    )

    return {
        "p01": q[0],
        "p99": q[1],
    }


def tail_metrics(
    reference_q,
    model_q,
):
    e = (
        np.asarray(
            model_q,
            dtype=np.float64,
        )
        -
        np.asarray(
            reference_q,
            dtype=np.float64,
        )
    )

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


def analyze_unit(
    scope,
    label,
    order,
    time_ids,
):
    rr = np.asarray(
        rcm_flat[
            time_ids,
            :
        ][
            :,
            cell_idx,
        ],
        dtype=np.float64,
    )

    tt = np.asarray(
        topo_flat[
            time_ids,
            :
        ][
            :,
            cell_idx,
        ],
        dtype=np.float64,
    )

    uu = np.asarray(
        unet_flat[
            time_ids,
            :
        ][
            :,
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
            f"Non-finite values in "
            f"{scope} {label}"
        )

    qr = quantile_pair(rr)
    qt = quantile_pair(tt)
    qu = quantile_pair(uu)

    rows = []

    for tail in TAILS:

        r = qr[tail]
        t = qt[tail]
        u = qu[tail]

        mt = tail_metrics(
            r,
            t,
        )

        mu = tail_metrics(
            r,
            u,
        )

        abs_t = np.abs(
            t - r
        )

        abs_u = np.abs(
            u - r
        )

        rows.append({
            "scope":
                scope,

            "label":
                str(label),

            "order":
                int(order),

            "tail":
                tail,

            "n_days":
                int(
                    len(time_ids)
                ),

            "topofusion_bias_C":
                mt["bias"],

            "unet_bias_C":
                mu["bias"],

            "topofusion_mae_C":
                mt["mae"],

            "unet_mae_C":
                mu["mae"],

            "mae_advantage_topofusion_C":
                (
                    mu["mae"]
                    - mt["mae"]
                ),

            "topofusion_rmse_C":
                mt["rmse"],

            "unet_rmse_C":
                mu["rmse"],

            "rmse_advantage_topofusion_C":
                (
                    mu["rmse"]
                    - mt["rmse"]
                ),

            "fraction_topofusion_closer":
                float(
                    np.mean(
                        abs_t
                        < abs_u
                    )
                ),
        })

    return rows


# ============================================================
# BUILD YEAR-/SEASON-RESOLVED METRICS
# ============================================================

unit_rows = []


# ------------------------------------------------------------
# Annual: 86 calendar years
# ------------------------------------------------------------

print()
print("=" * 100)
print("ANNUAL TAIL FIDELITY")
print("=" * 100)

for iy in range(NY):

    year = START_YEAR + iy

    t0 = iy * 365
    t1 = t0 + 365

    time_ids = np.arange(
        t0,
        t1,
        dtype=np.int64,
    )

    rows = analyze_unit(
        scope="annual",
        label=year,
        order=iy,
        time_ids=time_ids,
    )

    unit_rows.extend(rows)

    print(
        f"annual {year}"
    )


# ------------------------------------------------------------
# JJA: 86 complete winters
# ------------------------------------------------------------

print()
print("=" * 100)
print("JJA TAIL FIDELITY")
print("=" * 100)

for iy in range(NY):

    year = START_YEAR + iy

    base = iy * 365

    time_ids = np.arange(
        base + JJA_START,
        base + JJA_END,
        dtype=np.int64,
    )

    rows = analyze_unit(
        scope="JJA",
        label=year,
        order=iy,
        time_ids=time_ids,
    )

    unit_rows.extend(rows)

    print(
        f"JJA {year}"
    )


# ------------------------------------------------------------
# DJF: 85 complete austral summers
#
# DJF 2016 =
# Dec 2015 + Jan-Feb 2016
#
# ...
#
# DJF 2100 =
# Dec 2099 + Jan-Feb 2100
# ------------------------------------------------------------

print()
print("=" * 100)
print("DJF TAIL FIDELITY")
print("=" * 100)

djf_order = 0

for iy_current in range(
    1,
    NY,
):

    season_year = (
        START_YEAR
        + iy_current
    )

    prev_base = (
        iy_current - 1
    ) * 365

    curr_base = (
        iy_current
    ) * 365

    dec_ids = np.arange(
        prev_base + DEC_START,
        prev_base + DEC_END,
        dtype=np.int64,
    )

    janfeb_ids = np.arange(
        curr_base + JANFEB_START,
        curr_base + JANFEB_END,
        dtype=np.int64,
    )

    time_ids = np.concatenate(
        [
            dec_ids,
            janfeb_ids,
        ]
    )

    if len(time_ids) != 90:
        raise RuntimeError(
            f"DJF {season_year} "
            f"expected 90 days, "
            f"got {len(time_ids)}"
        )

    rows = analyze_unit(
        scope="DJF",
        label=season_year,
        order=djf_order,
        time_ids=time_ids,
    )

    unit_rows.extend(rows)

    print(
        f"DJF {season_year}"
    )

    djf_order += 1


# ============================================================
# SAVE UNIT METRICS
# ============================================================

unit_fields = [
    "scope",
    "label",
    "order",
    "tail",
    "n_days",
    "topofusion_bias_C",
    "unet_bias_C",
    "topofusion_mae_C",
    "unet_mae_C",
    "mae_advantage_topofusion_C",
    "topofusion_rmse_C",
    "unet_rmse_C",
    "rmse_advantage_topofusion_C",
    "fraction_topofusion_closer",
]


with open(
    OUT_UNIT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=unit_fields,
    )

    writer.writeheader()
    writer.writerows(
        unit_rows
    )


# ============================================================
# HELPER TO EXTRACT TEMPORAL SERIES
# ============================================================

def get_series(
    scope,
    tail,
    field,
):
    rows = [
        r
        for r in unit_rows
        if (
            r["scope"] == scope
            and r["tail"] == tail
        )
    ]

    rows = sorted(
        rows,
        key=lambda r: r["order"],
    )

    return np.asarray(
        [
            r[field]
            for r in rows
        ],
        dtype=np.float64,
    )


# ============================================================
# CIRCULAR MOVING-BLOCK BOOTSTRAP
# ============================================================

def make_bootstrap_indices(
    n,
    block_size,
    n_boot,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    n_blocks = math.ceil(
        n / block_size
    )

    out = np.empty(
        (
            n_boot,
            n,
        ),
        dtype=np.int16,
    )

    offsets = np.arange(
        block_size,
        dtype=np.int64,
    )

    for ib in range(n_boot):

        starts = rng.integers(
            0,
            n,
            size=n_blocks,
        )

        idx = (
            starts[:, None]
            + offsets[None, :]
        ) % n

        idx = idx.ravel()[:n]

        out[
            ib,
            :
        ] = idx

    return out


bootstrap_indices = {}

for scope, n_units in [
    ("annual", 86),
    ("DJF", 85),
    ("JJA", 86),
]:

    for block_size in BLOCK_SIZES:

        derived_seed = (
            SEED
            + 10000
            * n_units
            + 100
            * block_size
        )

        bootstrap_indices[
            (
                scope,
                block_size,
            )
        ] = make_bootstrap_indices(
            n=n_units,
            block_size=block_size,
            n_boot=N_BOOT,
            seed=derived_seed,
        )


# ============================================================
# BOOTSTRAP SUMMARIES
# ============================================================

boot_rows = []
boot_arrays = {}


for scope in [
    "annual",
    "DJF",
    "JJA",
]:

    for tail in TAILS:

        for metric in METRICS:

            field = (
                f"{metric}_advantage_"
                f"topofusion_C"
            )

            series = get_series(
                scope,
                tail,
                field,
            )

            n_units = len(series)

            observed = float(
                np.mean(series)
            )

            median_unit = float(
                np.median(series)
            )

            fraction_units_positive = float(
                np.mean(
                    series > 0
                )
            )

            for block_size in BLOCK_SIZES:

                idx = bootstrap_indices[
                    (
                        scope,
                        block_size,
                    )
                ]

                dist = np.mean(
                    series[
                        idx
                    ],
                    axis=1,
                )

                ci_low = float(
                    np.percentile(
                        dist,
                        2.5,
                    )
                )

                ci_high = float(
                    np.percentile(
                        dist,
                        97.5,
                    )
                )

                prob_positive = float(
                    np.mean(
                        dist > 0
                    )
                )

                robust_topo = bool(
                    ci_low > 0
                )

                robust_unet = bool(
                    ci_high < 0
                )

                if robust_topo:
                    direction = (
                        "TopoFusion-Res"
                    )

                elif robust_unet:
                    direction = (
                        "U-Net"
                    )

                else:
                    direction = (
                        "interval_includes_zero"
                    )

                boot_rows.append({
                    "scope":
                        scope,

                    "tail":
                        tail,

                    "metric":
                        metric,

                    "n_units":
                        n_units,

                    "block_size":
                        block_size,

                    "n_boot":
                        N_BOOT,

                    "observed_mean_advantage_topofusion_C":
                        observed,

                    "observed_median_unit_advantage_C":
                        median_unit,

                    "fraction_units_topofusion_better":
                        fraction_units_positive,

                    "bootstrap_mean_C":
                        float(
                            np.mean(dist)
                        ),

                    "ci95_low_C":
                        ci_low,

                    "ci95_high_C":
                        ci_high,

                    "bootstrap_fraction_positive":
                        prob_positive,

                    "robust_direction":
                        direction,
                })

                key = (
                    f"{scope}_{tail}_"
                    f"{metric}_"
                    f"block{block_size}"
                )

                boot_arrays[key] = (
                    dist.astype(
                        np.float32
                    )
                )


# ============================================================
# SAVE BOOTSTRAP SUMMARY
# ============================================================

boot_fields = [
    "scope",
    "tail",
    "metric",
    "n_units",
    "block_size",
    "n_boot",
    "observed_mean_advantage_topofusion_C",
    "observed_median_unit_advantage_C",
    "fraction_units_topofusion_better",
    "bootstrap_mean_C",
    "ci95_low_C",
    "ci95_high_C",
    "bootstrap_fraction_positive",
    "robust_direction",
]


with open(
    OUT_BOOT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=boot_fields,
    )

    writer.writeheader()
    writer.writerows(
        boot_rows
    )


np.savez_compressed(
    OUT_DIST,
    **boot_arrays,
)


# ============================================================
# POOLED REFERENCE VALUES FROM PREVIOUS ANALYSES
# ============================================================

pooled_reference = {}


if POOLED_54.exists():

    with open(
        POOLED_54,
        "r",
    ) as f:

        pooled54 = json.load(f)

    pooled_reference[
        "annual_pooled_2015_2100"
    ] = {
        "p01":
            pooled54.get(
                "p01"
            ),

        "p99":
            pooled54.get(
                "p99"
            ),
    }


if SEASONAL_55.exists():

    with open(
        SEASONAL_55,
        "r",
    ) as f:

        pooled55 = json.load(f)

    selected = []

    for row in pooled55.get(
        "quantile_fidelity",
        [],
    ):

        if (
            row.get("kind")
            == "season"
            and row.get("period")
            in ("DJF", "JJA")
        ):
            selected.append(
                row
            )

    pooled_reference[
        "seasonal_pooled_2015_2100"
    ] = selected


# ============================================================
# METADATA JSON
# ============================================================

metadata = {
    "analysis":
        (
            "paired circular moving-block bootstrap "
            "of year-/season-resolved temperature-tail "
            "fidelity"
        ),

    "scenario":
        "SSP3-7.0",

    "period":
        "2015-2100",

    "common_cells":
        NC,

    "tails":
        TAILS,

    "metrics":
        METRICS,

    "n_boot":
        N_BOOT,

    "block_sizes":
        BLOCK_SIZES,

    "seed":
        SEED,

    "units": {
        "annual":
            86,

        "JJA":
            86,

        "DJF":
            85,
    },

    "DJF_definition":
        (
            "December of previous calendar year "
            "plus January-February of labelled year; "
            "complete seasons 2016-2100."
        ),

    "advantage_definition":
        (
            "U-Net tail error minus "
            "TopoFusion-Res tail error. "
            "Positive values favour TopoFusion-Res."
        ),

    "interpretation":
        (
            "This bootstrap evaluates temporal robustness "
            "of year-/season-resolved tail-fidelity "
            "differences. It is distinct from the pooled "
            "2015-2100 percentile estimates."
        ),

    "pooled_reference":
        pooled_reference,

    "bootstrap_summary":
        boot_rows,
}


with open(
    OUT_JSON,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# REPORT HELPERS
# ============================================================

def selected_row(
    scope,
    tail,
    metric,
    block_size,
):
    for r in boot_rows:

        if (
            r["scope"] == scope
            and r["tail"] == tail
            and r["metric"] == metric
            and r["block_size"] == block_size
        ):
            return r

    raise KeyError(
        (
            scope,
            tail,
            metric,
            block_size,
        )
    )


# ============================================================
# CONSOLE — YEAR/SEASON ROBUSTNESS
# ============================================================

print()
print("=" * 100)
print("OBSERVED YEAR-/SEASON-RESOLVED ADVANTAGES")
print("Positive = lower error for TopoFusion-Res")
print("=" * 100)


for scope in [
    "annual",
    "DJF",
    "JJA",
]:

    print()
    print(scope.upper())

    for tail in TAILS:

        mae_series = get_series(
            scope,
            tail,
            "mae_advantage_topofusion_C",
        )

        rmse_series = get_series(
            scope,
            tail,
            "rmse_advantage_topofusion_C",
        )

        print(
            f"  {tail.upper()}  "
            f"MAE advantage="
            f"{np.mean(mae_series):+.4f} C  "
            f"Topo better in "
            f"{100*np.mean(mae_series > 0):.1f}% "
            f"of units"
        )

        print(
            f"       RMSE advantage="
            f"{np.mean(rmse_series):+.4f} C  "
            f"Topo better in "
            f"{100*np.mean(rmse_series > 0):.1f}% "
            f"of units"
        )


# ============================================================
# CONSOLE — MAIN 5-YEAR BOOTSTRAP
# ============================================================

print()
print("=" * 100)
print("MAIN BOOTSTRAP — 5-YEAR/SEASON BLOCKS")
print("=" * 100)


for scope in [
    "annual",
    "DJF",
    "JJA",
]:

    print()
    print(scope.upper())

    for tail in TAILS:

        for metric in METRICS:

            r = selected_row(
                scope,
                tail,
                metric,
                5,
            )

            print(
                f"  {tail.upper()} {metric.upper():4s}  "
                f"adv={r['observed_mean_advantage_topofusion_C']:+.4f} C  "
                f"CI95=[{r['ci95_low_C']:+.4f}, "
                f"{r['ci95_high_C']:+.4f}] C  "
                f"direction={r['robust_direction']}"
            )


# ============================================================
# CONSOLE — BLOCK-LENGTH SENSITIVITY
# ============================================================

print()
print("=" * 100)
print("BLOCK-LENGTH SENSITIVITY — MAE")
print("=" * 100)


for scope in [
    "annual",
    "DJF",
    "JJA",
]:

    print()
    print(scope.upper())

    for tail in TAILS:

        print(
            f"  {tail.upper()}"
        )

        for block_size in BLOCK_SIZES:

            r = selected_row(
                scope,
                tail,
                "mae",
                block_size,
            )

            print(
                f"    block={block_size:2d}  "
                f"CI95=[{r['ci95_low_C']:+.4f}, "
                f"{r['ci95_high_C']:+.4f}] C  "
                f"positive="
                f"{100*r['bootstrap_fraction_positive']:.1f}%  "
                f"{r['robust_direction']}"
            )


# ============================================================
# CONSOLE — KEY DECISION
# ============================================================

print()
print("=" * 100)
print("KEY DECISION CHECK")
print("=" * 100)

checks = [
    (
        "Annual p01",
        "annual",
        "p01",
    ),
    (
        "Annual p99",
        "annual",
        "p99",
    ),
    (
        "DJF p01",
        "DJF",
        "p01",
    ),
    (
        "DJF p99",
        "DJF",
        "p99",
    ),
    (
        "JJA p01",
        "JJA",
        "p01",
    ),
    (
        "JJA p99",
        "JJA",
        "p99",
    ),
]


for label, scope, tail in checks:

    directions = []

    for block_size in BLOCK_SIZES:

        r = selected_row(
            scope,
            tail,
            "mae",
            block_size,
        )

        directions.append(
            r["robust_direction"]
        )

    if all(
        d == "TopoFusion-Res"
        for d in directions
    ):
        result = (
            "TOPO ROBUST FOR ALL BLOCK LENGTHS"
        )

    elif all(
        d == "U-Net"
        for d in directions
    ):
        result = (
            "U-NET ROBUST FOR ALL BLOCK LENGTHS"
        )

    else:
        result = (
            "MIXED / NOT ROBUST ACROSS ALL BLOCK LENGTHS"
        )

    print(
        f"{label:12s}: {result}"
    )


# ============================================================
# OUTPUTS
# ============================================================

print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print(OUT_UNIT)
print(OUT_BOOT)
print(OUT_DIST)
print(OUT_JSON)

print()
print("DONE")
