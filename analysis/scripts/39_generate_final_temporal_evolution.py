from pathlib import Path
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[2]

CUBE = (
    EXP
    / "diagnostics"
    / "final_e2_vs_unet_advanced_diagnostics"
    / "prediction_cubes"
)

FINAL = EXP / "analysis"

OUT_FIG = FINAL / "figures" / "final_paper"
OUT_DIA = FINAL / "diagnostics"
OUT_TAB = FINAL / "tables"

OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DIA.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


# ============================================================
# METADATA
# ============================================================

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape

START_YEAR = 2015
END_YEAR = 2100

years = np.arange(START_YEAR, END_YEAR + 1)
n_years = len(years)

if shape != (31390, 64, 64):
    raise RuntimeError(f"Unexpected cube shape: {shape}")

if N != n_years * 365:
    raise RuntimeError(
        f"Expected {n_years * 365} days, found {N}"
    )

print("=" * 100)
print("FINAL SEEDED TEMPORAL EVOLUTION")
print("=" * 100)
print("Experiment :", EXP)
print("Cube dir   :", CUBE)
print("Shape      :", shape)
print("Years      :", START_YEAR, "-", END_YEAR)
print("Calendar   : 365 days")


# ============================================================
# FILES
# ============================================================

paths = {
    "RCM": CUBE / "rcm.dat",
    "TopoFusion-Res": CUBE / "e2_full_seeded.dat",
    "U-Net": CUBE / "unet.dat",
    "Mask": CUBE / "mask.dat",
}

for name, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")

print("\nInputs:")
for name, path in paths.items():
    print(f"{name:20s}: {path.name}")


# ============================================================
# MEMMAPS
# ============================================================

rcm = np.memmap(
    paths["RCM"],
    dtype="float32",
    mode="r",
    shape=shape,
)

topo = np.memmap(
    paths["TopoFusion-Res"],
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

mask = np.memmap(
    paths["Mask"],
    dtype="bool",
    mode="r",
    shape=shape,
)


# ============================================================
# DAILY DOMAIN MEANS
# ============================================================

daily_rcm = np.empty(N, dtype=np.float64)
daily_topo = np.empty(N, dtype=np.float64)
daily_unet = np.empty(N, dtype=np.float64)
daily_n = np.empty(N, dtype=np.int32)

chunk = 128

for s in range(0, N, chunk):
    e = min(s + chunk, N)

    m = np.array(
        mask[s:e],
        dtype=bool,
        copy=True,
    )

    y = np.asarray(rcm[s:e], dtype=np.float64)
    p_topo = np.asarray(topo[s:e], dtype=np.float64)
    p_unet = np.asarray(unet[s:e], dtype=np.float64)

    m &= np.isfinite(y)
    m &= np.isfinite(p_topo)
    m &= np.isfinite(p_unet)

    n = m.sum(axis=(1, 2))

    if np.any(n == 0):
        raise RuntimeError(
            f"Found day without common valid cells in [{s}:{e}]"
        )

    daily_n[s:e] = n

    daily_rcm[s:e] = (
        np.where(m, y, 0.0).sum(axis=(1, 2)) / n
    )

    daily_topo[s:e] = (
        np.where(m, p_topo, 0.0).sum(axis=(1, 2)) / n
    )

    daily_unet[s:e] = (
        np.where(m, p_unet, 0.0).sum(axis=(1, 2)) / n
    )

    if e == N or (s // chunk + 1) % 25 == 0:
        print(f"Processed {e:5d}/{N} days")


print("\nCommon-cell count:")
print("min =", int(daily_n.min()))
print("max =", int(daily_n.max()))

if daily_n.min() != 1852 or daily_n.max() != 1852:
    raise RuntimeError(
        "Expected exactly 1852 common cells on every day."
    )


# ============================================================
# YEAR x DAY-OF-YEAR REPRESENTATION
# ============================================================

rcm_yd = daily_rcm.reshape(n_years, 365)
topo_yd = daily_topo.reshape(n_years, 365)
unet_yd = daily_unet.reshape(n_years, 365)


# Annual domain means
annual_rcm = rcm_yd.mean(axis=1)
annual_topo = topo_yd.mean(axis=1)
annual_unet = unet_yd.mean(axis=1)

annual_err_topo = annual_topo - annual_rcm
annual_err_unet = annual_unet - annual_rcm


# Mean 365-day seasonal cycle across all 86 years
seasonal_rcm = rcm_yd.mean(axis=0)
seasonal_topo = topo_yd.mean(axis=0)
seasonal_unet = unet_yd.mean(axis=0)

day_of_year = np.arange(1, 366)


# ============================================================
# SAVE TABLES
# ============================================================

annual_df = pd.DataFrame({
    "year": years,
    "CNRM_ALADIN64_C": annual_rcm,
    "TopoFusion_Res_C": annual_topo,
    "UNet_CMIP_CMIP_C": annual_unet,
    "TopoFusion_minus_ALADIN_C": annual_err_topo,
    "UNet_minus_ALADIN_C": annual_err_unet,
})

annual_csv = OUT_TAB / "39_domain_annual_temperature_ssp370.csv"
annual_df.to_csv(annual_csv, index=False)


seasonal_df = pd.DataFrame({
    "day_of_year": day_of_year,
    "CNRM_ALADIN64_C": seasonal_rcm,
    "TopoFusion_Res_C": seasonal_topo,
    "UNet_CMIP_CMIP_C": seasonal_unet,
})

seasonal_csv = OUT_TAB / "39_domain_mean_seasonal_cycle_ssp370.csv"
seasonal_df.to_csv(seasonal_csv, index=False)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "period": "2015-2100 SSP3-7.0",
    "calendar": "365-day",
    "n_years": int(n_years),
    "n_days": int(N),
    "n_common_cells_per_day": int(daily_n[0]),

    "annual_error": {
        "TopoFusion_Res": {
            "mean_C": float(np.mean(annual_err_topo)),
            "mean_absolute_C": float(np.mean(np.abs(annual_err_topo))),
            "min_C": float(np.min(annual_err_topo)),
            "max_C": float(np.max(annual_err_topo)),
        },
        "UNet_CMIP_CMIP": {
            "mean_C": float(np.mean(annual_err_unet)),
            "mean_absolute_C": float(np.mean(np.abs(annual_err_unet))),
            "min_C": float(np.min(annual_err_unet)),
            "max_C": float(np.max(annual_err_unet)),
        },
    },

    "seasonal_cycle": {
        "TopoFusion_mean_abs_difference_C": float(
            np.mean(np.abs(seasonal_topo - seasonal_rcm))
        ),
        "UNet_mean_abs_difference_C": float(
            np.mean(np.abs(seasonal_unet - seasonal_rcm))
        ),
    },
}

summary_path = (
    OUT_DIA
    / "39_final_temporal_evolution_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# FIGURE — FINAL PUBLICATION VERSION
# ============================================================

# Consistent model colors across all panels
C_RCM = "tab:blue"
C_TOPO = "tab:orange"
C_UNET = "tab:green"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig, axes = plt.subplots(
    3,
    1,
    figsize=(11.2, 10.5),
)

fig.suptitle(
    "Temporal evolution and seasonal cycle under SSP3-7.0",
    fontsize=14,
    y=0.985,
)


# ------------------------------------------------------------
# (a) Domain-mean annual temperature
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    years,
    annual_rcm,
    color=C_RCM,
    linewidth=2.0,
    label="CNRM-ALADIN64",
)

ax.plot(
    years,
    annual_topo,
    color=C_TOPO,
    linewidth=1.8,
    label="TopoFusion-Res",
)

ax.plot(
    years,
    annual_unet,
    color=C_UNET,
    linewidth=1.8,
    label="U-Net CMIP-CMIP",
)

ax.set_title(
    "(a) Domain-mean annual temperature"
)

ax.set_ylabel(
    r"Temperature ($^\circ$C)"
)

ax.set_xlim(
    START_YEAR,
    END_YEAR,
)

ax.grid(
    alpha=0.22,
    linewidth=0.7,
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper left",
)


# ------------------------------------------------------------
# (b) Domain-mean annual temperature error relative to CNRM-ALADIN64
# ------------------------------------------------------------

ax = axes[1]

ax.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.25,
    alpha=0.85,
    zorder=1,
)

ax.plot(
    years,
    annual_err_topo,
    color=C_TOPO,
    linewidth=1.8,
    label="TopoFusion-Res $-$ CNRM-ALADIN64",
    zorder=2,
)

ax.plot(
    years,
    annual_err_unet,
    color=C_UNET,
    linewidth=1.8,
    label="U-Net CMIP-CMIP $-$ CNRM-ALADIN64",
    zorder=2,
)

ax.set_title(
    "(b) Domain-mean annual temperature error relative to CNRM-ALADIN64"
)

ax.set_ylabel(
    r"Error ($^\circ$C)"
)

ax.set_xlim(
    START_YEAR,
    END_YEAR,
)

ax.grid(
    alpha=0.22,
    linewidth=0.7,
)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper left",
)


# ------------------------------------------------------------
# (c) Mean seasonal cycle
# ------------------------------------------------------------

ax = axes[2]

ax.plot(
    day_of_year,
    seasonal_rcm,
    color=C_RCM,
    linewidth=2.0,
    label="CNRM-ALADIN64",
)

ax.plot(
    day_of_year,
    seasonal_topo,
    color=C_TOPO,
    linewidth=1.8,
    label="TopoFusion-Res",
)

ax.plot(
    day_of_year,
    seasonal_unet,
    color=C_UNET,
    linewidth=1.8,
    label="U-Net CMIP-CMIP",
)

ax.set_title(
    "(c) Mean day-of-year climatology over 2015–2100"
)

ax.set_xlabel(
    "Day of year"
)

ax.set_ylabel(
    r"Temperature ($^\circ$C)"
)

ax.set_xlim(
    1,
    365,
)

ax.grid(
    alpha=0.22,
    linewidth=0.7,
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
)


fig.tight_layout(
    rect=[0, 0, 1, 0.965]
)


png = OUT_FIG / "05_temporal_evolution_ssp370_publication.png"
pdf = OUT_FIG / "05_temporal_evolution_ssp370_publication.pdf"

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

plt.close(fig)


print("\nSaved publication figure:")
print(png)
print(pdf)

print("\nSummary:")
print(json.dumps(summary, indent=2))
