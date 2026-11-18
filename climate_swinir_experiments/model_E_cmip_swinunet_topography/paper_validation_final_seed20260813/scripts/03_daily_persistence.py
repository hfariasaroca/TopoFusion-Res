from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"

CUBE = EXP / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
DATA = EXP / "data/torch_dataset_model_E_physical_v1"

PAPER = EXP / "paper_validation_final_seed20260813"
FIG = PAPER / "figures/persistence"
TAB = PAPER / "tables"
DIA = PAPER / "diagnostics"

for p in [FIG, TAB, DIA]:
    p.mkdir(parents=True, exist_ok=True)

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape

rcm = np.memmap(CUBE/"rcm.dat", dtype="float32", mode="r", shape=shape)
e2 = np.memmap(CUBE/"e2_full_seeded.dat", dtype="float32", mode="r", shape=shape)
unet = np.memmap(CUBE/"unet.dat", dtype="float32", mode="r", shape=shape)
mask = np.memmap(CUBE/"mask.dat", dtype="bool", mode="r", shape=shape)

time = np.load(DATA/"test_time_ssp370.npy")
dt = pd.DatetimeIndex(time)

if N % 365 != 0:
    raise RuntimeError("Se esperaba calendario de 365 días.")

NY = N // 365

common = np.all(mask, axis=0)

ds = xr.open_dataset(ROOT/"data_tomas/unet_outputs/tas_cmip_cmip_output.nc")
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

print("="*90)
print("DAILY PERSISTENCE DIAGNOSTICS")
print("="*90)
print("Shape:", shape)
print("Years:", NY)
print("Common valid cells:", int(common.sum()))

# -------------------------------------------------------------------
# Daily climatologies: own climatology for each dataset
# -------------------------------------------------------------------

sum_r = np.zeros((365,H,W), dtype=np.float64)
sum_e = np.zeros_like(sum_r)
sum_u = np.zeros_like(sum_r)
cnt = np.zeros((365,H,W), dtype=np.float64)

for iy in range(NY):
    s = iy*365
    ee = s+365

    m = np.asarray(mask[s:ee], dtype=bool)

    rr = np.asarray(rcm[s:ee], dtype=np.float64)
    e2b = np.asarray(e2[s:ee], dtype=np.float64)
    ub = np.asarray(unet[s:ee], dtype=np.float64)

    sum_r += np.where(m, rr, 0.0)
    sum_e += np.where(m, e2b, 0.0)
    sum_u += np.where(m, ub, 0.0)
    cnt += m

clim_r = np.divide(sum_r, cnt, out=np.full_like(sum_r,np.nan), where=cnt>0)
clim_e = np.divide(sum_e, cnt, out=np.full_like(sum_e,np.nan), where=cnt>0)
clim_u = np.divide(sum_u, cnt, out=np.full_like(sum_u,np.nan), where=cnt>0)

# -------------------------------------------------------------------
# Lag-1 correlation accumulators
# -------------------------------------------------------------------

def lag1_map(arr, clim):
    sx = np.zeros((H,W), dtype=np.float64)
    sy = np.zeros((H,W), dtype=np.float64)
    sxx = np.zeros((H,W), dtype=np.float64)
    syy = np.zeros((H,W), dtype=np.float64)
    sxy = np.zeros((H,W), dtype=np.float64)
    n = np.zeros((H,W), dtype=np.float64)

    prev = None
    prev_m = None

    for iy in range(NY):
        s = iy*365
        ee = s+365

        a = np.asarray(arr[s:ee], dtype=np.float64) - clim
        m = np.asarray(mask[s:ee], dtype=bool)

        # within each year
        x = a[:-1]
        y = a[1:]
        vm = m[:-1] & m[1:] & np.isfinite(x) & np.isfinite(y)

        xv = np.where(vm, x, 0.0)
        yv = np.where(vm, y, 0.0)
        w = vm.astype(np.float64)

        n += w.sum(axis=0)
        sx += xv.sum(axis=0)
        sy += yv.sum(axis=0)
        sxx += (xv*xv).sum(axis=0)
        syy += (yv*yv).sum(axis=0)
        sxy += (xv*yv).sum(axis=0)

        # bridge Dec31 -> Jan1
        if prev is not None:
            x0 = prev
            y0 = a[0]
            vm0 = prev_m & m[0] & np.isfinite(x0) & np.isfinite(y0)

            xv0 = np.where(vm0, x0, 0.0)
            yv0 = np.where(vm0, y0, 0.0)
            w0 = vm0.astype(np.float64)

            n += w0
            sx += xv0
            sy += yv0
            sxx += xv0*xv0
            syy += yv0*yv0
            sxy += xv0*yv0

        prev = a[-1]
        prev_m = m[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx*sy/n
        vx = sxx - sx*sx/n
        vy = syy - sy*sy/n
        corr = cov / np.sqrt(vx*vy)

    corr[(n < 100) | ~common] = np.nan
    return corr


print("[1] Lag-1 anomaly autocorrelation")

lag_r = lag1_map(rcm, clim_r)
lag_e = lag1_map(e2, clim_e)
lag_u = lag1_map(unet, clim_u)

err_e = np.abs(lag_e - lag_r)
err_u = np.abs(lag_u - lag_r)
improvement = err_u - err_e

# -------------------------------------------------------------------
# Common thresholds from RCM anomaly distribution
# -------------------------------------------------------------------

print("[2] RCM p10/p90 daily-anomaly thresholds")

anom_r = np.empty((N,H,W), dtype=np.float32)

for iy in range(NY):
    s = iy*365
    ee = s+365
    anom_r[s:ee] = (
        np.asarray(rcm[s:ee], dtype=np.float32)
        - clim_r.astype(np.float32)
    )

q10 = np.nanpercentile(anom_r, 10, axis=0)
q90 = np.nanpercentile(anom_r, 90, axis=0)

del anom_r

# -------------------------------------------------------------------
# Run statistics
# -------------------------------------------------------------------

def run_stats_for_cell(series, valid, threshold, above=True):
    x = np.asarray(series)
    v = np.asarray(valid)

    event = (x > threshold) if above else (x < threshold)
    event &= v & np.isfinite(x)

    runs = []
    current = 0

    for flag in event:
        if flag:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0

    if current > 0:
        runs.append(current)

    if not runs:
        return np.nan, np.nan, np.nan, 0

    runs = np.asarray(runs, dtype=float)

    return (
        float(np.mean(runs)),
        float(np.percentile(runs,90)),
        float(np.max(runs)),
        int(len(runs)),
    )


metrics = {}

for name in ["rcm","e2","unet"]:
    metrics[name] = {
        "warm_mean": np.full((H,W),np.nan,dtype=np.float32),
        "warm_p90": np.full((H,W),np.nan,dtype=np.float32),
        "warm_max": np.full((H,W),np.nan,dtype=np.float32),
        "warm_n": np.full((H,W),np.nan,dtype=np.float32),
        "cold_mean": np.full((H,W),np.nan,dtype=np.float32),
        "cold_p90": np.full((H,W),np.nan,dtype=np.float32),
        "cold_max": np.full((H,W),np.nan,dtype=np.float32),
        "cold_n": np.full((H,W),np.nan,dtype=np.float32),
    }

arrays = {"rcm":rcm, "e2":e2, "unet":unet}
clims = {"rcm":clim_r, "e2":clim_e, "unet":clim_u}

cells = np.argwhere(common)

print("[3] Event-duration statistics")

for k,(iy,ix) in enumerate(cells,1):
    valid = np.asarray(mask[:,iy,ix], dtype=bool)

    for name,arr in arrays.items():
        raw = np.asarray(arr[:,iy,ix], dtype=np.float64)

        # own daily climatology repeated over all years
        clim_series = np.tile(clims[name][:,iy,ix], NY)
        anom = raw - clim_series

        wm, wp, wx, wn = run_stats_for_cell(
            anom, valid, q90[iy,ix], above=True
        )
        cm, cp, cx, cn = run_stats_for_cell(
            anom, valid, q10[iy,ix], above=False
        )

        metrics[name]["warm_mean"][iy,ix] = wm
        metrics[name]["warm_p90"][iy,ix] = wp
        metrics[name]["warm_max"][iy,ix] = wx
        metrics[name]["warm_n"][iy,ix] = wn

        metrics[name]["cold_mean"][iy,ix] = cm
        metrics[name]["cold_p90"][iy,ix] = cp
        metrics[name]["cold_max"][iy,ix] = cx
        metrics[name]["cold_n"][iy,ix] = cn

    if k % 100 == 0 or k == len(cells):
        print(f"  cell {k}/{len(cells)}")

# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------

def panel(ax,z,title,cmap="viridis",vmin=None,vmax=None,label=""):
    zz=np.asarray(z,dtype=float).copy()
    zz[~common]=np.nan

    im=ax.pcolormesh(
        lon,lat,zz,shading="auto",
        cmap=cmap,vmin=vmin,vmax=vmax
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb=plt.colorbar(im,ax=ax,shrink=.82)
    cb.set_label(label)


# Lag1 figure
vals = np.concatenate([
    lag_r[common], lag_e[common], lag_u[common]
])
lo = float(np.nanpercentile(vals,2))
hi = float(np.nanpercentile(vals,98))

diff = lag_e-lag_u
lim = float(np.nanpercentile(np.abs(diff[common]),98))

fig,ax=plt.subplots(1,4,figsize=(20,5))

panel(ax[0],lag_r,"ALADIN/RCM","viridis",lo,hi,"lag-1 correlation")
panel(ax[1],lag_e,"E2 v2","viridis",lo,hi,"lag-1 correlation")
panel(ax[2],lag_u,"U-Net CMIP-CMIP","viridis",lo,hi,"lag-1 correlation")
panel(
    ax[3],improvement,
    "Improvement in |error|\npositive = E2 closer to RCM",
    "RdBu_r",
    -np.nanpercentile(np.abs(improvement[common]),98),
    np.nanpercentile(np.abs(improvement[common]),98),
    "Δ absolute error"
)

fig.suptitle("Lag-1 autocorrelation of daily temperature anomalies")
fig.tight_layout()
fig.savefig(FIG/"03_lag1_daily_anomaly_persistence.png",dpi=180,bbox_inches="tight")
plt.close(fig)

# Warm/cold mean duration
for event in ["warm","cold"]:
    r = metrics["rcm"][f"{event}_mean"]
    e = metrics["e2"][f"{event}_mean"]
    u = metrics["unet"][f"{event}_mean"]

    vals=np.concatenate([r[common],e[common],u[common]])
    vmax=float(np.nanpercentile(vals,98))

    fig,ax=plt.subplots(1,3,figsize=(16,5))

    panel(ax[0],r,"ALADIN/RCM","viridis",0,vmax,"mean run length (days)")
    panel(ax[1],e,"E2 v2","viridis",0,vmax,"mean run length (days)")
    panel(ax[2],u,"U-Net CMIP-CMIP","viridis",0,vmax,"mean run length (days)")

    fig.suptitle(
        f"Mean duration of {event} anomaly events — common RCM threshold"
    )
    fig.tight_layout()
    fig.savefig(
        FIG/f"03_{event}_event_mean_duration.png",
        dpi=180,bbox_inches="tight"
    )
    plt.close(fig)

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

def stats(z):
    v=np.asarray(z)[common]
    v=v[np.isfinite(v)]
    return {
        "mean":float(np.mean(v)),
        "median":float(np.median(v)),
        "p10":float(np.percentile(v,10)),
        "p90":float(np.percentile(v,90)),
    }

rows=[]

for model,z in [
    ("ALADIN_RCM",lag_r),
    ("E2_v2",lag_e),
    ("UNet_CMIP_CMIP",lag_u),
]:
    rows.append({
        "diagnostic":"lag1_daily_anomaly_autocorrelation",
        "model":model,
        **stats(z)
    })

for event in ["warm","cold"]:
    for metric in ["mean","p90","max"]:
        for model_key,model_name in [
            ("rcm","ALADIN_RCM"),
            ("e2","E2_v2"),
            ("unet","UNet_CMIP_CMIP"),
        ]:
            rows.append({
                "diagnostic":f"{event}_event_{metric}_duration_days",
                "model":model_name,
                **stats(metrics[model_key][f"{event}_{metric}"])
            })

df=pd.DataFrame(rows)
df.to_csv(TAB/"03_daily_persistence_summary.csv",index=False)

np.savez_compressed(
    DIA/"03_daily_persistence_maps.npz",
    lag1_rcm=lag_r,
    lag1_e2=lag_e,
    lag1_unet=lag_u,
    lag1_abs_error_improvement=improvement,
    q10_rcm_anomaly=q10,
    q90_rcm_anomaly=q90,
    **{
        f"{model}_{name}":arr
        for model,d in metrics.items()
        for name,arr in d.items()
    }
)

with open(DIA/"03_daily_persistence_method.json","w") as f:
    json.dump({
        "anomaly_definition":
            "Each dataset's own 365-day daily climatology removed.",
        "lag1_definition":
            "Pearson lag-1 autocorrelation of daily anomalies.",
        "event_threshold_definition":
            "Cell-wise p90/p10 of ALADIN/RCM daily anomalies over SSP3-7.0 2015-2100.",
        "threshold_application":
            "Same ALADIN/RCM threshold applied to RCM, E2 v2 and U-Net.",
        "warm_event":
            "Consecutive days with daily anomaly > RCM p90 threshold.",
        "cold_event":
            "Consecutive days with daily anomaly < RCM p10 threshold.",
        "note":
            "Persistence diagnostic; not presented as a formal ETCCDI heatwave/cold-spell index."
    },f,indent=2)

print("\nDONE")
print("Summary:",TAB/"03_daily_persistence_summary.csv")
