from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path("/workspace/SwinIR_HF/Normalizacion")
EXP = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"

CUBE = EXP / "diagnostics/final_e2_vs_unet_advanced_diagnostics/prediction_cubes"
TAB = EXP / "paper_validation_final_seed20260813/tables"
DIA = EXP / "paper_validation_final_seed20260813/diagnostics"

N_BOOT = 2000
BLOCK_SIZES = [3, 5, 10]
SEED = 20260814
BATCH_BOOT = 100

with open(CUBE / "cube_metadata.json") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
N, H, W = shape
NY = N // 365

rcm = np.memmap(CUBE/"rcm.dat", dtype="float32", mode="r", shape=shape)
full = np.memmap(CUBE/"e2_full_seeded.dat", dtype="float32", mode="r", shape=shape)
notopo = np.memmap(CUBE/"e2_no_fine_topo_seeded.dat", dtype="float32", mode="r", shape=shape)
mask = np.memmap(CUBE/"mask.dat", dtype="bool", mode="r", shape=shape)

common = np.all(mask, axis=0)
idx = np.flatnonzero(common.ravel())
NC = len(idx)

print("="*100)
print("TOPOGRAPHY ABLATION BOOTSTRAP")
print("="*100)
print("years       :", NY)
print("common cells:", NC)
print("replicates  :", N_BOOT)
print("blocks      :", BLOCK_SIZES)

def year(arr, iy):
    s = iy*365
    return np.asarray(arr[s:s+365], dtype=np.float64).reshape(365,-1)[:,idx]

# ------------------------------------------------------------------
# Daily climatology
# ------------------------------------------------------------------

clim_r = np.zeros((365,NC))
clim_f = np.zeros((365,NC))
clim_n = np.zeros((365,NC))

for y in range(NY):
    clim_r += year(rcm,y)
    clim_f += year(full,y)
    clim_n += year(notopo,y)

clim_r /= NY
clim_f /= NY
clim_n /= NY

# ------------------------------------------------------------------
# Year sufficient statistics
# ------------------------------------------------------------------

sse_f = np.zeros(NY)
sse_n = np.zeros(NY)
sae_f = np.zeros(NY)
sae_n = np.zeros(NY)
se_f = np.zeros(NY)
se_n = np.zeros(NY)

sum_r = np.zeros((NY,NC))
sum_f = np.zeros((NY,NC))
sum_n = np.zeros((NY,NC))
ss_r = np.zeros((NY,NC))
ss_f = np.zeros((NY,NC))
ss_n = np.zeros((NY,NC))

sum_ty_r = np.zeros(NC)
sum_ty_f = np.zeros(NC)
sum_ty_n = np.zeros(NC)
sum_y_r = np.zeros(NC)
sum_y_f = np.zeros(NC)
sum_y_n = np.zeros(NC)

T = np.arange(N,dtype=np.float64)
sum_t = T.sum()
sum_t2 = (T*T).sum()
tmean = T.mean()

for y in range(NY):
    r = year(rcm,y)
    f = year(full,y)
    n = year(notopo,y)

    df = f-r
    dn = n-r

    sse_f[y] = np.sum(df*df)
    sse_n[y] = np.sum(dn*dn)
    sae_f[y] = np.sum(np.abs(df))
    sae_n[y] = np.sum(np.abs(dn))
    se_f[y] = np.sum(df)
    se_n[y] = np.sum(dn)

    sum_r[y] = r.sum(0)
    sum_f[y] = f.sum(0)
    sum_n[y] = n.sum(0)
    ss_r[y] = (r*r).sum(0)
    ss_f[y] = (f*f).sum(0)
    ss_n[y] = (n*n).sum(0)

    ar = r-clim_r
    af = f-clim_f
    an = n-clim_n

    s = y*365
    t = np.arange(s,s+365,dtype=np.float64)[:,None]

    sum_ty_r += (t*ar).sum(0)
    sum_ty_f += (t*af).sum(0)
    sum_ty_n += (t*an).sum(0)
    sum_y_r += ar.sum(0)
    sum_y_f += af.sum(0)
    sum_y_n += an.sum(0)

den = sum_t2-sum_t**2/N

slope_r = (sum_ty_r-sum_t*sum_y_r/N)/den
slope_f = (sum_ty_f-sum_t*sum_y_f/N)/den
slope_n = (sum_ty_n-sum_t*sum_y_n/N)/den

mean_r = sum_y_r/N
mean_fa = sum_y_f/N
mean_na = sum_y_n/N

def empty_corr():
    return {k:np.zeros((NY,NC)) for k in ["sx","sy","sxx","syy","sxy"]}

cf = empty_corr()
cn = empty_corr()
cdf = empty_corr()
cdn = empty_corr()

def add(stats,y,x,z):
    stats["sx"][y]=x.sum(0)
    stats["sy"][y]=z.sum(0)
    stats["sxx"][y]=(x*x).sum(0)
    stats["syy"][y]=(z*z).sum(0)
    stats["sxy"][y]=(x*z).sum(0)

for y in range(NY):
    r=year(rcm,y)
    f=year(full,y)
    n=year(notopo,y)

    ar=r-clim_r
    af=f-clim_f
    an=n-clim_n

    s=y*365
    t=np.arange(s,s+365,dtype=np.float64)[:,None]

    ard=ar-(mean_r[None,:]+slope_r[None,:]*(t-tmean))
    afd=af-(mean_fa[None,:]+slope_f[None,:]*(t-tmean))
    and_=an-(mean_na[None,:]+slope_n[None,:]*(t-tmean))

    add(cf,y,ar,af)
    add(cn,y,ar,an)
    add(cdf,y,ard,afd)
    add(cdn,y,ard,and_)

def corr(weights, stats):
    sx=weights@stats["sx"]
    sy=weights@stats["sy"]
    sxx=weights@stats["sxx"]
    syy=weights@stats["syy"]
    sxy=weights@stats["sxy"]

    n=weights.sum(1)[:,None]*365.0

    cov=sxy-sx*sy/n
    vx=sxx-sx*sx/n
    vy=syy-sy*sy/n

    with np.errstate(divide="ignore",invalid="ignore"):
        c=cov/np.sqrt(vx*vy)

    return np.nanmean(c,axis=1)

def variance(weights):
    n=weights.sum(1)[:,None]*365.0

    sr=weights@sum_r
    sf=weights@sum_f
    sn=weights@sum_n

    qr=weights@ss_r
    qf=weights@ss_f
    qn=weights@ss_n

    vr=qr/n-(sr/n)**2
    vf=qf/n-(sf/n)**2
    vn=qn/n-(sn/n)**2

    with np.errstate(divide="ignore",invalid="ignore"):
        rf=100*vf/vr
        rn=100*vn/vr

    meanf=np.nanmean(rf,1)
    meann=np.nanmean(rn,1)

    madf=np.nanmean(np.abs(rf-100),1)
    madn=np.nanmean(np.abs(rn-100),1)

    return meanf,meann,madf,madn

# ------------------------------------------------------------------
# Point estimates
# ------------------------------------------------------------------

wf=np.ones((1,NY))
nobs=N*NC

point = {
    "rmse_f":np.sqrt(sse_f.sum()/nobs),
    "rmse_n":np.sqrt(sse_n.sum()/nobs),
    "mae_f":sae_f.sum()/nobs,
    "mae_n":sae_n.sum()/nobs,
    "bias_f":se_f.sum()/nobs,
    "bias_n":se_n.sum()/nobs,
    "corr_f":corr(wf,cf)[0],
    "corr_n":corr(wf,cn)[0],
    "corrdt_f":corr(wf,cdf)[0],
    "corrdt_n":corr(wf,cdn)[0],
}

point["absbias_f"]=abs(point["bias_f"])
point["absbias_n"]=abs(point["bias_n"])

vf,vn,mf,mn=variance(wf)
point["var_f"]=vf[0]
point["var_n"]=vn[0]
point["varmad_f"]=mf[0]
point["varmad_n"]=mn[0]

print("\nPoint estimates:")
for k,v in point.items():
    print(f"{k:15s}: {v}")

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

rows=[]
rngmaster=np.random.default_rng(SEED)

def CI(x):
    return float(np.percentile(x,2.5)),float(np.percentile(x,97.5))

for block in BLOCK_SIZES:

    rng=np.random.default_rng(int(rngmaster.integers(0,2**31-1)))
    nb=math.ceil(NY/block)
    W=np.zeros((N_BOOT,NY))

    for b in range(N_BOOT):
        yrs=[]
        starts=rng.integers(0,NY,size=nb)
        for s in starts:
            yrs.extend((s+j)%NY for j in range(block))
        yrs=yrs[:NY]
        W[b]=np.bincount(yrs,minlength=NY)

    d={
        "rmse_f":np.zeros(N_BOOT),
        "rmse_n":np.zeros(N_BOOT),
        "mae_f":np.zeros(N_BOOT),
        "mae_n":np.zeros(N_BOOT),
        "absbias_f":np.zeros(N_BOOT),
        "absbias_n":np.zeros(N_BOOT),
        "corr_f":np.zeros(N_BOOT),
        "corr_n":np.zeros(N_BOOT),
        "corrdt_f":np.zeros(N_BOOT),
        "corrdt_n":np.zeros(N_BOOT),
        "varmad_f":np.zeros(N_BOOT),
        "varmad_n":np.zeros(N_BOOT),
    }

    for b0 in range(0,N_BOOT,BATCH_BOOT):
        b1=min(b0+BATCH_BOOT,N_BOOT)
        w=W[b0:b1]
        nn=w.sum(1)*365.0*NC

        d["rmse_f"][b0:b1]=np.sqrt((w@sse_f)/nn)
        d["rmse_n"][b0:b1]=np.sqrt((w@sse_n)/nn)
        d["mae_f"][b0:b1]=(w@sae_f)/nn
        d["mae_n"][b0:b1]=(w@sae_n)/nn
        d["absbias_f"][b0:b1]=np.abs((w@se_f)/nn)
        d["absbias_n"][b0:b1]=np.abs((w@se_n)/nn)

        d["corr_f"][b0:b1]=corr(w,cf)
        d["corr_n"][b0:b1]=corr(w,cn)
        d["corrdt_f"][b0:b1]=corr(w,cdf)
        d["corrdt_n"][b0:b1]=corr(w,cdn)

        _,_,mF,mN=variance(w)
        d["varmad_f"][b0:b1]=mF
        d["varmad_n"][b0:b1]=mN

    tests=[
        ("RMSE_global","degC",
         point["rmse_n"]-point["rmse_f"],
         d["rmse_n"]-d["rmse_f"]),

        ("MAE_global","degC",
         point["mae_n"]-point["mae_f"],
         d["mae_n"]-d["mae_f"]),

        ("absolute_bias_global","degC",
         point["absbias_n"]-point["absbias_f"],
         d["absbias_n"]-d["absbias_f"]),

        ("daily_anomaly_correlation","correlation",
         point["corr_f"]-point["corr_n"],
         d["corr_f"]-d["corr_n"]),

        ("detrended_daily_anomaly_correlation","correlation",
         point["corrdt_f"]-point["corrdt_n"],
         d["corrdt_f"]-d["corrdt_n"]),

        ("variance_fidelity","percentage_points",
         point["varmad_n"]-point["varmad_f"],
         d["varmad_n"]-d["varmad_f"]),
    ]

    for name,unit,padv,adv in tests:
        lo,hi=CI(adv)
        rows.append({
            "block_years":block,
            "metric":name,
            "units":unit,
            "FULL_advantage_point":padv,
            "advantage_ci2.5":lo,
            "advantage_ci97.5":hi,
            "bootstrap_probability_FULL_better":float(np.mean(adv>0)),
            "robust_95pct":bool(lo>0 or hi<0),
            "direction":
                "FULL_better" if lo>0 else
                "NO_TOPO_better" if hi<0 else
                "uncertain",
        })

df=pd.DataFrame(rows)

out=TAB/"35_topography_ablation_bootstrap_summary.csv"
df.to_csv(out,index=False)

print("\n"+"="*100)
print(df.to_string(index=False))
print("\nSaved:",out)
