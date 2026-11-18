#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.format import open_memmap
from scipy.ndimage import uniform_filter


ROOT = Path("/workspace/SwinIR_HF/Normalizacion")

CONFIG_PATH = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography/configs/config_model_E.json"

EXP_DIR = ROOT / "climate_swinir_experiments/model_E_cmip_swinunet_topography"
LOG_DIR = EXP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DIAG_DIR = EXP_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

CMIP_OROG_22_PATH = EXP_DIR / "diagnostics/cmip_orog_22x22_first_valid.npy"

DYNAMIC_VARS = [
    "hus250",
    "hus500",
    "hus700",
    "hus850",
    "psl",
    "ta250",
    "ta500",
    "ta700",
    "ta850",
    "tas",
    "tasmax",
    "tasmin",
    "tos",
    "ua250",
    "ua500",
    "ua700",
    "ua850",
    "uas",
    "va250",
    "va500",
    "va700",
    "va850",
    "vas",
    "zg500",
]

GHG_COLUMNS = [
    "year",
    "CO2",
    "CH4",
    "N2O",
    "CFC_11eq",
    "CFC_12",
    "Solaire",
    "Chlore",
    "CFC_11",
    "CO",
    "CFC113",
    "CCl4",
    "CH3CCl3",
    "CH3Cl",
    "HCFC22",
    "CH3Br",
    "H1211",
    "H1301",
    "Cly",
    "Bry",
]

GHG_SELECTED = ["CO2", "CH4", "N2O", "Solaire"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def start_log():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"01_prepare_dataset_model_E_{timestamp}.log"
    f = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)

    print("=" * 100)
    print("Prepare dataset Model E")
    print(f"Log path : {log_path}")
    print(f"Root     : {ROOT}")
    print(f"Started  : {timestamp}")
    print("=" * 100)

    return original_stdout, original_stderr, f, log_path


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def parse_time(values) -> pd.DatetimeIndex:
    return pd.to_datetime([str(v)[:19] for v in values])


def inclusive_mask(times: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return np.asarray((times >= start_ts) & (times <= end_ts), dtype=bool)


def safe_open_dataset(path: Path) -> xr.Dataset:
    drop_candidates = ["lat_bnds", "lon_bnds", "time_bnds"]
    try:
        return xr.open_dataset(path, drop_variables=drop_candidates)
    except Exception:
        return xr.open_dataset(path)


def find_orography_var(ds: xr.Dataset) -> str:
    for name in ["orography", "orog", "elevation", "altitude", "Band1", "z"]:
        if name in ds.data_vars:
            return name

    vars_2d = [v for v in ds.data_vars if len(ds[v].shape) == 2]
    if not vars_2d:
        raise RuntimeError(f"No 2D orography-like variable found. Vars={list(ds.data_vars)}")

    return vars_2d[0]


def kelvin_to_celsius_if_needed(arr: np.ndarray, units: str | None = None) -> tuple[np.ndarray, str]:
    finite = arr[np.isfinite(arr)]
    mean_val = float(np.mean(finite)) if finite.size else np.nan
    units_l = str(units).lower()

    if units_l in ["k", "kelvin"] or "kelvin" in units_l or mean_val > 100:
        return arr - 273.15, "K_to_C"

    return arr, "kept_as_is"


def normalize_with_train_stats(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict]:
    finite_train = train_values[np.isfinite(train_values)]

    if finite_train.size == 0:
        raise RuntimeError("No finite values in train_values for normalization.")

    mean = float(np.mean(finite_train))
    std = float(np.std(finite_train))

    if std < 1e-8:
        std = 1.0

    out = (values - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    stats = {
        "mean": mean,
        "std": std,
        "train_min": float(np.min(finite_train)),
        "train_max": float(np.max(finite_train)),
        "train_p01": float(np.percentile(finite_train, 1)),
        "train_p99": float(np.percentile(finite_train, 99)),
    }

    return out, stats


def preprocess_dynamic_variable(values: np.ndarray, var_name: str, units: str | None = None) -> tuple[np.ndarray, dict]:
    """
    Preprocesamiento físico no destructivo para predictores CMIP.

    Objetivo:
    - Respetar la naturaleza física de cada variable.
    - Mejorar estabilidad numérica.
    - No borrar la señal climática absoluta ni temporal.
    - Luego se aplica z-score usando SOLO train interno.

    Transformaciones:
    - hus*: log(x + epsilon), porque humedad específica es positiva y sesgada.
    - ta*, tas, tasmax, tasmin, tos: K -> °C si parece Kelvin.
    - psl: Pa -> hPa si parece Pascales.
    - vientos: se mantienen en m/s.
    - zg*: se mantiene en m.
    """
    x = values.astype(np.float32, copy=False)
    units_l = str(units).lower()
    action = "none"
    epsilon = 1e-8

    finite = x[np.isfinite(x)]
    raw_mean = float(np.mean(finite)) if finite.size else float("nan")
    raw_min = float(np.min(finite)) if finite.size else float("nan")
    raw_max = float(np.max(finite)) if finite.size else float("nan")

    if var_name.startswith("hus"):
        # Evitar log de cero o valores negativos espurios.
        x = np.maximum(x, 0.0)
        x = np.log(x + epsilon).astype(np.float32)
        action = f"log_x_plus_{epsilon}"

    elif var_name.startswith("ta") or var_name in ["tas", "tasmax", "tasmin", "tos"]:
        # CMIP normalmente viene en K. Convertimos si metadata o magnitud lo sugieren.
        if units_l in ["k", "kelvin"] or "kelvin" in units_l or raw_mean > 100:
            x = (x - 273.15).astype(np.float32)
            action = "K_to_C"
        else:
            action = "kept_temperature_scale"

    elif var_name == "psl":
        # Presión a nivel del mar usualmente en Pa. Convertimos a hPa.
        if units_l in ["pa", "pascal", "pascals"] or raw_mean > 2000:
            x = (x / 100.0).astype(np.float32)
            action = "Pa_to_hPa"
        else:
            action = "kept_pressure_scale"

    elif var_name.startswith("ua") or var_name.startswith("va") or var_name in ["uas", "vas"]:
        action = "kept_wind_ms"

    elif var_name.startswith("zg"):
        action = "kept_geopotential_height_m"

    else:
        action = "none"

    finite2 = x[np.isfinite(x)]
    out_mean = float(np.mean(finite2)) if finite2.size else float("nan")
    out_min = float(np.min(finite2)) if finite2.size else float("nan")
    out_max = float(np.max(finite2)) if finite2.size else float("nan")

    meta = {
        "variable": var_name,
        "units_original": str(units),
        "preprocess_action": action,
        "raw_mean": raw_mean,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "preprocessed_mean": out_mean,
        "preprocessed_min": out_min,
        "preprocessed_max": out_max,
    }

    return x.astype(np.float32), meta


def normalize_array_with_mask(arr: np.ndarray, mask: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    vals = arr[mask & np.isfinite(arr)]

    if vals.size == 0:
        vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        raise RuntimeError(f"No finite values found for static feature: {name}")

    mean = float(np.mean(vals))
    std = float(np.std(vals))

    if std < 1e-8:
        std = 1.0

    out = (arr - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    stats = {
        "name": name,
        "mean": mean,
        "std": std,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p99": float(np.percentile(vals, 99)),
    }

    return out, stats


def minmax_to_minus1_plus1(arr: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        raise RuntimeError(f"No finite values found for {name}")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if abs(vmax - vmin) < 1e-12:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = 2.0 * ((arr - vmin) / (vmax - vmin)) - 1.0
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    stats = {
        "name": name,
        "min": vmin,
        "max": vmax,
        "method": "minmax_to_minus1_plus1",
    }

    return out, stats


def build_slope_and_roughness(orog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(orog.astype(np.float32))
    slope = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    mean = uniform_filter(orog.astype(np.float32), size=3, mode="nearest")
    mean2 = uniform_filter((orog.astype(np.float32) ** 2), size=3, mode="nearest")
    roughness = np.sqrt(np.maximum(mean2 - mean ** 2, 0.0)).astype(np.float32)

    return slope, roughness


def load_ghg_table(path: Path) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < len(GHG_COLUMNS):
                continue

            values = parts[: len(GHG_COLUMNS)]
            rows.append(values)

    if not rows:
        raise RuntimeError(f"No data rows parsed from GHG file: {path}")

    df = pd.DataFrame(rows, columns=GHG_COLUMNS)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df


def build_ghg_daily_features(times: pd.DatetimeIndex, ghg_path: Path) -> np.ndarray:
    df = load_ghg_table(ghg_path)
    years = times.year

    df_idx = df.set_index("year")

    arr = np.zeros((len(times), len(GHG_SELECTED)), dtype=np.float32)

    for j, col in enumerate(GHG_SELECTED):
        values = []

        for y in years:
            if y in df_idx.index:
                values.append(float(df_idx.loc[y, col]))
            else:
                raise RuntimeError(f"Year {y} not found in GHG file {ghg_path}")

        arr[:, j] = np.array(values, dtype=np.float32)

    return arr


def build_time_features(times: pd.DatetimeIndex, year_min: int = 1950, year_max: int = 2100) -> np.ndarray:
    month = times.month.to_numpy().astype(np.float32)
    year = times.year.to_numpy().astype(np.float32)

    month_angle = 2.0 * np.pi * (month - 1.0) / 12.0

    month_sin = np.sin(month_angle)
    month_cos = np.cos(month_angle)
    year_norm = (year - year_min) / (year_max - year_min)

    out = np.stack([month_sin, month_cos, year_norm], axis=1).astype(np.float32)
    return out


def save_json(path: Path, obj: dict | list):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def describe_np(name: str, arr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None and arr.ndim == 3:
        vals = arr[:, mask]
    elif mask is not None and arr.ndim == 2:
        vals = arr[mask]
    else:
        vals = arr.reshape(-1)

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "name": name,
            "n": 0,
            "min": None,
            "mean": None,
            "max": None,
            "std": None,
        }

    return {
        "name": name,
        "n": int(vals.size),
        "min": float(np.min(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p05": float(np.percentile(vals, 5)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
    }


def main():
    cfg = json.loads(CONFIG_PATH.read_text())

    dataset_path = resolve_path(cfg["dataset_path"])
    dataset_path.mkdir(parents=True, exist_ok=True)

    raw = cfg["raw_data"]
    paths = cfg["paths"]
    split = cfg["split"]

    training_features_path = resolve_path(raw["training_features_path"])
    testing_features_path = resolve_path(raw["testing_features_path"])
    training_target_path = resolve_path(raw["training_target_path"])
    testing_target_path = resolve_path(raw["testing_target_path"])
    training_ghg_path = resolve_path(raw["training_ghg_path"])
    testing_ghg_path = resolve_path(raw["testing_ghg_path"])
    mask_path = resolve_path(raw["mask_path"])
    orography_path = resolve_path(paths["orography_path"])

    print("\nResolved paths")
    print("=" * 100)
    for label, path in [
        ("training_features_path", training_features_path),
        ("testing_features_path", testing_features_path),
        ("training_target_path", training_target_path),
        ("testing_target_path", testing_target_path),
        ("training_ghg_path", training_ghg_path),
        ("testing_ghg_path", testing_ghg_path),
        ("mask_path", mask_path),
        ("orography_path", orography_path),
        ("cmip_orog_22_path", CMIP_OROG_22_PATH),
        ("dataset_path", dataset_path),
    ]:
        print(f"{label:24s}: {path}")
        if label != "dataset_path" and not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("\nDynamic variables")
    print("=" * 100)
    print(DYNAMIC_VARS)
    print(f"n_dynamic_vars: {len(DYNAMIC_VARS)}")

    ds_x_train = xr.open_dataset(training_features_path)
    ds_x_test = xr.open_dataset(testing_features_path)
    ds_y_train = safe_open_dataset(training_target_path)
    ds_y_test = safe_open_dataset(testing_target_path)
    ds_mask = xr.open_dataset(mask_path)
    ds_orog = xr.open_dataset(orography_path)

    missing_train = [v for v in DYNAMIC_VARS if v not in ds_x_train.data_vars]
    missing_test = [v for v in DYNAMIC_VARS if v not in ds_x_test.data_vars]

    if missing_train:
        raise RuntimeError(f"Missing variables in training features: {missing_train}")

    if missing_test:
        raise RuntimeError(f"Missing variables in testing features: {missing_test}")

    train_times_all = parse_time(ds_x_train["time"].values)
    test_times = parse_time(ds_x_test["time"].values)

    y_train_times = parse_time(ds_y_train["time"].values)
    y_test_times = parse_time(ds_y_test["time"].values)

    print("\nTime ranges")
    print("=" * 100)
    print(f"X train all : {train_times_all.min()} -> {train_times_all.max()} | n={len(train_times_all)}")
    print(f"Y train all : {y_train_times.min()} -> {y_train_times.max()} | n={len(y_train_times)}")
    print(f"X test SSP370: {test_times.min()} -> {test_times.max()} | n={len(test_times)}")
    print(f"Y test SSP370: {y_test_times.min()} -> {y_test_times.max()} | n={len(y_test_times)}")

    if len(train_times_all) != len(y_train_times):
        raise RuntimeError("Training feature and target time lengths differ.")

    if len(test_times) != len(y_test_times):
        raise RuntimeError("Testing feature and target time lengths differ.")

    train_mask_time = inclusive_mask(train_times_all, split["train_start"], split["train_end"])
    val_mask_time = inclusive_mask(train_times_all, split["val_start"], split["val_end"])

    train_idx = np.where(train_mask_time)[0]
    val_idx = np.where(val_mask_time)[0]

    if len(train_idx) + len(val_idx) != len(train_times_all):
        print("WARNING: train + val split does not cover all training times.")

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_times)

    print("\nSplit")
    print("=" * 100)
    print(f"train: {split['train_start']} -> {split['train_end']} | n={n_train}")
    print(f"val  : {split['val_start']} -> {split['val_end']} | n={n_val}")
    print(f"test : {split['test_start']} -> {split['test_end']} | n={n_test}")

    c = len(DYNAMIC_VARS)
    h22 = ds_x_train[DYNAMIC_VARS[0]].shape[-2]
    w22 = ds_x_train[DYNAMIC_VARS[0]].shape[-1]

    print("\nCreating dynamic input memmaps")
    print("=" * 100)

    train_inputs_path = dataset_path / "train_inputs.npy"
    val_inputs_path = dataset_path / "val_inputs.npy"
    test_inputs_path = dataset_path / "test_inputs_ssp370.npy"

    train_inputs = open_memmap(train_inputs_path, mode="w+", dtype=np.float32, shape=(n_train, c, h22, w22))
    val_inputs = open_memmap(val_inputs_path, mode="w+", dtype=np.float32, shape=(n_val, c, h22, w22))
    test_inputs = open_memmap(test_inputs_path, mode="w+", dtype=np.float32, shape=(n_test, c, h22, w22))

    dynamic_stats = {}
    missing_report = {}

    for j, var in enumerate(DYNAMIC_VARS):
        print(f"[{j+1:02d}/{c}] Processing dynamic variable: {var}")

        units_train = ds_x_train[var].attrs.get("units", "unknown")
        units_test = ds_x_test[var].attrs.get("units", "unknown")

        data_all_raw = ds_x_train[var].values.astype(np.float32)
        data_test_raw = ds_x_test[var].values.astype(np.float32)

        data_all, preprocess_meta_train = preprocess_dynamic_variable(data_all_raw, var, units_train)
        data_test, preprocess_meta_test = preprocess_dynamic_variable(data_test_raw, var, units_test)

        train_raw = data_all[train_idx]
        val_raw = data_all[val_idx]

        finite_train = train_raw[np.isfinite(train_raw)]

        if finite_train.size == 0:
            raise RuntimeError(f"No finite train values for variable {var}")

        mean = float(np.mean(finite_train))
        std = float(np.std(finite_train))

        if std < 1e-8:
            std = 1.0

        train_norm = (train_raw - mean) / std
        val_norm = (val_raw - mean) / std
        test_norm = (data_test - mean) / std

        train_inputs[:, j, :, :] = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        val_inputs[:, j, :, :] = np.nan_to_num(val_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        test_inputs[:, j, :, :] = np.nan_to_num(test_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        dynamic_stats[var] = {
            "mean": mean,
            "std": std,
            "train_min": float(np.min(finite_train)),
            "train_max": float(np.max(finite_train)),
            "train_p01": float(np.percentile(finite_train, 1)),
            "train_p99": float(np.percentile(finite_train, 99)),
            "units_train_original": units_train,
            "units_test_original": units_test,
            "preprocess_train": preprocess_meta_train,
            "preprocess_test": preprocess_meta_test,
            "long_name": ds_x_train[var].attrs.get("long_name", ""),
            "standard_name": ds_x_train[var].attrs.get("standard_name", ""),
            "normalization": "zscore_using_train_split_after_physical_preprocessing",
        }

        missing_report[var] = {
            "train_nan_count_raw": int(np.isnan(data_all_raw[train_idx]).sum()),
            "val_nan_count_raw": int(np.isnan(data_all_raw[val_idx]).sum()),
            "test_nan_count_raw": int(np.isnan(data_test_raw).sum()),
            "train_nan_count_after_preprocess": int(np.isnan(train_raw).sum()),
            "val_nan_count_after_preprocess": int(np.isnan(val_raw).sum()),
            "test_nan_count_after_preprocess": int(np.isnan(data_test).sum()),
            "train_total": int(train_raw.size),
            "val_total": int(val_raw.size),
            "test_total": int(data_test.size),
        }

        print(
            f"    preprocess={preprocess_meta_train['preprocess_action']} | "
            f"train mean/std after preprocess before zscore = {mean:.6g}/{std:.6g}"
        )

        del data_all_raw, data_test_raw, data_all, data_test, train_raw, val_raw
        del finite_train, train_norm, val_norm, test_norm

    train_inputs.flush()
    val_inputs.flush()
    test_inputs.flush()

    print("\nDynamic inputs saved")
    print("=" * 100)
    print(train_inputs_path)
    print(val_inputs_path)
    print(test_inputs_path)

    print("\nPreparing targets")
    print("=" * 100)

    y_train_da = ds_y_train["tas"]
    y_test_da = ds_y_test["tas"]

    if set(["time", "y", "x"]).issubset(set(y_train_da.dims)):
        y_train_da = y_train_da.transpose("time", "y", "x")

    if set(["time", "y", "x"]).issubset(set(y_test_da.dims)):
        y_test_da = y_test_da.transpose("time", "y", "x")

    y_all = y_train_da.values.astype(np.float32)
    y_test = y_test_da.values.astype(np.float32)

    y_all, y_train_unit_action = kelvin_to_celsius_if_needed(y_all, y_train_da.attrs.get("units", "unknown"))
    y_test, y_test_unit_action = kelvin_to_celsius_if_needed(y_test, y_test_da.attrs.get("units", "unknown"))

    print(f"target train unit action: {y_train_unit_action}")
    print(f"target test unit action : {y_test_unit_action}")

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    sftlf = ds_mask["sftlf"].values.astype(np.float32)
    sftlf_fraction = sftlf / 100.0

    # Máscara oficial para entrenamiento/evaluación:
    # usamos tierra dominante para ser comparables con el benchmark U-Net CMIP-CMIP.
    target_valid_mask = (sftlf > 50).astype(bool)

    print(f"target_valid_mask cells: {int(target_valid_mask.sum())}")
    print(f"sftlf > 0 cells        : {int((sftlf > 0).sum())}")
    print(f"sftlf > 50 cells       : {int((sftlf > 50).sum())}")

    np.save(dataset_path / "target_valid_mask_64x64.npy", target_valid_mask.astype(np.float32))
    np.save(dataset_path / "sftlf_64x64.npy", sftlf.astype(np.float32))
    np.save(dataset_path / "sftlf_fraction_64x64.npy", sftlf_fraction.astype(np.float32))

    target_summary = {
        "train_target_C": describe_np("train_target_C", y_train, target_valid_mask),
        "val_target_C": describe_np("val_target_C", y_val, target_valid_mask),
        "test_target_C": describe_np("test_target_C", y_test, target_valid_mask),
    }

    train_targets_path = dataset_path / "train_targets.npy"
    val_targets_path = dataset_path / "val_targets.npy"
    test_targets_path = dataset_path / "test_targets_ssp370.npy"

    np.save(train_targets_path, np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    np.save(val_targets_path, np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    np.save(test_targets_path, np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

    print("Targets saved:")
    print(train_targets_path)
    print(val_targets_path)
    print(test_targets_path)

    print("\nPreparing static features 64x64")
    print("=" * 100)

    orog_var = find_orography_var(ds_orog)
    orog = ds_orog[orog_var].values.astype(np.float32)

    if orog.shape != (64, 64):
        raise RuntimeError(f"Expected orography 64x64, got {orog.shape}")

    lat64 = ds_mask["lat"].values.astype(np.float32)
    lon64 = ds_mask["lon"].values.astype(np.float32)

    slope, roughness = build_slope_and_roughness(orog)

    static_stats = {}

    orog_norm, st = normalize_array_with_mask(orog, target_valid_mask, "orography_64")
    static_stats["orography_64"] = st

    slope_norm, st = normalize_array_with_mask(slope, target_valid_mask, "slope_64")
    static_stats["slope_64"] = st

    roughness_norm, st = normalize_array_with_mask(roughness, target_valid_mask, "roughness_64")
    static_stats["roughness_64"] = st

    lat_norm, st = minmax_to_minus1_plus1(lat64, "lat_64")
    static_stats["lat_64"] = st

    lon_norm, st = minmax_to_minus1_plus1(lon64, "lon_64")
    static_stats["lon_64"] = st

    land_feature = sftlf_fraction.astype(np.float32)

    static_features = np.stack(
        [
            orog_norm,
            land_feature,
            lat_norm,
            lon_norm,
            slope_norm,
            roughness_norm,
        ],
        axis=0,
    ).astype(np.float32)

    static_names = [
        "orography_64_norm",
        "sftlf_fraction_64",
        "lat_64_norm",
        "lon_64_norm",
        "slope_64_norm",
        "roughness_64_norm",
    ]

    np.save(dataset_path / "static_features_64x64.npy", static_features)
    save_json(dataset_path / "static_feature_names_64x64.json", static_names)

    print("static_features_64x64.npy shape:", static_features.shape)

    print("\nPreparing coarse static 22x22")
    print("=" * 100)

    cmip_orog = np.load(CMIP_OROG_22_PATH).astype(np.float32)
    cmip_orog = np.maximum(cmip_orog, 0.0)

    cmip_vals = cmip_orog[np.isfinite(cmip_orog)]

    cmip_mean = float(np.mean(cmip_vals))
    cmip_std = float(np.std(cmip_vals))

    if cmip_std < 1e-8:
        cmip_std = 1.0

    cmip_orog_norm = (cmip_orog - cmip_mean) / cmip_std
    cmip_orog_norm = np.nan_to_num(cmip_orog_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    coarse_static = cmip_orog_norm[None, :, :].astype(np.float32)

    np.save(dataset_path / "coarse_static_22x22.npy", coarse_static)
    save_json(dataset_path / "coarse_static_feature_names_22x22.json", ["cmip_orog_22_norm"])

    coarse_static_stats = {
        "cmip_orog_22": {
            "mean": cmip_mean,
            "std": cmip_std,
            "min": float(np.min(cmip_vals)),
            "max": float(np.max(cmip_vals)),
            "p01": float(np.percentile(cmip_vals, 1)),
            "p99": float(np.percentile(cmip_vals, 99)),
            "note": "negative values clipped to zero before normalization",
        }
    }

    print("coarse_static_22x22.npy shape:", coarse_static.shape)

    print("\nPreparing time features")
    print("=" * 100)

    train_times = train_times_all[train_idx]
    val_times = train_times_all[val_idx]

    time_train = build_time_features(train_times)
    time_val = build_time_features(val_times)
    time_test = build_time_features(test_times)

    np.save(dataset_path / "time_features_train.npy", time_train)
    np.save(dataset_path / "time_features_val.npy", time_val)
    np.save(dataset_path / "time_features_test_ssp370.npy", time_test)

    np.save(dataset_path / "train_time.npy", train_times.astype("datetime64[ns]").to_numpy())
    np.save(dataset_path / "val_time.npy", val_times.astype("datetime64[ns]").to_numpy())
    np.save(dataset_path / "test_time_ssp370.npy", test_times.astype("datetime64[ns]").to_numpy())

    save_json(dataset_path / "time_feature_names.json", ["month_sin", "month_cos", "year_norm"])

    print("time_features_train:", time_train.shape)
    print("time_features_val  :", time_val.shape)
    print("time_features_test :", time_test.shape)

    print("\nPreparing GHG features")
    print("=" * 100)

    ghg_train_all = build_ghg_daily_features(train_times_all, training_ghg_path)
    ghg_test_raw = build_ghg_daily_features(test_times, testing_ghg_path)

    ghg_train_raw = ghg_train_all[train_idx]
    ghg_val_raw = ghg_train_all[val_idx]

    ghg_stats = {}

    ghg_train_norm = np.zeros_like(ghg_train_raw, dtype=np.float32)
    ghg_val_norm = np.zeros_like(ghg_val_raw, dtype=np.float32)
    ghg_test_norm = np.zeros_like(ghg_test_raw, dtype=np.float32)

    for j, name in enumerate(GHG_SELECTED):
        vals_train = ghg_train_raw[:, j]
        mean = float(np.mean(vals_train))
        std = float(np.std(vals_train))

        if std < 1e-8:
            std = 1.0

        ghg_train_norm[:, j] = (ghg_train_raw[:, j] - mean) / std
        ghg_val_norm[:, j] = (ghg_val_raw[:, j] - mean) / std
        ghg_test_norm[:, j] = (ghg_test_raw[:, j] - mean) / std

        ghg_stats[name] = {
            "mean": mean,
            "std": std,
            "train_min": float(np.min(vals_train)),
            "train_max": float(np.max(vals_train)),
            "train_p01": float(np.percentile(vals_train, 1)),
            "train_p99": float(np.percentile(vals_train, 99)),
        }

    np.save(dataset_path / "ghg_features_train.npy", ghg_train_norm.astype(np.float32))
    np.save(dataset_path / "ghg_features_val.npy", ghg_val_norm.astype(np.float32))
    np.save(dataset_path / "ghg_features_test_ssp370.npy", ghg_test_norm.astype(np.float32))

    save_json(dataset_path / "ghg_feature_names.json", GHG_SELECTED)

    print("ghg_features_train:", ghg_train_norm.shape)
    print("ghg_features_val  :", ghg_val_norm.shape)
    print("ghg_features_test :", ghg_test_norm.shape)

    print("\nSaving metadata")
    print("=" * 100)

    save_json(dataset_path / "dynamic_variable_names.json", DYNAMIC_VARS)

    normalization_stats = {
        "dynamic": dynamic_stats,
        "static_64x64": static_stats,
        "coarse_static_22x22": coarse_static_stats,
        "ghg": ghg_stats,
        "target": {
            "train_unit_action": y_train_unit_action,
            "test_unit_action": y_test_unit_action,
            "kelvin_offset": 273.15,
        },
    }

    save_json(dataset_path / "normalization_stats.json", normalization_stats)
    save_json(dataset_path / "dynamic_missing_report.json", missing_report)

    dataset_summary = {
        "experiment_name": cfg["experiment_name"],
        "dataset_path": str(dataset_path),
        "created_at": datetime.now().isoformat(),
        "paths": {
            "training_features_path": str(training_features_path),
            "testing_features_path": str(testing_features_path),
            "training_target_path": str(training_target_path),
            "testing_target_path": str(testing_target_path),
            "training_ghg_path": str(training_ghg_path),
            "testing_ghg_path": str(testing_ghg_path),
            "mask_path": str(mask_path),
            "orography_path": str(orography_path),
            "cmip_orog_22_path": str(CMIP_OROG_22_PATH),
        },
        "split": {
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "val_start": split["val_start"],
            "val_end": split["val_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
        },
        "shapes": {
            "train_inputs": list(train_inputs.shape),
            "val_inputs": list(val_inputs.shape),
            "test_inputs_ssp370": list(test_inputs.shape),
            "train_targets": list(y_train.shape),
            "val_targets": list(y_val.shape),
            "test_targets_ssp370": list(y_test.shape),
            "static_features_64x64": list(static_features.shape),
            "coarse_static_22x22": list(coarse_static.shape),
            "time_features_train": list(time_train.shape),
            "time_features_val": list(time_val.shape),
            "time_features_test_ssp370": list(time_test.shape),
            "ghg_features_train": list(ghg_train_norm.shape),
            "ghg_features_val": list(ghg_val_norm.shape),
            "ghg_features_test_ssp370": list(ghg_test_norm.shape),
        },
        "dynamic_variables": DYNAMIC_VARS,
        "static_feature_names_64x64": static_names,
        "coarse_static_feature_names_22x22": ["cmip_orog_22_norm"],
        "time_feature_names": ["month_sin", "month_cos", "year_norm"],
        "ghg_feature_names": GHG_SELECTED,
        "target_valid_mask_cells": int(target_valid_mask.sum()),
        "sftlf_gt0_cells": int((sftlf > 0).sum()),
        "sftlf_gt50_cells": int((sftlf > 50).sum()),
        "target_summary": target_summary,
    }

    save_json(dataset_path / "dataset_summary.json", dataset_summary)

    report_path = dataset_path / "dataset_audit_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model E dataset audit report\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Dataset path: {dataset_path}\n\n")
        f.write("Shapes:\n")
        for k, v in dataset_summary["shapes"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\nSplit:\n")
        for k, v in dataset_summary["split"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\nTarget summaries:\n")
        for k, v in target_summary.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nDynamic variables:\n")
        for v in DYNAMIC_VARS:
            f.write(f"  - {v}\n")

        f.write("\nStatic features 64x64:\n")
        for v in static_names:
            f.write(f"  - {v}\n")

        f.write("\nFiles generated:\n")
        for p in sorted(dataset_path.glob("*")):
            f.write(f"  {p.name}\n")

    print("\nDataset summary")
    print("=" * 100)
    print(json.dumps(dataset_summary["shapes"], indent=2))
    print("\nTarget summary")
    print(json.dumps(target_summary, indent=2))
    print("\nSaved report:")
    print(report_path)

    print("\nDONE")
    print("=" * 100)


if __name__ == "__main__":
    original_stdout, original_stderr, log_file, log_path = start_log()

    try:
        main()
        print("\nFinished successfully.")
        print(f"Log saved to: {log_path}")

    except Exception as e:
        print("\nFAILED")
        print(repr(e))
        print(f"Log saved to: {log_path}")
        raise

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
