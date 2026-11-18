# Environment and installation notes

The recovered code directly imports PyTorch, NumPy, pandas, xarray, SciPy and Matplotlib. The true original Python/CUDA/environment versions were **not yet provided**. This is deliberately *not* a lockfile.

For static source and hash checks, only Python 3 with a standard library is needed:

```bash
python tools/preflight.py
python tools/author_source_integrity.py
python tools/check_reference_files.py
```

For a **synthetic** model-forward smoke test, install a PyTorch version appropriate for your CPU/GPU platform, then run:

```bash
python tools/smoke_test_models.py
```

The archived training scripts use `torch.amp`, and the preparation/diagnostics use `xarray` for NetCDF and NumPy memmaps. Suitable xarray backends (`netCDF4`, `h5netcdf`, `cftime`) may be required depending on the datasets. `requirements-unpinned.txt` is an import inventory, **not** a version-tested environment.

To capture the exact environment on the experiment server, activate the actual Conda environment where seeded training was executed (not an unrelated `base` shell) and run:

```bash
python --version
python -m pip freeze > ~/topofusion_active_pip_freeze.txt
python - <<'PY2'
import sys, torch
print('Python:',sys.version)
print('Torch:',torch.__version__,'built with CUDA:',torch.version.cuda)
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
PY2
conda env export --from-history > ~/topofusion_conda_from_history.yml
```

Review the exports for private channels, secrets and personally identifying filesystem paths before publishing.
