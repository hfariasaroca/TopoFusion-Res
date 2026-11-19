# TopoFusion-Res

**Fusión topográfica multirresolución y aprendizaje residual para emulación climática regional.**

Repositorio del código científico utilizado para emular temperatura diaria del RCM CNRM-ALADIN (64 × 64) a partir de predictores del GCM CNRM-ESM2-1 (22 × 22). El modelo es **convolucional, no SwinIR/SwinUNet**. Combina reconstrucción 22 → 44 → 64, topografía de alta resolución y una segunda etapa de corrección residual acotada.

**Protocolo final:** semilla `20260813`, 40 épocas para la red base y 12 para E2; entrenamiento histórico + SSP5-8.5 y evaluación independiente en SSP3-7.0 (2015–2100). RMSE diario reportado contra ALADIN: 1,37 °C (TopoFusion-Res) frente a 1,54 °C (U-Net).

## Estructura

- `models/`: arquitectura base, E2 y clases de datos.
- `scripts/`: preparación, entrenamiento y evaluación finales.
- `analysis/scripts/`: bootstrap, ablaciones y figuras científicas.
- `configs/`: configuración y nombres de entradas.
- `assets/`: orografía GCM derivada de 22 × 22.
- `results/`: tablas, metadatos y figuras de referencia.
- `docs/`: requisitos de datos, metodología y trazabilidad.

## Instalación

```bash
conda create -n topofusion-res python=3.11 -y
conda activate topofusion-res
# Instala PyTorch según tu GPU o CPU y después:
python -m pip install numpy scipy pandas xarray matplotlib netCDF4 cftime
python tools/author_source_integrity.py
python tools/check_reference_files.py
python tools/smoke_test_models.py
```

La versión exacta del entorno de entrenamiento no está documentada en los archivos recuperados. Python 3.11 es una sugerencia de instalación.

## Preparar los datos

La lista de NetCDF y forzamientos está en [`configs/config_model_E.json`](configs/config_model_E.json). Ubica los archivos originales autorizados bajo `inputs/raw/`, la orografía fina en `inputs/orography_regridded_64x64_cellmean_copernicus_dem.nc` y el U-Net de referencia en `inputs/unet/tas_cmip_cmip_output.nc`.

```bash
python tools/check_run_inputs.py --mode prepare
python scripts/01_prepare_dataset_model_E.py
```

Si ya dispones de los datos preprocesados del experimento, ubícalos en `data/torch_dataset_model_E_physical_v1/` y omite el paso anterior. Consulta [requisitos de datos](docs/DATA.md).

## Entrenamiento y evaluación

Los siguientes comandos se ejecutan directamente desde la raíz del repositorio, sin rutas de CEAZA ni copias adicionales:

```bash
python tools/check_run_inputs.py --mode train
MODEL_E_SEED=20260813 python scripts/30_train_phase1_full_seeded.py
MODEL_E_SEED=20260813 python scripts/31_train_phase1_no_fine_topo_seeded.py

python tools/check_run_inputs.py --mode residual
MODEL_E_SEED=20260813 python scripts/32_train_e2_full_seeded.py
MODEL_E_SEED=20260813 python scripts/33_train_e2_no_fine_topo_seeded.py

python tools/check_run_inputs.py --mode evaluate
python scripts/34_evaluate_final_seeded_ssp370.py
```

Los checkpoints se generan en `checkpoints/` y las evaluaciones en `diagnostics/`; Git excluye ambos. Las épocas definitivas están fijadas en los scripts con semilla, no en el JSON histórico.

## Análisis y límites de reproducción

Los scripts de `analysis/scripts/` incluyen bootstrap de bloques de 3/5/10 años, persistencia diaria, tendencias, extremos y ablaciones topográficas. Requieren cubos de predicciones externos que no están publicados. Los pasos y las dependencias conocidas están explicados en [reproducibilidad](docs/REPRODUCIBILITY.md); los resultados originales de referencia se encuentran en `results/`.

No se ha verificado una reproducción independiente completa desde NetCDF hasta las cifras del paper. Los datos climáticos originales, pesos entrenados y cubos grandes no forman parte del repositorio. El código se distribuye bajo la [licencia BSD de 3 cláusulas](LICENSE). La orografía derivada se mantiene para reproducibilidad, respetando las condiciones aplicables a los datos de origen.

[README en inglés](README.md) · [Datos](docs/DATA.md) · [Reproducibilidad](docs/REPRODUCIBILITY.md)
