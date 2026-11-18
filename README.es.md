# TopoFusion-Res — Guía en español

**Código del artículo:** *TopoFusion-Res: Multi-resolution topographic fusion and residual learning for regional climate emulation.*

El objetivo del repositorio es publicar el **código original** y explicar cómo preparar datos, entrenar las cuatro configuraciones finales y evaluar SSP3-7.0. El modelo final es CNN multirresolución + cabeza residual E2, no SwinIR. No se incluyen NetCDF, checkpoints ni cubos grandes; hay que obtenerlos con la autorización correspondiente. Se incluye la orografía derivada GCM 22×22 (2 KB), cuyo permiso de redistribución debe verificarse antes de publicar.

El [README principal en inglés](README.md) incluye los comandos completos, rutas y limitaciones. Esta guía resume cómo comenzar.

## 1. Entorno Conda y verificaciones

```bash
cd TopoFusion-Res
conda create -n topofusion-res python=3.11 -y
conda activate topofusion-res
# Instalar PyTorch apropiado para CPU/GPU desde sus instrucciones oficiales.
python -m pip install numpy pandas scipy xarray matplotlib netCDF4 cftime
python tools/author_source_integrity.py
python tools/smoke_test_models.py
```

**Precaución:** Python 3.11 es una propuesta de instalación; las versiones exactas del entorno original aún no están confirmadas. La prueba sintética no reproduce el paper.

## 2. Preparar una copia local portable

La implementación original usa rutas antiguas de `/workspace`. El siguiente comando crea `run/` con **solo las rutas adaptadas**, preservando los scripts originales y el manifiesto de hashes:

```bash
python tools/make_portable_copy.py --dest run
```

## 3. Datos y preparación

Coloca los NetCDF, archivos GHG, la máscara y la orografía regional en las rutas que explica el [README en inglés](README.md#5-supply-climate-inputs-not-hosted-in-this-repository). Comprueba:

```bash
python tools/check_run_inputs.py --root run --mode prepare
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/01_prepare_dataset_model_E.py
```

Si cuentas con los arreglos `.npy` oficiales ya preparados, puedes omitir la preparación y ubicarlos en `run/climate_swinir_experiments/model_E_cmip_swinunet_topography/data/torch_dataset_model_E_physical_v1/`.

## 4. Entrenamiento y evaluación final

```bash
python tools/check_run_inputs.py --root run --mode train
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/30_train_phase1_full_seeded.py
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/31_train_phase1_no_fine_topo_seeded.py

python tools/check_run_inputs.py --root run --mode residual
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/32_train_e2_full_seeded.py
MODEL_E_SEED=20260813 python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/33_train_e2_no_fine_topo_seeded.py

python tools/check_run_inputs.py --root run --mode evaluate
python run/climate_swinir_experiments/model_E_cmip_swinunet_topography/scripts/34_evaluate_final_seeded_ssp370.py
```

Primera etapa: **40 épocas**. Segunda etapa E2: **12 épocas**. Semilla: **20260813**. Se requiere el NetCDF de referencia de U-Net para la evaluación comparativa.

## 5. Reproducir las tablas y figuras

Las salidas de referencia originales se encuentran en `reference_results/`. Los scripts estadísticos finales están en `paper_validation_final_seed20260813/scripts/`, pero muchos requieren los **cubos de predicciones `.dat`** calculados anteriormente. Recuperamos el script que genera los dos cubos E2 finales (`34_evaluate_topography_ablation_ssp370.py`) y el que genera el cubo base (`36_create_phase1_full_seeded_cube.py`), además de su dependencia `17_train_model_E2_residual_head_base_eval.py`. Su ejecución requiere antes `rcm.dat`, `mask.dat`, `cube_metadata.json` y los checkpoints correspondientes. El origen de los cubos RCM y U-Net y de los metadatos históricos sigue sin resolverse por completo; consulta [Cubos de predicción](docs/PREDICTION_CUBES.md). Por tanto, el repositorio documenta un flujo de trabajo basado en el código original, **sin afirmar que cualquier tercero ya puede reproducir todas las métricas sin esos datos**.

Para el detalle técnico, disponibilidad y limitaciones, usa el [README completo](README.md) y [Reproducibilidad](docs/REPRODUCIBILITY.md).
