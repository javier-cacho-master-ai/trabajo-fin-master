# Flujo de extracción de datos astronómicos

Flujo modular en Python que descarga y empareja espectros de los catálogos
**Gaia DR3** y **SDSS DR13**, y los transforma en un conjunto de datos listo para
entrenar un modelo de aprendizaje automático.

## Estructura de ficheros

```text
pipeline/
├── config.py          configuración inmutable de toda el flujo
├── run.py             ejecutor por línea de comandos
├── __init__.py        paquete; documenta el flujo de datos canónico
└── steps/
    ├── __init__.py    registro de pasos (REGISTRY)
    ├── 1_gaia_crossmatch.py   extracción del cruce Gaia × SDSS
    ├── 2_sdss_references.py   búsqueda de metadatos SpecObj en SDSS
    ├── 3_sdss_spectra.py      descarga de espectros FITS desde SDSS
    ├── 4_prune.py             depuración: elimina objetos sin espectro descargado
    ├── 5_gaia_spectra.py      extracción de espectros XP calibrados de Gaia
    └── 6_training_data.py     transformación a pares X/y en formato HDF5
```

## Cómo ejecutar el flujo

Los comandos deben ejecutarse desde el directorio `../` (es decir, `solution/`),
donde se encuentra el fichero `pyproject.toml`. Se usa `uv run` para garantizar
que el entorno virtual y las dependencias de `uv.lock` están sincronizados antes
de la ejecución.

```bash
# Ejecución completa (pasos 1–6)
uv run python -m pipeline.run

# Sólo los pasos 1 y 2
uv run python -m pipeline.run --steps 1,2

# Forzar la re-ejecución del paso 3 aunque sus salidas ya existan en disco
uv run python -m pipeline.run --steps 3 --force

# Ajuste de paralelismo y tamaños de lote
uv run python -m pipeline.run \
    --gaia-end 500000 \
    --gaia-batch-size 20000 \
    --gaia-workers 8 \
    --sdss-wave-n 1800
```

Todos los parámetros tienen valores por omisión razonables declarados en
`PipelineConfig`; la línea de comandos los sobreescribe sin modificar el código.

## Flujo de datos

```text
01_gaia_crossmatch/batch_*.ecsv
        │  cruce Gaia×SDSS por lotes paralelos
        ▼
02_sdss_references/chunk_*.ecsv
        │  metadatos SpecObj agrupados en fragmentos ≤ 250 IDs
        ▼
03_sdss_spectra/<bestObjID>.fits
        │  un fichero FITS por objeto con el espectro óptico SDSS
        ▼
04_pruned/{gaia,sdss}.ecsv
        │  tablas depuradas; sólo objetos con espectro descargado
        │  vínculo entre sistemas: original_ext_source_id == bestObjID
        ▼
05_gaia_spectra/{spectra.vot, sampling.npy}
        │  espectros XP calibrados en VOTable; rejilla de longitudes de onda en .npy
        ▼
06_training_data/training.h5
           pares emparejados X (Gaia) / y (SDSS) listos para ML
```

### Vínculo entre sistemas

Cada fila `i` en `training.h5` representa el mismo objeto celeste:

```text
gaia.ecsv[i].source_id                → /gaia_source_id[i] en training.h5
gaia.ecsv[i].original_ext_source_id   → nombre del .fits en 03_sdss_spectra/
                                       → /sdss_obj_id[i] en training.h5
```

## Estructura del fichero HDF5 de entrenamiento

| Dataset              | Forma     | Descripción                                   |
|----------------------|-----------|-----------------------------------------------|
| `/gaia_source_id`    | (N,)      | Identificador de fuente Gaia (int64)          |
| `/sdss_obj_id`       | (N,)      | Identificador de objeto SDSS (int64)          |
| `/gaia_wavelength_nm`| (W_g,)    | Rejilla de longitudes de onda Gaia en nm      |
| `/sdss_wavelength_aa`| (W_s,)    | Rejilla SDSS interpolada en Ångströms         |
| `/X`                 | (N, W_g)  | Flujo Gaia calibrado — entrada del modelo     |
| `/X_err`             | (N, W_g)  | Incertidumbre del flujo Gaia                  |
| `/y`                 | (N, W_s)  | Flujo SDSS remuestreado — objetivo del modelo |

El índice de fila `i` es coherente en todos los datasets: `X[i]` y `y[i]` son
siempre el mismo objeto celeste.

## Decisiones de diseño

### Estilo funcional y declarativo

Los bucles imperativos se han sustituido por `filter`, `map` y `functools.partial`,
de forma que cada función hace exactamente una cosa y no tiene estado interno.
`partial` se emplea para fijar los parámetros de configuración y obtener funciones
de un único argumento aptas para `pool.map`, evitando la necesidad de lambdas
o clases auxiliares.

```python
fetch   = partial(_fetch_and_save, output_dir, query_template)
pending = filter(partial(_is_pending, output_dir), all_batches)

with ThreadPoolExecutor(max_workers=config.gaia_max_workers) as pool:
    list(pool.map(fetch, pending))
```

### Reanudabilidad

Cada paso persiste su resultado en disco de forma inmediata, objeto a objeto o
lote a lote, antes de continuar con el siguiente.  Si la ejecución se interrumpe,
la siguiente llamada comprueba qué ficheros ya existen mediante `is_complete` o
la función `_is_pending` correspondiente, y retoma el trabajo desde el primer
elemento ausente sin repetir el trabajo ya hecho.

Este invariante se mantiene de forma estricta en los pasos con acceso a red
(pasos 1, 2, 3 y 5), donde un fallo parcial es más probable y más costoso.

### Desacoplamiento total entre pasos

Cada paso sólo lee del disco y sólo escribe en disco.  No existe ninguna
dependencia en memoria entre pasos: el paso 2 no conoce la existencia del
paso 1 salvo por los ficheros que éste ha dejado en `01_gaia_crossmatch/`.
Esto permite ejecutar, depurar o re-ejecutar cualquier subconjunto de pasos
de forma completamente independiente.

### Configuración como tipo de datos inmutable

`PipelineConfig` es un `dataclass(frozen=True)`: sus campos no pueden
modificarse tras la construcción, lo que garantiza que todos los pasos que
comparten la misma instancia de configuración ven exactamente los mismos
valores.  Los directorios de salida son propiedades calculadas derivadas de
`data_dir`, por lo que mover todo el directorio de datos es tan sencillo
como cambiar un único campo.

`prepare_dirs()` crea todos los directorios de salida de forma idempotente
al inicio de la ejecución, por lo que ningún paso tiene que preocuparse por
la existencia previa de su directorio de destino.

### Paralelismo con `ThreadPoolExecutor`

Las operaciones de red (consultas TAP de Gaia y SDSS, descarga de FITS,
llamadas a `gaiaxpy.calibrate`) son limitadas por E/S, no por CPU, por lo que
`ThreadPoolExecutor` es más adecuado que `ProcessPoolExecutor`.  El número de
hilos es configurable por paso para permitir ajustar el paralelismo según
los límites de los servicios remotos.

### Nomenclatura numérica de los módulos de paso

Los ficheros de paso se llaman `1_gaia_crossmatch.py`, `2_sdss_references.py`,
etc., para que su orden sea inequívoco en cualquier listado de directorios.
Como Python no permite importar módulos cuyos nombres comiencen por un dígito,
el registro `steps/__init__.py` los carga explícitamente mediante
`importlib.util.spec_from_file_location`.

### Formatos de fichero

| Directorio            | Formato   | Motivo                                                       |
|-----------------------|-----------|--------------------------------------------------------------|
| `01_gaia_crossmatch/` | ECSV      | Preserva tipos de columna y unidades; legible con astropy    |
| `02_sdss_references/` | ECSV      | Ídem                                                         |
| `03_sdss_spectra/`    | FITS      | Formato nativo de `SDSS.get_spectra`; no hay conversión      |
| `04_pruned/`          | ECSV      | Tabla de cruce ligera; fácil de inspeccionar manualmente     |
| `05_gaia_spectra/`    | VOTable   | Estándar astronómico para tablas con columnas vectoriales    |
| `06_training_data/`   | HDF5      | Acceso aleatorio por índice; compatible con PyTorch/TF/NumPy |

El fichero HDF5 es el único formato orientado al entrenamiento de modelos.
Los formatos astronómicos anteriores (ECSV, FITS, VOTable) se conservan íntegros
para facilitar la trazabilidad y la verificación de los datos brutos.
