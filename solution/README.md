# Redes Generativas para la Mejora de la Señal Astronómica

Este proyecto explora el uso de modelos generativos para mejorar la resolución de espectros XP de Gaia hasta la calidad de SDSS. Combina la adquisición de datos de dos grandes catálogos astronómicos con una flujo de entrenamiento para aprendizaje automático.

## Descripción general

El objetivo es entrenar un modelo que tome un espectro XP calibrado de Gaia (R ≈ 50) como entrada y produzca un espectro SDSS de mayor resolución (R ≈ 2000) como salida, para el mismo objeto celeste. Los pares de entrenamiento se obtienen mediante el cruce de los catálogos Gaia DR3 y SDSS DR13.

## Estructura del proyecto

```
solution/
├── main.ipynb              cuaderno exploratorio: extracción de datos y visualización
├── main.py                 versión en script del cuaderno
├── pyproject.toml          dependencias del proyecto (uv/pip)
├── pipeline/               flujo de datos de producción (véase pipeline/README.md)
│   ├── config.py           dataclass inmutable PipelineConfig
│   ├── run.py              punto de entrada por línea de comandos
│   └── steps/              pasos numerados de el flujo (1–6)
├── services/               módulos de consulta y utilidades reutilizables
│   ├── gaia.py             utilidades TAP de Gaia
│   ├── sdss.py             análisis de FITS de SDSS y conversión de unidades
│   ├── plotting.py         utilidades para representación gráfica de espectros
│   ├── querying.py         utilidades de saneamiento de ADQL
│   └── queries/
│       ├── gaia_sdss_random_subset.adql   consulta de cruce Gaia×SDSS
│       └── sdss.adql                      consulta de metadatos SpecObj de SDSS
└── data/                   salida de el flujo (no rastreada en git)
    ├── 01_gaia_crossmatch/
    ├── 02_sdss_references/
    ├── 03_sdss_spectra/
    ├── 04_pruned/
    ├── 05_gaia_spectra/
    └── 06_training_data/training.h5
```

## Inicio rápido

**Requisitos previos:** Python 3.13+, [uv](https://github.com/astral-sh/uv)

```bash
cd solution
uv sync
```

**Ejecutar el flujo completa** (descarga ~100 000 objetos de Gaia y SDSS):

```bash
uv run python -m pipeline.run
```

**Ejecutar únicamente pasos concretos:**

```bash
uv run python -m pipeline.run --steps 1,2
```

**Reanudación tras una interrupción:** el flujo es totalmente reanudable — cada paso comprueba qué salidas existen ya en disco y omite el trabajo completado.

## Flujo de datos

El flujo descarga y alinea espectros de ambos catálogos y produce un fichero de entrenamiento en formato HDF5:

| Paso | Script | Salida |
|------|--------|--------|
| 1 | `1_gaia_crossmatch.py` | `01_gaia_crossmatch/batch_*.ecsv` — cruces Gaia×SDSS |
| 2 | `2_sdss_references.py` | `02_sdss_references/chunk_*.ecsv` — metadatos SpecObj de SDSS |
| 3 | `3_sdss_spectra.py` | `03_sdss_spectra/<bestObjID>.fits` — espectros SDSS |
| 4 | `4_prune.py` | `04_pruned/{gaia,sdss}.ecsv` — objetos con ambos espectros disponibles |
| 5 | `5_gaia_spectra.py` | `05_gaia_spectra/spectra.vot` — espectros XP calibrados de Gaia |
| 6 | `6_training_data.py` | `06_training_data/training.h5` — pares X/y alineados para ML |

Véase [pipeline/README.md](pipeline/README.md) para la documentación detallada de cada paso, el esquema HDF5 y las decisiones de diseño.

## Cuaderno exploratorio

[main.ipynb](main.ipynb) recorre el flujo completo de forma interactiva:

1. Consultar una muestra aleatoria del cruce Gaia×SDSS mediante TAP
2. Obtener los metadatos SpecObj de SDSS (placa, MJD, fibra)
3. Descargar espectros SDSS (FITS)
4. Recuperar espectros XP calibrados de Gaia mediante `gaiaxpy`
5. Visualizar y comparar espectros del mismo objeto en ambos catálogos
6. Demostrar la degradación de resolución con `iSpec/SPECTRUM` (R=100 000 → R=2 000 → R=50)

## Dependencias

Bibliotecas principales:

| Biblioteca | Propósito |
|------------|-----------|
| `astroquery` | Consultas TAP a los archivos de Gaia y SDSS |
| `gaiaxpy` | Calibración espectral XP de Gaia |
| `astropy` | Entrada/salida de tablas (ECSV, FITS, VOTable), unidades |
| `specutils` | Objetos de espectro y operaciones sobre ellos |
| `h5py` | Entrada/salida del conjunto de datos de entrenamiento HDF5 |
| `matplotlib` / `plotly` | Visualización de espectros |

Instalar todas las dependencias con `uv sync` o `pip install -e .`.

## Referencias

[IVOA Standards](https://www.ivoa.net/)  
[IVOA DataLink recommendation](https://www.ivoa.net/documents/DataLink/)

[GaiaXPy overview](https://www.cosmos.esa.int/web/gaia/gaiaxpy)  
[GaiaXPy API](https://gaiaxpy.readthedocs.io/en/latest/gaiaxpy.html)  
[Gaia Data Model](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/)  
[Gaia TAP - Astroquery](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html#datalink-service-public-and-authenticated)  
[Gaia Datalink service](https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products)  
[Descripción XP_CONTINOUS y XP_SAMPLED](https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products#datalink_serialisation)  
[Calibración del espectro](https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_specProcessing/cu5pho_ssec_specInternCal.html)

[Magnitudes SDSS](https://www.sdss4.org/dr12/algorithms/magnitudes/)  
[Astropy SDSSClass](https://astroquery.readthedocs.io/en/latest/api/astroquery.sdss.SDSSClass.html#astroquery.sdss.SDSSClass.get_spectra)  
[SDSS Overview](https://skyserver.sdss.org/dr12/en/tools/toolshome.aspx)  
[Tutoriales SDSS](https://live-sdss4org-dr13.pantheonsite.io/tutorials/)  
[SDSS - Sample queries](https://skyserver.sdss.org/dr12/en/help/docs/realquery.aspx#top)  
[SDSS Schema Browser](https://skyserver.sdss.org/dr13/en/help/browser/browser.aspx)  
[SDSS Data Model](https://data.sdss.org/datamodel/)  
[SDSS CasJobs](https://skyserver.sdss.org/CasJobs/)
