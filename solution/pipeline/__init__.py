"""
Flujo de extracción de datos astronómicos.

Flujo de datos canónico
-----------------------
    PipelineConfig                          ← configuración inmutable
        │
        ├─ 1. steps.gaia_crossmatch   →  01_gaia_crossmatch/batch_*.ecsv
        ├─ 2. steps.sdss_references   →  02_sdss_references/chunk_*.ecsv
        ├─ 3. steps.sdss_spectra      →  03_sdss_spectra/<bestObjID>.fits
        ├─ 4. steps.prune             →  04_pruned/{gaia,sdss}.ecsv          (tabla de cruce)
        ├─ 5. steps.gaia_spectra      →  05_gaia_spectra/{spectra.vot, sampling.npy}
        └─ 6. steps.training_data     →  06_training_data/training.h5        (pares X/y para ML)

Cada paso está completamente desacoplado: sólo lee del disco y sólo escribe en disco,
de modo que cualquier subconjunto puede ejecutarse o volver a ejecutarse de forma independiente.
"""

from pipeline.config import PipelineConfig
from pipeline import steps

__all__ = ["PipelineConfig", "steps"]
