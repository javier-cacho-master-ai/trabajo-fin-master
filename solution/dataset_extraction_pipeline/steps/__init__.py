"""
Registro de pasos.

Los ficheros se denominan 1_*.py … 5_*.py de modo que su orden sea inequívoco
en cualquier listado de directorios.  Como los identificadores de Python no pueden
comenzar por un dígito, cada módulo se carga explícitamente mediante importlib
y se expone bajo un alias legible.

REGISTRY asocia número de paso → (etiqueta, run_fn, is_complete_fn).
El ejecutor utiliza is_complete para omitir los pasos cuyas salidas ya están en disco.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


# ---------------------------------------------------------------------------
# Cargador de módulos para ficheros con nombre numérico
# ---------------------------------------------------------------------------


def _load(filename: str) -> ModuleType:
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None, f"No se puede localizar el módulo: {filename}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sdss_specobj    = _load("1_sdss_specobj.py")
gaia_crossmatch = _load("2_gaia_crossmatch.py")
sdss_spectra    = _load("3_sdss_spectra.py")
prune           = _load("4_prune.py")
gaia_spectra    = _load("5_gaia_spectra.py")
training_data   = _load("6_training_data.py")


# ---------------------------------------------------------------------------
# Registro ordenado  {número_de_paso: (etiqueta, run_fn, is_complete_fn)}
# ---------------------------------------------------------------------------


REGISTRY: dict[int, tuple[str, Any, Any]] = {
    1: ("Catálogo espectroscópico SDSS",            sdss_specobj.run,    sdss_specobj.is_complete)
  , 2: ("Cruce inverso Gaia",                       gaia_crossmatch.run, gaia_crossmatch.is_complete)
  , 3: ("Descarga de espectros SDSS",               sdss_spectra.run,    sdss_spectra.is_complete)
  , 4: ("Depuración de referencias",                prune.run,           prune.is_complete)
  , 5: ("Espectros calibrados de Gaia",             gaia_spectra.run,    gaia_spectra.is_complete)
  , 6: ("Transformación a datos de entrenamiento",  training_data.run,   training_data.is_complete)
}

__all__ = [
    "sdss_specobj"
  , "gaia_crossmatch"
  , "sdss_spectra"
  , "prune"
  , "gaia_spectra"
  , "training_data"
  , "REGISTRY"
]
