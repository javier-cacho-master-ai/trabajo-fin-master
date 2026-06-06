"""
Transformación a formato de entrenamiento – Paso 6.

Carga los pares emparejados de espectros Gaia (VOTable, paso 5) y SDSS (FITS, paso 3)
usando la tabla de referencias depuradas del paso 4 como vínculo entre sistemas.
Los espectros SDSS se remuestrean a una rejilla lineal fija y se escriben junto a
los espectros Gaia en un único fichero HDF5.

Estructura del fichero HDF5 de salida
--------------------------------------
  /gaia_source_id          (N,)       identificadores de fuente Gaia (int64)
  /sdss_obj_id             (N,)       identificadores de objeto SDSS (int64)
  /gaia_wavelength_nm      (W_g,)     rejilla de longitudes de onda Gaia en nm
  /sdss_wavelength_aa      (W_s,)     rejilla SDSS interpolada en Ångströms
  /X                       (N, W_g)   flujo Gaia calibrado  ← entrada del modelo
  /X_err                   (N, W_g)   incertidumbre del flujo Gaia
  /y                       (N, W_s)   flujo SDSS remuestreado ← objetivo del modelo
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

from pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Tipos del dominio
# ---------------------------------------------------------------------------


class _SpecPair(NamedTuple):
    gaia_source_id: int
    sdss_obj_id: int
    gaia_flux: np.ndarray
    gaia_flux_err: np.ndarray
    sdss_flux: np.ndarray


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _build_sdss_grid(config: PipelineConfig) -> np.ndarray:
    return np.linspace(
        config.sdss_interp_wavelength_start,
        config.sdss_interp_wavelength_end,
        config.sdss_interp_n_points,
    )


def _index_gaia_spectra(spectra_vot: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_ids  = np.array(spectra_vot["source_id"],  dtype=np.int64)
    flux_matrix = np.array(spectra_vot["flux"],       dtype=np.float32)
    err_matrix  = np.array(spectra_vot["flux_error"], dtype=np.float32)
    return source_ids, flux_matrix, err_matrix


def _make_id_index(source_ids: np.ndarray) -> dict[int, int]:
    return {int(sid): i for i, sid in enumerate(source_ids)}


# ---------------------------------------------------------------------------
# Con efectos – E/S en el límite
# ---------------------------------------------------------------------------


def _load_sdss_flux(spectra_dir: Path, sdss_grid: np.ndarray, sdss_id: int) -> np.ndarray:
    """Lee un fichero FITS SDSS y remuestrea el flujo a la rejilla fija indicada."""
    with fits.open(str(spectra_dir / f"{sdss_id}.fits")) as hdul:
        data   = hdul[1].data
        loglam = data["loglam"].astype(np.float64)
        flux   = data["flux"].astype(np.float64)
    wavelength = 10.0 ** loglam
    return np.interp(sdss_grid, wavelength, flux).astype(np.float32)


def _build_pair(
    id_to_idx: dict[int, int]
  , flux_matrix: np.ndarray
  , err_matrix: np.ndarray
  , spectra_dir: Path
  , sdss_grid: np.ndarray
  , xref_row
) -> _SpecPair:
    source_id = int(xref_row["source_id"])
    sdss_id   = int(xref_row["original_ext_source_id"])
    idx       = id_to_idx[source_id]
    sdss_flux = _load_sdss_flux(spectra_dir, sdss_grid, sdss_id)
    return _SpecPair(source_id, sdss_id, flux_matrix[idx], err_matrix[idx], sdss_flux)


def _write_hdf5(
    output_path: Path
  , pairs: list[_SpecPair]
  , gaia_sampling: np.ndarray
  , sdss_grid: np.ndarray
) -> None:
    tmp = output_path.with_suffix(".tmp")
    with h5py.File(str(tmp), "w") as f:
        f.create_dataset("gaia_source_id",     data=np.array([p.gaia_source_id for p in pairs], dtype=np.int64))
        f.create_dataset("sdss_obj_id",        data=np.array([p.sdss_obj_id    for p in pairs], dtype=np.int64))
        f.create_dataset("gaia_wavelength_nm", data=gaia_sampling.astype(np.float32))
        f.create_dataset("sdss_wavelength_aa", data=sdss_grid.astype(np.float32))
        f.create_dataset("X",                  data=np.vstack([p.gaia_flux     for p in pairs]))
        f.create_dataset("X_err",              data=np.vstack([p.gaia_flux_err for p in pairs]))
        f.create_dataset("y",                  data=np.vstack([p.sdss_flux     for p in pairs]))
    tmp.rename(output_path)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def is_complete(config: PipelineConfig) -> bool:
    """Devuelve True cuando el fichero HDF5 de entrenamiento existe en disco."""
    return (config.training_data_dir / "training.h5").exists()


def run(config: PipelineConfig) -> Path:
    output_dir = config.training_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    xref_table = Table.read(str(config.pruned_dir / "gaia.ecsv"), format="ascii.ecsv")
    if not len(xref_table):
        print("[training_data] AVISO: tabla de referencias vacía — no hay pares que escribir")
        return output_dir / "training.h5"

    sampling_path = config.gaia_spectra_dir / "sampling.npy"
    if not sampling_path.exists():
        print("[training_data] AVISO: sampling.npy no encontrado — ejecuta primero el paso 5")
        return output_dir / "training.h5"

    spectra_vot   = Table.read(str(config.gaia_spectra_dir / "spectra.vot"), format="votable")
    gaia_sampling = np.load(str(sampling_path))
    sdss_grid     = _build_sdss_grid(config)

    source_ids, flux_matrix, err_matrix = _index_gaia_spectra(spectra_vot)
    id_to_idx = _make_id_index(source_ids)

    build = partial(
        _build_pair
      , id_to_idx
      , flux_matrix
      , err_matrix
      , config.sdss_spectra_dir
      , sdss_grid
    )

    with ThreadPoolExecutor(max_workers=config.sdss_spectra_max_workers) as pool:
        pairs = list(pool.map(build, xref_table))

    output_path = output_dir / "training.h5"
    _write_hdf5(output_path, pairs, gaia_sampling, sdss_grid)

    print(f"[training_data] {len(pairs)} pares  →  {output_path.name}")

    return output_path
