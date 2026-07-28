"""
Depuración de referencias – Paso 4.

Lee la tabla de cruce Gaia-SDSS del paso 2 y conserva únicamente las filas cuyo
espectro SDSS fue descargado correctamente en el paso 3.  Guarda dos vistas del
resultado bajo el directorio pruned/:

  gaia.ecsv  — columnas Gaia (source_id, original_ext_source_id, ...)
                usada por los pasos 5 y 6
  sdss.ecsv  — columnas SDSS (plate, mjd, fiberID, redshift)
                referencia para auditoría
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

from pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _load_tables(directory: Path, pattern: str) -> Table:
    files = sorted(directory.glob(pattern))
    if not files:
        return Table()
    tables    = [Table.read(str(f), format="ascii.ecsv") for f in files]
    non_empty = [t for t in tables if len(t)]
    return vstack(non_empty) if non_empty else Table()


def _downloaded_ids(spectra_dir: Path) -> frozenset[str]:
    return frozenset(p.stem for p in spectra_dir.glob("*.fits"))


def _prune(table: Table, id_column: str, keep_ids: frozenset[str]) -> Table:
    if not len(table):
        return table
    mask = np.isin(table[id_column].astype(str), list(keep_ids))
    return table[mask]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def is_complete(config: PipelineConfig) -> bool:
    """Devuelve True cuando ambos ficheros depurados existen en disco."""
    return (
        (config.pruned_dir / "gaia.ecsv").exists()
        and (config.pruned_dir / "sdss.ecsv").exists()
    )


def run(config: PipelineConfig) -> tuple[Path, Path]:
    output_dir = config.pruned_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    crossmatch = _load_tables(config.gaia_crossmatch_dir, "chunk_*.ecsv")
    downloaded = _downloaded_ids(config.sdss_spectra_dir)

    print(f"[prune] {len(downloaded)} espectros SDSS en disco")

    if not len(crossmatch):
        print("[prune] AVISO: tabla de cruce vacía — ejecuta primero los pasos 1 y 2")
    if not downloaded:
        print("[prune] AVISO: sin espectros FITS en disco — ejecuta primero el paso 3")

    pruned = _prune(crossmatch, "original_ext_source_id", downloaded)

    # Vista Gaia: todas las columnas Gaia + original_ext_source_id (clave de unión con SDSS)
    gaia_cols = [
        "source_id", "ra", "dec"
      , "phot_bp_mean_mag", "phot_rp_mean_mag"
      , "phot_bp_mean_flux_over_error", "phot_rp_mean_flux_over_error"
      , "astrometric_excess_noise", "bp_rp"
      , "teff_gspphot", "logg_gspphot", "mh_gspphot", "vbroad", "alphafe_gspspec"
      , "parallax", "phot_g_mean_mag"
      , "original_ext_source_id", "angular_distance"
    ]
    sdss_cols = ["original_ext_source_id", "plate", "mjd", "fiberID", "redshift"]

    existing_gaia = [c for c in gaia_cols if c in pruned.colnames]
    existing_sdss = [c for c in sdss_cols if c in pruned.colnames]

    pruned_gaia = pruned[existing_gaia] if len(pruned) else Table()
    pruned_sdss = pruned[existing_sdss] if len(pruned) else Table()

    gaia_path = output_dir / "gaia.ecsv"
    sdss_path = output_dir / "sdss.ecsv"

    pruned_gaia.write(str(gaia_path), format="ascii.ecsv", overwrite=True)
    pruned_sdss.write(str(sdss_path), format="ascii.ecsv", overwrite=True)

    print(f"[prune] {len(pruned_gaia)} pares Gaia-SDSS  →  {gaia_path.name}")

    return gaia_path, sdss_path
